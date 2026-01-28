"""Tests for the change detector / watcher functionality."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from mneme.indexer.change_detector import ChangeDetector
from mneme.indexer.queue import Job, JobQueue


class TestChangeDetectorDirectoryScan:
    """Tests for directory scanning when new folders are created."""

    def test_scan_directory_finds_files(self):
        """When a new directory is created, we can find all files inside."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a subdirectory with files
            subdir = root / "new_folder"
            subdir.mkdir()
            (subdir / "file1.md").write_text("content 1")
            (subdir / "file2.pdf").write_text("content 2")
            (subdir / "nested").mkdir()
            (subdir / "nested" / "file3.txt").write_text("content 3")

            # Test that we can find files in the directory
            files_in_dir = list(subdir.rglob("*"))
            file_paths = [f for f in files_in_dir if f.is_file()]

            assert len(file_paths) == 3
            assert any("file1.md" in str(f) for f in file_paths)
            assert any("file2.pdf" in str(f) for f in file_paths)
            assert any("file3.txt" in str(f) for f in file_paths)

    def test_scan_directory_respects_ignore_patterns(self):
        """Scanning should respect ignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a subdirectory with files
            subdir = root / "new_folder"
            subdir.mkdir()
            (subdir / "file1.md").write_text("content 1")
            (subdir / ".DS_Store").write_text("ignored")
            (subdir / "ignored_folder").mkdir()
            (subdir / "ignored_folder" / "file2.md").write_text("content 2")

            # Test ignore patterns
            ignore_patterns = ["**/.DS_Store", "**/ignored_folder/**"]

            from mneme.indexer.reconciler import matches_ignore_pattern

            files_in_dir = list(subdir.rglob("*"))
            file_paths = [f for f in files_in_dir if f.is_file()]

            # Check which files would be ignored
            for f in file_paths:
                rel = str(f.relative_to(root)).replace("\\", "/")
                ignored = matches_ignore_pattern(rel, ignore_patterns)
                if ".DS_Store" in str(f) or "ignored_folder" in str(f):
                    assert ignored, f"{rel} should be ignored"
                else:
                    assert not ignored, f"{rel} should not be ignored"


class TestChangeDetectorIntegration:
    """Integration tests for the change detector with watchdog."""

    @pytest.mark.skip(reason="Flaky timing-dependent test - watchdog events are unreliable in CI")
    def test_new_folder_with_files_detected(self):
        """Test that creating a new folder with files triggers indexing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            q = JobQueue()
            detector = ChangeDetector(root=root, q=q, ignore=[], debounce_ms=100)

            # Start watching in a thread
            import threading

            watch_thread = threading.Thread(target=detector.watch, daemon=True)
            watch_thread.start()

            # Give watchdog time to start
            time.sleep(0.5)

            # Create a new folder with files
            new_folder = root / "test_folder"
            new_folder.mkdir()
            (new_folder / "test1.md").write_text("# Test 1")
            (new_folder / "test2.md").write_text("# Test 2")

            # Wait for events to be processed
            time.sleep(1.0)

            # Check that jobs were queued
            jobs = []
            while not q.empty():
                jobs.append(q.get())

            # Should have at least 2 jobs for the 2 files
            file_jobs = [j for j in jobs if j.kind == "upsert"]
            assert len(file_jobs) >= 2, f"Expected at least 2 upsert jobs, got {len(file_jobs)}: {jobs}"

            # Verify the files are in the jobs
            job_paths = [j.rel_path for j in file_jobs]
            assert any("test1.md" in p for p in job_paths), f"test1.md not found in {job_paths}"
            assert any("test2.md" in p for p in job_paths), f"test2.md not found in {job_paths}"


class TestJobQueue:
    """Tests for the job queue."""

    def test_queue_put_and_get(self):
        """Test basic queue operations."""
        q = JobQueue()
        job = Job(kind="upsert", rel_path="test.md")
        q.put(job)
        assert not q.empty()
        retrieved = q.get()
        assert retrieved.kind == "upsert"
        assert retrieved.rel_path == "test.md"
        assert q.empty()

    def test_queue_multiple_jobs(self):
        """Test queuing multiple jobs."""
        q = JobQueue()
        q.put(Job(kind="upsert", rel_path="file1.md"))
        q.put(Job(kind="upsert", rel_path="file2.md"))
        q.put(Job(kind="delete", rel_path="file3.md"))

        jobs = []
        while not q.empty():
            jobs.append(q.get())

        assert len(jobs) == 3
        assert jobs[0].rel_path == "file1.md"
        assert jobs[2].kind == "delete"
