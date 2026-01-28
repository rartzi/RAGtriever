"""
Unit tests for BatchCollector class.

Tests the thread-safe batch collection logic used by watch_batched().
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from mneme.indexer.indexer import BatchCollector
from mneme.indexer.queue import Job


class TestBatchCollectorBasic:
    """Test basic BatchCollector functionality."""

    def test_add_job_returns_none_when_not_full(self):
        """Test that add_job returns None when batch is not full."""
        collector = BatchCollector(max_batch_size=5, batch_timeout_seconds=60.0)

        job = Job(kind="upsert", rel_path="file1.md")
        result = collector.add_job(job)

        assert result is None
        assert collector.pending_count() == 1

    def test_add_job_returns_batch_when_full(self):
        """Test that add_job returns batch when size threshold reached."""
        collector = BatchCollector(max_batch_size=3, batch_timeout_seconds=60.0)

        # Add jobs up to threshold
        collector.add_job(Job(kind="upsert", rel_path="file1.md"))
        collector.add_job(Job(kind="upsert", rel_path="file2.md"))

        # Third job should trigger batch
        batch = collector.add_job(Job(kind="upsert", rel_path="file3.md"))

        assert batch is not None
        assert len(batch) == 3
        assert collector.pending_count() == 0

    def test_flush_returns_pending_jobs(self):
        """Test that flush returns all pending jobs."""
        collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=60.0)

        collector.add_job(Job(kind="upsert", rel_path="file1.md"))
        collector.add_job(Job(kind="delete", rel_path="file2.md"))

        batch = collector.flush()

        assert batch is not None
        assert len(batch) == 2
        assert collector.pending_count() == 0

    def test_flush_returns_none_when_empty(self):
        """Test that flush returns None when no pending jobs."""
        collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=60.0)

        batch = collector.flush()

        assert batch is None

    def test_flush_if_timeout_returns_none_before_timeout(self):
        """Test that flush_if_timeout returns None before timeout expires."""
        collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=10.0)

        collector.add_job(Job(kind="upsert", rel_path="file1.md"))

        # Immediately check - should not trigger
        batch = collector.flush_if_timeout()

        assert batch is None
        assert collector.pending_count() == 1

    def test_flush_if_timeout_returns_batch_after_timeout(self):
        """Test that flush_if_timeout returns batch after timeout expires."""
        collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=0.1)

        collector.add_job(Job(kind="upsert", rel_path="file1.md"))

        # Wait for timeout
        time.sleep(0.15)

        batch = collector.flush_if_timeout()

        assert batch is not None
        assert len(batch) == 1
        assert collector.pending_count() == 0

    def test_flush_if_timeout_returns_none_when_empty(self):
        """Test that flush_if_timeout returns None when no pending jobs."""
        collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=0.1)

        batch = collector.flush_if_timeout()

        assert batch is None


class TestBatchCollectorJobTypes:
    """Test BatchCollector with different job types."""

    def test_collects_mixed_job_types(self):
        """Test that collector handles upsert, delete, and move jobs."""
        collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=60.0)

        collector.add_job(Job(kind="upsert", rel_path="new.md"))
        collector.add_job(Job(kind="delete", rel_path="old.md"))
        collector.add_job(Job(kind="move", rel_path="before.md", new_rel_path="after.md"))

        batch = collector.flush()

        assert batch is not None
        assert len(batch) == 3

        kinds = [j.kind for j in batch]
        assert "upsert" in kinds
        assert "delete" in kinds
        assert "move" in kinds

    def test_preserves_job_order(self):
        """Test that jobs are returned in order added."""
        collector = BatchCollector(max_batch_size=5, batch_timeout_seconds=60.0)

        paths = ["a.md", "b.md", "c.md", "d.md", "e.md"]
        batch = None
        for p in paths:
            result = collector.add_job(Job(kind="upsert", rel_path=p))
            if result:
                batch = result

        # Batch should have been returned on the 5th add
        assert batch is not None
        result_paths = [j.rel_path for j in batch]
        assert result_paths == paths


class TestBatchCollectorThreadSafety:
    """Test BatchCollector thread safety."""

    def test_concurrent_add_job(self):
        """Test concurrent add_job calls are thread-safe."""
        collector = BatchCollector(max_batch_size=100, batch_timeout_seconds=60.0)
        num_jobs = 50
        num_threads = 10

        errors = []
        batches_returned = []

        def add_jobs(thread_id: int):
            for i in range(num_jobs // num_threads):
                job = Job(kind="upsert", rel_path=f"thread{thread_id}_file{i}.md")
                try:
                    batch = collector.add_job(job)
                    if batch:
                        batches_returned.append(batch)
                except Exception as e:
                    errors.append(str(e))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(add_jobs, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors during concurrent add: {errors}"

        # Flush remaining and count total
        remaining = collector.flush()
        total_jobs = sum(len(b) for b in batches_returned)
        if remaining:
            total_jobs += len(remaining)

        assert total_jobs == num_jobs

    def test_concurrent_add_and_flush(self):
        """Test concurrent add_job and flush_if_timeout calls."""
        collector = BatchCollector(max_batch_size=100, batch_timeout_seconds=0.05)
        num_jobs = 30
        stop_flag = threading.Event()

        all_jobs_added = []
        all_batches = []
        errors = []

        def add_jobs():
            for i in range(num_jobs):
                job = Job(kind="upsert", rel_path=f"file{i}.md")
                all_jobs_added.append(job)
                try:
                    batch = collector.add_job(job)
                    if batch:
                        all_batches.append(batch)
                except Exception as e:
                    errors.append(str(e))
                time.sleep(0.01)  # Small delay between adds
            stop_flag.set()

        def check_timeout():
            while not stop_flag.is_set():
                try:
                    batch = collector.flush_if_timeout()
                    if batch:
                        all_batches.append(batch)
                except Exception as e:
                    errors.append(str(e))
                time.sleep(0.02)

        # Start both threads
        add_thread = threading.Thread(target=add_jobs)
        timeout_thread = threading.Thread(target=check_timeout)

        add_thread.start()
        timeout_thread.start()

        add_thread.join()
        timeout_thread.join(timeout=1.0)

        # Flush any remaining
        remaining = collector.flush()
        if remaining:
            all_batches.append(remaining)

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"

        # Count total jobs in batches
        total_in_batches = sum(len(b) for b in all_batches)
        assert total_in_batches == num_jobs


class TestBatchCollectorEdgeCases:
    """Test BatchCollector edge cases."""

    def test_batch_size_one(self):
        """Test collector with batch_size=1 returns immediately."""
        collector = BatchCollector(max_batch_size=1, batch_timeout_seconds=60.0)

        batch = collector.add_job(Job(kind="upsert", rel_path="file.md"))

        assert batch is not None
        assert len(batch) == 1

    def test_very_short_timeout(self):
        """Test collector with very short timeout."""
        collector = BatchCollector(max_batch_size=100, batch_timeout_seconds=0.001)

        collector.add_job(Job(kind="upsert", rel_path="file.md"))
        time.sleep(0.01)

        batch = collector.flush_if_timeout()

        assert batch is not None

    def test_multiple_flush_calls(self):
        """Test that multiple flush calls don't cause issues."""
        collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=60.0)

        collector.add_job(Job(kind="upsert", rel_path="file.md"))

        batch1 = collector.flush()
        batch2 = collector.flush()
        batch3 = collector.flush()

        assert batch1 is not None
        assert len(batch1) == 1
        assert batch2 is None
        assert batch3 is None

    def test_pending_count_accuracy(self):
        """Test that pending_count is accurate during operations."""
        collector = BatchCollector(max_batch_size=5, batch_timeout_seconds=60.0)

        assert collector.pending_count() == 0

        collector.add_job(Job(kind="upsert", rel_path="file1.md"))
        assert collector.pending_count() == 1

        collector.add_job(Job(kind="upsert", rel_path="file2.md"))
        assert collector.pending_count() == 2

        collector.flush()
        assert collector.pending_count() == 0
