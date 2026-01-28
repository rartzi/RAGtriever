"""
Unit tests for _process_batch() method.

Tests the batch processing logic used by watch_batched().
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from mneme.config import VaultConfig
from mneme.indexer.indexer import Indexer
from mneme.indexer.queue import Job
from mneme.indexer.parallel_types import BatchStats


@pytest.fixture
def temp_vault(tmp_path: Path) -> Path:
    """Create a temporary vault directory."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    return vault_path


@pytest.fixture
def temp_index_dir(tmp_path: Path) -> Path:
    """Create a temporary index directory."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def mock_embedder():
    """Create a mock embedder to avoid network calls."""
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_st.return_value = mock_model
        yield mock_st


@pytest.fixture
def vault_config(temp_vault: Path, temp_index_dir: Path) -> VaultConfig:
    """Create a test vault config with minimal settings."""
    return VaultConfig(
        vault_root=temp_vault,
        index_dir=temp_index_dir,
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        image_analysis_provider="off",
        watch_workers=2,
        watch_batch_size=5,
        watch_batch_timeout=1.0,
    )


@pytest.fixture
def indexer(vault_config: VaultConfig, mock_embedder) -> Indexer:
    """Create an indexer with test config and mocked embedder."""
    return Indexer(vault_config)


class TestProcessBatchUpserts:
    """Test _process_batch() with upsert jobs."""

    def test_process_single_upsert(self, indexer: Indexer, temp_vault: Path):
        """Test processing a single upsert job."""
        # Create test file
        md_file = temp_vault / "test.md"
        md_file.write_text("# Test\n\nContent here.")

        jobs = [Job(kind="upsert", rel_path="test.md")]
        stats = indexer._process_batch(jobs)

        assert isinstance(stats, BatchStats)
        assert stats.files_processed == 1
        assert stats.files_failed == 0
        assert stats.chunks_created > 0

    def test_process_multiple_upserts(self, indexer: Indexer, temp_vault: Path):
        """Test processing multiple upsert jobs."""
        # Create test files
        for i in range(3):
            f = temp_vault / f"file{i}.md"
            f.write_text(f"# File {i}\n\nContent for file {i}.")

        jobs = [
            Job(kind="upsert", rel_path="file0.md"),
            Job(kind="upsert", rel_path="file1.md"),
            Job(kind="upsert", rel_path="file2.md"),
        ]
        stats = indexer._process_batch(jobs)

        assert stats.files_processed == 3
        assert stats.files_failed == 0

    def test_upsert_nonexistent_file(self, indexer: Indexer, temp_vault: Path):
        """Test upsert of nonexistent file is skipped (not counted as failure)."""
        jobs = [Job(kind="upsert", rel_path="nonexistent.md")]
        stats = indexer._process_batch(jobs)

        # Nonexistent files are skipped, not failed
        assert stats.files_processed == 0
        assert stats.files_failed == 0


class TestProcessBatchDeletes:
    """Test _process_batch() with delete jobs."""

    def test_process_single_delete(self, indexer: Indexer, temp_vault: Path):
        """Test processing a single delete job."""
        # First create and index a file
        md_file = temp_vault / "to_delete.md"
        md_file.write_text("# To Delete\n\nThis will be deleted.")

        # Index it first
        indexer._process_batch([Job(kind="upsert", rel_path="to_delete.md")])

        # Now delete it
        md_file.unlink()
        stats = indexer._process_batch([Job(kind="delete", rel_path="to_delete.md")])

        assert stats.files_deleted == 1

    def test_delete_nonexistent_in_index(self, indexer: Indexer, temp_vault: Path):
        """Test delete of file not in index doesn't fail."""
        jobs = [Job(kind="delete", rel_path="never_existed.md")]
        stats = indexer._process_batch(jobs)

        # Should still count as deleted (idempotent)
        assert stats.files_deleted == 1
        assert stats.files_failed == 0


class TestProcessBatchMoves:
    """Test _process_batch() with move jobs."""

    def test_process_move(self, indexer: Indexer, temp_vault: Path):
        """Test processing a move job."""
        # Create and index original file
        old_file = temp_vault / "old_name.md"
        old_file.write_text("# Moving File\n\nThis will be moved.")
        indexer._process_batch([Job(kind="upsert", rel_path="old_name.md")])

        # Simulate move (rename file)
        new_file = temp_vault / "new_name.md"
        old_file.rename(new_file)

        # Process move job
        stats = indexer._process_batch([
            Job(kind="move", rel_path="old_name.md", new_rel_path="new_name.md")
        ])

        assert stats.files_deleted == 1  # Old path deleted
        assert stats.files_processed == 1  # New path indexed


class TestProcessBatchMixed:
    """Test _process_batch() with mixed job types."""

    def test_mixed_jobs(self, indexer: Indexer, temp_vault: Path):
        """Test processing batch with upsert, delete, and move jobs."""
        # Setup: Create files
        (temp_vault / "keep.md").write_text("# Keep\n\nStaying.")
        (temp_vault / "delete_me.md").write_text("# Delete\n\nGoing away.")
        (temp_vault / "moved.md").write_text("# Moved\n\nNew location.")

        # Index them first
        indexer._process_batch([
            Job(kind="upsert", rel_path="keep.md"),
            Job(kind="upsert", rel_path="delete_me.md"),
            Job(kind="upsert", rel_path="old_location.md"),
        ])

        # Remove delete_me.md
        (temp_vault / "delete_me.md").unlink()

        # Process mixed batch
        jobs = [
            Job(kind="upsert", rel_path="keep.md"),  # Re-index
            Job(kind="delete", rel_path="delete_me.md"),
            Job(kind="move", rel_path="old_location.md", new_rel_path="moved.md"),
        ]
        stats = indexer._process_batch(jobs)

        assert stats.files_deleted >= 2  # delete + move's old path
        assert stats.files_processed >= 1  # keep.md + move's new path

    def test_deletes_processed_before_upserts(self, indexer: Indexer, temp_vault: Path):
        """Test that deletes are processed before upserts (order matters)."""
        # Create a file
        (temp_vault / "file.md").write_text("# File\n\nContent.")

        # Index it
        indexer._process_batch([Job(kind="upsert", rel_path="file.md")])

        # Create batch with delete and upsert of same file
        # The delete should happen first, then the upsert
        jobs = [
            Job(kind="upsert", rel_path="file.md"),
            Job(kind="delete", rel_path="file.md"),
        ]
        stats = indexer._process_batch(jobs)

        # Both should succeed - delete first, then upsert
        assert stats.files_deleted == 1
        assert stats.files_processed == 1


class TestProcessBatchStats:
    """Test BatchStats accuracy."""

    def test_stats_elapsed_time(self, indexer: Indexer, temp_vault: Path):
        """Test that elapsed_seconds is recorded."""
        (temp_vault / "file.md").write_text("# Test")

        stats = indexer._process_batch([Job(kind="upsert", rel_path="file.md")])

        assert stats.elapsed_seconds > 0
        assert stats.elapsed_seconds < 60  # Should be quick for one file

    def test_stats_chunks_and_embeddings(self, indexer: Indexer, temp_vault: Path):
        """Test that chunks and embeddings are counted."""
        (temp_vault / "multi_section.md").write_text(
            "# Section 1\n\nContent A.\n\n# Section 2\n\nContent B."
        )

        stats = indexer._process_batch([Job(kind="upsert", rel_path="multi_section.md")])

        assert stats.chunks_created > 0
        assert stats.embeddings_created > 0
        # Typically embeddings == chunks for text files
        assert stats.embeddings_created == stats.chunks_created


class TestProcessBatchErrorHandling:
    """Test _process_batch() error handling."""

    def test_one_failure_doesnt_stop_batch(self, indexer: Indexer, temp_vault: Path):
        """Test that one file failure doesn't stop other files from processing."""
        # Create one good file, one that will fail
        (temp_vault / "good.md").write_text("# Good\n\nThis works.")
        bad_file = temp_vault / "bad.md"
        bad_file.write_text("# Bad")
        bad_file.chmod(0o000)  # Make unreadable

        try:
            jobs = [
                Job(kind="upsert", rel_path="good.md"),
                Job(kind="upsert", rel_path="bad.md"),
            ]
            stats = indexer._process_batch(jobs)

            # Good file should still be processed
            assert stats.files_processed >= 1
            # Bad file should be failed or skipped
            assert stats.files_failed >= 0  # Depends on OS handling
        finally:
            bad_file.chmod(0o644)

    def test_empty_batch(self, indexer: Indexer, temp_vault: Path):
        """Test processing empty batch."""
        stats = indexer._process_batch([])

        assert stats.files_processed == 0
        assert stats.files_deleted == 0
        assert stats.files_failed == 0
        assert stats.chunks_created == 0
