"""
Unit tests for the unified _process_file() method.

Tests the shared file processing logic used by both scan and watch pipelines.
Uses mocking to avoid network calls for model downloads.
"""

import pytest
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock

from mneme.config import VaultConfig
from mneme.indexer.indexer import Indexer
from mneme.indexer.parallel_types import ProcessResult


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
    )


@pytest.fixture
def indexer(vault_config: VaultConfig, mock_embedder) -> Indexer:
    """Create an indexer with test config and mocked embedder."""
    return Indexer(vault_config)


class TestProcessMarkdownFile:
    """Test _process_file() with markdown files."""

    def test_process_simple_markdown(self, indexer: Indexer, temp_vault: Path):
        """Test processing a simple markdown file."""
        md_file = temp_vault / "simple.md"
        md_file.write_text("# Test Heading\n\nSome content here.")

        result = indexer._process_file(md_file)

        assert isinstance(result, ProcessResult)
        assert result.error is None
        assert result.skipped is False
        assert result.rel_path == "simple.md"
        assert result.file_type == "markdown"
        assert len(result.chunks) > 0
        assert result.doc_id != ""
        assert result.vault_id == indexer.vault_id

    def test_process_markdown_with_wikilinks(self, indexer: Indexer, temp_vault: Path):
        """Test processing markdown with wikilinks extracts links."""
        md_file = temp_vault / "with_links.md"
        md_file.write_text(
            "# Notes\n\nLinked to [[other-note]] and [[Project/Task]]."
        )

        result = indexer._process_file(md_file)

        assert result.error is None
        assert len(result.links) >= 2
        link_targets = [link[0] for link in result.links]
        assert "other-note" in link_targets
        assert "Project/Task" in link_targets

    def test_process_markdown_with_image_reference(self, indexer: Indexer, temp_vault: Path):
        """Test processing markdown with image reference creates image task."""
        # Create an image file
        img_dir = temp_vault / "images"
        img_dir.mkdir()
        img_file = img_dir / "test.png"
        # Write a minimal PNG (1x1 pixel)
        img_file.write_bytes(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )

        md_file = temp_vault / "with_image.md"
        md_file.write_text(f"# Note\n\n![Alt text](images/test.png)")

        result = indexer._process_file(md_file)

        assert result.error is None
        assert len(result.image_tasks) >= 1
        assert result.image_tasks[0].task_type == "reference"

    def test_process_markdown_enriched_metadata(self, indexer: Indexer, temp_vault: Path):
        """Test that processed result includes enriched metadata."""
        md_file = temp_vault / "metadata_test.md"
        md_file.write_text("# Test\n\nContent for metadata.")

        result = indexer._process_file(md_file)

        assert result.error is None
        assert result.full_path == str(md_file)
        assert result.vault_root == str(temp_vault)
        assert result.file_name == "metadata_test.md"
        assert result.file_extension == ".md"
        assert result.modified_at != ""

        # Check chunk metadata
        assert len(result.chunks) > 0
        chunk = result.chunks[0]
        assert chunk.metadata["full_path"] == str(md_file)
        assert chunk.metadata["file_type"] == "markdown"


class TestProcessSkippedFiles:
    """Test _process_file() for files that should be skipped."""

    def test_skip_nonexistent_file(self, indexer: Indexer, temp_vault: Path):
        """Test that nonexistent files are skipped."""
        nonexistent = temp_vault / "does_not_exist.md"

        result = indexer._process_file(nonexistent)

        assert result.skipped is True
        assert result.error is None

    def test_skip_directory(self, indexer: Indexer, temp_vault: Path):
        """Test that directories are skipped."""
        subdir = temp_vault / "subdir"
        subdir.mkdir()

        result = indexer._process_file(subdir)

        assert result.skipped is True

    def test_skip_file_without_suffix(self, indexer: Indexer, temp_vault: Path):
        """Test that files without suffix are skipped."""
        no_suffix = temp_vault / "README"
        no_suffix.write_text("No extension file")

        result = indexer._process_file(no_suffix)

        assert result.skipped is True

    def test_skip_unsupported_extension(self, indexer: Indexer, temp_vault: Path):
        """Test that unsupported extensions are skipped."""
        unsupported = temp_vault / "data.json"
        unsupported.write_text('{"key": "value"}')

        result = indexer._process_file(unsupported)

        assert result.skipped is True


class TestProcessErrorHandling:
    """Test _process_file() error handling."""

    def test_error_on_unreadable_file(self, indexer: Indexer, temp_vault: Path):
        """Test error handling for unreadable files."""
        import os
        # Skip this test on systems that don't support chmod properly
        if os.name == 'nt':
            pytest.skip("chmod not effective on Windows")

        # Create a file and make it unreadable
        unreadable = temp_vault / "unreadable.md"
        unreadable.write_text("# Content")
        unreadable.chmod(0o000)

        try:
            result = indexer._process_file(unreadable)
            # Should return error result, not raise
            assert result.error is not None
            assert result.skipped is False
        finally:
            # Restore permissions for cleanup
            unreadable.chmod(0o644)


class TestProcessDeterministicIds:
    """Test that _process_file() generates deterministic IDs."""

    def test_same_file_same_doc_id(self, indexer: Indexer, temp_vault: Path):
        """Test that same file always gets same doc_id."""
        md_file = temp_vault / "consistent.md"
        md_file.write_text("# Test\n\nContent here.")

        result1 = indexer._process_file(md_file)
        result2 = indexer._process_file(md_file)

        assert result1.doc_id == result2.doc_id

    def test_same_content_same_chunk_ids(self, indexer: Indexer, temp_vault: Path):
        """Test that same content generates same chunk IDs."""
        md_file = temp_vault / "chunks.md"
        md_file.write_text("# Section 1\n\nContent A.\n\n# Section 2\n\nContent B.")

        result1 = indexer._process_file(md_file)
        result2 = indexer._process_file(md_file)

        assert len(result1.chunks) == len(result2.chunks)
        for c1, c2 in zip(result1.chunks, result2.chunks):
            assert c1.chunk_id == c2.chunk_id

    def test_different_files_different_doc_ids(self, indexer: Indexer, temp_vault: Path):
        """Test that different files get different doc_ids."""
        file1 = temp_vault / "file1.md"
        file2 = temp_vault / "file2.md"
        file1.write_text("# File 1")
        file2.write_text("# File 2")

        result1 = indexer._process_file(file1)
        result2 = indexer._process_file(file2)

        assert result1.doc_id != result2.doc_id


class TestProcessThreadSafety:
    """Test that _process_file() is thread-safe."""

    def test_concurrent_processing(self, indexer: Indexer, temp_vault: Path):
        """Test processing multiple files concurrently."""
        # Create multiple test files
        num_files = 10
        files = []
        for i in range(num_files):
            f = temp_vault / f"concurrent_{i}.md"
            f.write_text(f"# File {i}\n\nContent for file {i}.")
            files.append(f)

        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(indexer._process_file, f): f for f in files}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))

        assert len(errors) == 0, f"Errors during concurrent processing: {errors}"
        assert len(results) == num_files
        for result in results:
            assert result.error is None
            assert result.skipped is False

    def test_concurrent_same_file(self, indexer: Indexer, temp_vault: Path):
        """Test processing the same file concurrently returns consistent results."""
        md_file = temp_vault / "shared.md"
        md_file.write_text("# Shared\n\nContent that will be read concurrently.")

        results = []
        num_threads = 5

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(indexer._process_file, md_file) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        # All results should have the same doc_id
        doc_ids = {r.doc_id for r in results}
        assert len(doc_ids) == 1, "Concurrent reads produced different doc_ids"

        # All results should have the same chunks
        chunk_counts = {len(r.chunks) for r in results}
        assert len(chunk_counts) == 1, "Concurrent reads produced different chunk counts"


class TestProcessObsidianUri:
    """Test Obsidian URI generation."""

    def test_obsidian_uri_with_vault_name(self, temp_vault: Path, temp_index_dir: Path, mock_embedder):
        """Test that Obsidian URI is generated when vault_name is set."""
        config = VaultConfig(
            vault_root=temp_vault,
            index_dir=temp_index_dir,
            vault_name="test-vault",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_device="cpu",
            image_analysis_provider="off",
        )
        indexer = Indexer(config)

        md_file = temp_vault / "note.md"
        md_file.write_text("# Test")

        result = indexer._process_file(md_file)

        assert result.obsidian_uri != ""
        assert "obsidian://open" in result.obsidian_uri
        assert "vault=test-vault" in result.obsidian_uri
        assert "file=note.md" in result.obsidian_uri

    def test_no_obsidian_uri_without_vault_name(self, indexer: Indexer, temp_vault: Path):
        """Test that Obsidian URI is empty when vault_name is not set."""
        md_file = temp_vault / "note.md"
        md_file.write_text("# Test")

        result = indexer._process_file(md_file)

        # Default config has no vault_name
        assert result.obsidian_uri == ""


class TestProcessFileTypes:
    """Test _process_file() with different file types."""

    def test_process_nested_markdown(self, indexer: Indexer, temp_vault: Path):
        """Test processing markdown in nested directory."""
        nested_dir = temp_vault / "projects" / "2024"
        nested_dir.mkdir(parents=True)
        md_file = nested_dir / "project_notes.md"
        md_file.write_text("# Project Notes\n\nDetailed content.")

        result = indexer._process_file(md_file)

        assert result.error is None
        assert result.rel_path == "projects/2024/project_notes.md"
        assert result.file_name == "project_notes.md"

    def test_process_markdown_with_special_chars(self, indexer: Indexer, temp_vault: Path):
        """Test processing file with special characters in name."""
        md_file = temp_vault / "note-with-dashes_and_underscores.md"
        md_file.write_text("# Special Name\n\nContent here.")

        result = indexer._process_file(md_file)

        assert result.error is None
        assert result.rel_path == "note-with-dashes_and_underscores.md"

    def test_process_empty_markdown(self, indexer: Indexer, temp_vault: Path):
        """Test processing empty markdown file."""
        md_file = temp_vault / "empty.md"
        md_file.write_text("")

        result = indexer._process_file(md_file)

        # Should not error, but may have no chunks
        assert result.error is None
        assert result.skipped is False
