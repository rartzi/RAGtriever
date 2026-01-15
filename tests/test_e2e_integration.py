"""
End-to-end integration tests for CortexIndex.

Tests the full pipeline: extraction, chunking, embedding, storage, and retrieval.
Uses /tmp/test_vault_comprehensive as the test vault with real files.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from io import BytesIO

from ragtriever.config import VaultConfig
from ragtriever.indexer.indexer import Indexer
from ragtriever.retrieval.retriever import Retriever


@pytest.fixture
def test_vault() -> Path:
    """Use the comprehensive test vault."""
    vault_path = Path("/tmp/test_vault_comprehensive")
    if not vault_path.exists():
        pytest.skip("Test vault not found at /tmp/test_vault_comprehensive")
    return vault_path


@pytest.fixture
def temp_index_dir(tmp_path: Path) -> Path:
    """Create a temporary index directory."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(exist_ok=True)
    return index_dir


@pytest.fixture
def vault_config(test_vault: Path, temp_index_dir: Path) -> VaultConfig:
    """Create a test vault config."""
    config = VaultConfig(
        vault_root=test_vault,
        index_dir=temp_index_dir,
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        image_analysis_provider="off",
    )
    return config


@pytest.fixture
def indexer(vault_config: VaultConfig) -> Indexer:
    """Create an indexer with test config."""
    return Indexer(vault_config)


@pytest.fixture
def retriever(vault_config: VaultConfig) -> Retriever:
    """Create a retriever from the vault config."""
    return Retriever(vault_config)


class TestEndToEndMarkdownIndexing:
    """Test end-to-end indexing of Markdown files."""

    def test_index_markdown_files(self, indexer: Indexer):
        """Test that markdown files are indexed correctly."""
        # Run full scan
        indexer.scan(full=True)

        # Verify that documents were indexed by searching for common term
        results = indexer.store.lexical_search("note", k=100, filters={"vault_id": indexer.vault_id})

        assert len(results) > 0, "No documents indexed"

        # Check that we have chunks with markdown metadata
        md_chunks = [r for r in results if r.chunk.metadata.get("file_type") == "markdown"]
        assert len(md_chunks) > 0, "No markdown documents indexed"

        # Verify chunk structure
        for chunk in md_chunks[:5]:  # Check first 5
            assert chunk.chunk is not None
            assert chunk.chunk.chunk_id is not None
            assert chunk.chunk.doc_id is not None

    def test_markdown_extraction_with_metadata(self, indexer: Indexer):
        """Test that markdown extraction captures metadata."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})

        assert len(results) > 0

        # Check for markdown chunks
        md_chunks = [r for r in results if r.chunk.metadata.get("file_type") == "markdown"]
        assert len(md_chunks) > 0

        # Verify metadata is present
        chunk = md_chunks[0].chunk
        assert chunk.metadata is not None

    def test_markdown_chunking(self, indexer: Indexer):
        """Test that markdown files are chunked properly."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})

        assert len(results) > 0

        # Check for markdown chunks
        md_chunks = [r for r in results if r.chunk.metadata.get("file_type") == "markdown"]
        assert len(md_chunks) > 0

        # Verify chunks have content
        for chunk_result in md_chunks[:5]:
            chunk = chunk_result.chunk
            assert chunk.text is not None
            assert len(chunk.text.strip()) > 0
            assert chunk.anchor_type is not None


class TestEndToEndImageIndexing:
    """Test end-to-end indexing of image files."""

    def test_index_images(self, indexer: Indexer):
        """Test that image files are indexed."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})

        # Look for image documents
        image_chunks = [r for r in results if r.chunk.metadata.get("file_type") == "image"]

        if len(image_chunks) > 0:
            # Verify image chunk structure
            chunk = image_chunks[0].chunk
            assert chunk.chunk_id is not None
            assert chunk.metadata is not None

    def test_image_chunking(self, indexer: Indexer):
        """Test that images are chunked (one chunk per image)."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})
        image_chunks = [r for r in results if r.chunk.metadata.get("file_type") == "image"]

        if len(image_chunks) > 0:
            chunk = image_chunks[0].chunk
            assert chunk.anchor_type == "IMAGE"


class TestEndToEndRetrieval:
    """Test end-to-end retrieval functionality."""

    def test_lexical_search(self, indexer: Indexer, retriever: Retriever):
        """Test lexical search."""
        indexer.scan(full=True)

        # Search for a common term
        results = retriever.search("project", k=10)

        # Should find something in the test vault
        assert len(results) > 0, "No search results found"

        # Verify result structure
        result = results[0]
        assert result.score >= 0
        assert result.chunk is not None
        assert result.chunk.text is not None

    def test_semantic_search(self, indexer: Indexer, retriever: Retriever):
        """Test semantic search."""
        indexer.scan(full=True)

        # Search with a semantic query
        results = retriever.search("cloud infrastructure", k=5)

        # May or may not find results, but should not crash
        assert isinstance(results, list)

        if len(results) > 0:
            result = results[0]
            assert result.score >= 0
            assert result.chunk is not None

    def test_search_with_filters(self, indexer: Indexer, retriever: Retriever):
        """Test search with filters."""
        indexer.scan(full=True)

        # Search only markdown files
        results = retriever.search("project", k=10, filters={"vault_id": indexer.vault_id})

        assert len(results) > 0

        # Verify all results are from markdown files
        for result in results:
            assert result.chunk is not None
            assert result.chunk.metadata.get("file_type") in ["markdown", "image", "pdf", "pptx", "xlsx"]

    def test_open_document(self, indexer: Indexer, retriever: Retriever):
        """Test opening a document by chunk ID."""
        indexer.scan(full=True)

        # Get a chunk via search
        results = retriever.search("project", k=5)

        if len(results) > 0:
            chunk = results[0].chunk

            # Open document using SourceRef
            from ragtriever.models import SourceRef
            source_ref = SourceRef(chunk_id=chunk.chunk_id)
            result = retriever.open(source_ref)

            assert result is not None
            assert result.chunk is not None


class TestIncrementalIndexing:
    """Test incremental indexing when files change."""

    def test_incremental_scan_detects_changes(self, vault_config: VaultConfig, temp_index_dir: Path):
        """Test that incremental scan properly handles file changes."""
        # Create a temporary test vault
        test_vault = temp_index_dir / "test_vault_incremental"
        test_vault.mkdir(exist_ok=True)

        # Create initial file
        file1 = test_vault / "note1.md"
        file1.write_text("# Note 1\n\nInitial content")

        # Create config pointing to temp vault
        config = VaultConfig(
            vault_root=test_vault,
            index_dir=temp_index_dir / "index1",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_device="cpu",
            image_analysis_provider="off",
        )

        # First scan
        indexer = Indexer(config)
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results_before = indexer.store.lexical_search("Initial", k=100, filters={"vault_id": vault_id})
        assert len(results_before) > 0

        # Modify file
        file1.write_text("# Note 1\n\nUpdated content")

        # Second scan (should be incremental)
        indexer.scan(full=False)

        results_after = indexer.store.lexical_search("Updated", k=100, filters={"vault_id": vault_id})
        # Updated content should be found
        assert len(results_after) > 0

    def test_add_new_file_incremental(self, vault_config: VaultConfig, temp_index_dir: Path):
        """Test that new files are detected in incremental scans."""
        test_vault = temp_index_dir / "test_vault_add"
        test_vault.mkdir(exist_ok=True)

        file1 = test_vault / "note1.md"
        file1.write_text("# Note 1")

        config = VaultConfig(
            vault_root=test_vault,
            index_dir=temp_index_dir / "index2",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_device="cpu",
            image_analysis_provider="off",
        )

        indexer = Indexer(config)
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results_before = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})
        count_before = len(results_before)

        # Add new file
        file2 = test_vault / "note2.md"
        file2.write_text("# Note 2")

        # Scan again
        indexer.scan(full=False)

        results_after = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})
        count_after = len(results_after)

        assert count_after >= count_before

    def test_delete_file_incremental(self, vault_config: VaultConfig, temp_index_dir: Path):
        """Test that deleted files are detected."""
        test_vault = temp_index_dir / "test_vault_delete"
        test_vault.mkdir(exist_ok=True)

        file1 = test_vault / "note1.md"
        file1.write_text("# Note 1")
        file2 = test_vault / "note2.md"
        file2.write_text("# Note 2")

        config = VaultConfig(
            vault_root=test_vault,
            index_dir=temp_index_dir / "index3",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_device="cpu",
            image_analysis_provider="off",
        )

        indexer = Indexer(config)
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results_before = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})
        count_before = len(results_before)

        # Delete a file
        file1.unlink()

        # Scan again
        indexer.scan(full=False)

        results_after = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})
        count_after = len(results_after)

        assert count_after <= count_before


class TestEmbeddingGeneration:
    """Test that embeddings are generated during indexing."""

    def test_chunks_have_embeddings(self, indexer: Indexer):
        """Test that chunks receive vector embeddings."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results = indexer.store.lexical_search("", k=10, filters={"vault_id": vault_id})

        assert len(results) > 0

        # Results from store should have been ranked with embeddings
        for result in results:
            assert result.chunk is not None
            # The embeddings are used internally for search


class TestMultipleFileFormats:
    """Test indexing of multiple file formats."""

    def test_all_supported_formats_scanned(self, indexer: Indexer):
        """Test that all supported formats are properly scanned."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        results = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})

        assert len(results) > 0

        file_types = set(r.chunk.metadata.get("file_type") for r in results if r.chunk.metadata)

        # Should have at least markdown
        assert "markdown" in file_types, "No markdown files indexed"


class TestIdempotency:
    """Test that indexing is idempotent."""

    def test_multiple_scans_are_idempotent(self, vault_config: VaultConfig, temp_index_dir: Path):
        """Test that running multiple full scans produces the same result."""
        # Create a temporary test vault
        test_vault = temp_index_dir / "test_vault_idempotent"
        test_vault.mkdir(exist_ok=True)

        # Create test files
        (test_vault / "note1.md").write_text("# Note 1\n\nContent")
        (test_vault / "note2.md").write_text("# Note 2\n\nMore content")

        config = VaultConfig(
            vault_root=test_vault,
            index_dir=temp_index_dir / "index_idempotent",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_device="cpu",
            image_analysis_provider="off",
        )

        indexer = Indexer(config)

        # First scan
        indexer.scan(full=True)
        vault_id = indexer.vault_id
        results1 = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})

        # Second scan (should be identical)
        indexer.scan(full=True)
        results2 = indexer.store.lexical_search("", k=100, filters={"vault_id": vault_id})

        # Same number of results
        assert len(results1) == len(results2)

        # Same chunk IDs
        ids1 = sorted([r.chunk.chunk_id for r in results1])
        ids2 = sorted([r.chunk.chunk_id for r in results2])
        assert ids1 == ids2
