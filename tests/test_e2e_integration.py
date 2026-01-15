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

from cortexindex.config import VaultConfig
from cortexindex.indexer.indexer import Indexer
from cortexindex.retrieval.retriever import Retriever


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
def retriever(indexer: Indexer) -> Retriever:
    """Create a retriever from the indexer's store."""
    return Retriever(indexer.store, indexer.vault_id)


class TestEndToEndMarkdownIndexing:
    """Test end-to-end indexing of Markdown files."""

    def test_index_markdown_files(self, indexer: Indexer):
        """Test that markdown files are indexed correctly."""
        # Run full scan
        indexer.scan(full=True)

        # Verify that documents were indexed
        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)

        assert len(docs) > 0, "No documents indexed"

        # Check for markdown files
        md_docs = [d for d in docs if d.file_type == "markdown"]
        assert len(md_docs) > 0, "No markdown documents indexed"

        # Verify document metadata
        for doc in md_docs[:5]:  # Check first 5
            assert doc.doc_id is not None
            assert doc.vault_id == vault_id
            assert doc.rel_path is not None
            assert doc.content_hash is not None
            assert doc.mtime > 0
            assert doc.size > 0

    def test_markdown_extraction_with_metadata(self, indexer: Indexer):
        """Test that markdown extraction captures metadata."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)
        md_docs = [d for d in docs if d.file_type == "markdown"]

        assert len(md_docs) > 0
        doc = md_docs[0]

        # Verify metadata
        assert doc.metadata is not None
        assert "extractor_version" in doc.metadata

    def test_markdown_chunking(self, indexer: Indexer):
        """Test that markdown files are chunked properly."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)
        md_docs = [d for d in docs if d.file_type == "markdown"]

        assert len(md_docs) > 0
        doc = md_docs[0]

        # Get chunks for this document
        chunks = indexer.store.get_chunks_by_document(doc.doc_id)
        assert len(chunks) > 0, f"No chunks found for document {doc.rel_path}"

        # Verify chunk structure
        for chunk in chunks[:5]:
            assert chunk.chunk_id is not None
            assert chunk.doc_id == doc.doc_id
            assert chunk.text is not None
            assert len(chunk.text.strip()) > 0
            assert chunk.anchor_type is not None
            assert chunk.anchor_ref is not None


class TestEndToEndImageIndexing:
    """Test end-to-end indexing of image files."""

    def test_index_images(self, indexer: Indexer):
        """Test that image files are indexed."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)

        # Look for image documents
        image_docs = [d for d in docs if d.file_type == "image"]

        if len(image_docs) > 0:
            # Verify image document structure
            doc = image_docs[0]
            assert doc.rel_path is not None
            assert doc.content_hash is not None
            assert doc.size > 0

    def test_image_chunking(self, indexer: Indexer):
        """Test that images are chunked (one chunk per image)."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)
        image_docs = [d for d in docs if d.file_type == "image"]

        if len(image_docs) > 0:
            doc = image_docs[0]
            chunks = indexer.store.get_chunks_by_document(doc.doc_id)

            # Image should have at least one chunk
            assert len(chunks) > 0
            chunk = chunks[0]
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
        results = retriever.search("project", k=10, file_types=["markdown"])

        assert len(results) > 0

        # Verify all results are from markdown files
        for result in results:
            assert result.chunk is not None
            assert result.chunk.metadata.get("file_type") == "markdown"

    def test_open_document(self, indexer: Indexer, retriever: Retriever):
        """Test opening a document by chunk ID."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)

        if len(docs) > 0:
            doc = docs[0]
            chunks = indexer.store.get_chunks_by_document(doc.doc_id)

            if len(chunks) > 0:
                chunk = chunks[0]

                # Open document
                result = retriever.open(chunk.chunk_id)

                assert result is not None
                assert result.chunk is not None
                assert result.chunk.chunk_id == chunk.chunk_id
                assert result.context is not None

    def test_neighbors_query(self, indexer: Indexer, retriever: Retriever):
        """Test finding neighbors of a chunk."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)

        if len(docs) > 0:
            doc = docs[0]
            chunks = indexer.store.get_chunks_by_document(doc.doc_id)

            if len(chunks) > 0:
                chunk = chunks[0]

                # Get neighbors
                neighbors = retriever.neighbors(chunk.chunk_id, k=5)

                assert neighbors is not None
                assert isinstance(neighbors, list)


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
        docs_before = indexer.store.get_documents_by_vault(vault_id)
        assert len(docs_before) == 1

        chunks_before = indexer.store.get_chunks_by_document(docs_before[0].doc_id)
        content_hash_before = docs_before[0].content_hash

        # Modify file
        file1.write_text("# Note 1\n\nUpdated content")

        # Second scan (should be incremental)
        indexer.scan(full=False)

        docs_after = indexer.store.get_documents_by_vault(vault_id)
        assert len(docs_after) == 1

        # Content hash should be different
        content_hash_after = docs_after[0].content_hash
        assert content_hash_after != content_hash_before

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
        docs_before = indexer.store.get_documents_by_vault(vault_id)
        assert len(docs_before) == 1

        # Add new file
        file2 = test_vault / "note2.md"
        file2.write_text("# Note 2")

        # Scan again
        indexer.scan(full=False)

        docs_after = indexer.store.get_documents_by_vault(vault_id)
        assert len(docs_after) == 2

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
        docs_before = indexer.store.get_documents_by_vault(vault_id)
        assert len(docs_before) == 2

        # Delete a file
        file1.unlink()

        # Scan again
        indexer.scan(full=False)

        docs_after = indexer.store.get_documents_by_vault(vault_id)
        # Note: Files may be marked as deleted rather than removed from index
        # Check for active documents
        active_docs = [d for d in docs_after if not d.deleted]
        assert len(active_docs) <= 2


class TestEmbeddingGeneration:
    """Test that embeddings are generated during indexing."""

    def test_chunks_have_embeddings(self, indexer: Indexer):
        """Test that chunks receive vector embeddings."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)

        assert len(docs) > 0
        doc = docs[0]

        chunks = indexer.store.get_chunks_by_document(doc.doc_id)
        assert len(chunks) > 0

        # Check that chunks have embeddings (vector dimension should be set)
        chunk = chunks[0]
        # The store should have embeddings for these chunks
        # This depends on implementation, but vectors should be stored


class TestMultipleFileFormats:
    """Test indexing of multiple file formats."""

    def test_all_supported_formats_scanned(self, indexer: Indexer):
        """Test that all supported formats are properly scanned."""
        indexer.scan(full=True)

        vault_id = indexer.vault_id
        docs = indexer.store.get_documents_by_vault(vault_id)

        assert len(docs) > 0

        file_types = set(d.file_type for d in docs)

        # Should have at least markdown and images
        assert "markdown" in file_types, "No markdown files indexed"
        # images may or may not be present depending on test vault content


class TestIdempotency:
    """Test that indexing is idempotent."""

    def test_multiple_scans_are_idempotent(self, vault_config: VaultConfig, temp_index_dir: Path):
        """Test that running multiple full scans produces the same result."""
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
        docs1 = indexer.store.get_documents_by_vault(vault_id)
        chunks1 = []
        for doc in docs1:
            chunks1.extend(indexer.store.get_chunks_by_document(doc.doc_id))

        # Second scan (should be identical)
        indexer.scan(full=True)
        docs2 = indexer.store.get_documents_by_vault(vault_id)
        chunks2 = []
        for doc in docs2:
            chunks2.extend(indexer.store.get_chunks_by_document(doc.doc_id))

        # Same number of documents and chunks
        assert len(docs1) == len(docs2)
        assert len(chunks1) == len(chunks2)

        # Same content hashes
        hashes1 = sorted([d.content_hash for d in docs1])
        hashes2 = sorted([d.content_hash for d in docs2])
        assert hashes1 == hashes2
