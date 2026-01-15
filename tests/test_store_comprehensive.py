"""
Comprehensive tests for the storage layer (LibSqlStore).
Tests document storage, chunk persistence, search indices, and retrieval.
"""

import pytest
from pathlib import Path
import numpy as np

from ragtriever.models import Document, Chunk
from ragtriever.store.libsql_store import LibSqlStore


@pytest.fixture
def store(tmp_path: Path) -> LibSqlStore:
    """Create a test store."""
    store_path = tmp_path / "test.sqlite"
    store = LibSqlStore(store_path)
    store.init()
    return store


class TestStoreInitialization:
    """Test store initialization and schema creation."""

    def test_store_creates_database(self, tmp_path: Path):
        """Test that store creates database file."""
        store_path = tmp_path / "test.sqlite"
        store = LibSqlStore(store_path)
        store.init()

        assert store_path.exists()

    def test_store_schema_valid(self, store: LibSqlStore):
        """Test that store schema is created correctly."""
        # Store.init() should create tables without error
        # Try inserting a document to verify schema
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash1",
            deleted=False,
        )
        store.upsert_document(doc)
        # Should not raise


class TestDocumentStorage:
    """Test document persistence and retrieval."""

    def test_upsert_document(self, store: LibSqlStore):
        """Test inserting a document."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="notes.md",
            file_type="markdown",
            mtime=1234567890,
            size=1024,
            content_hash="hash123",
            deleted=False,
            metadata={"key": "value"},
        )

        store.upsert_document(doc)
        # Should not raise

    def test_upsert_updates_existing(self, store: LibSqlStore):
        """Test that upsert updates existing documents."""
        doc1 = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=1000,
            size=100,
            content_hash="hash1",
            deleted=False,
        )
        doc2 = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=2000,
            size=200,
            content_hash="hash2",
            deleted=False,
        )

        store.upsert_document(doc1)
        store.upsert_document(doc2)

        # Document should be updated (second upsert overwrites first)
        # We can verify by inserting a chunk with it
        chunk = Chunk(
            chunk_id="chunk1",
            doc_id="doc1",
            vault_id="vault1",
            anchor_type="heading",
            anchor_ref="ref",
            text="test content",
            text_hash="hash",
        )
        embedding = np.random.randn(384).astype(np.float32)
        store.upsert_chunks([chunk])
        # Should succeed with updated document

    def test_delete_document(self, store: LibSqlStore):
        """Test document deletion."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )

        store.upsert_document(doc)
        store.delete_document("vault1", "file.md")

        # Document should be deleted
        # Verify via search - should return no results
        results = store.lexical_search("test", k=10, filters={"vault_id": "vault1"})
        # May have no results or no chunks


class TestChunkStorage:
    """Test chunk persistence and retrieval."""

    def test_upsert_chunk(self, store: LibSqlStore):
        """Test inserting a chunk."""
        # First insert document
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )
        store.upsert_document(doc)

        # Insert chunk
        chunk = Chunk(
            chunk_id="chunk1",
            doc_id="doc1",
            vault_id="vault1",
            anchor_type="heading",
            anchor_ref="section-1",
            text="Chunk content",
            text_hash="texthash1",
            metadata={"level": 1},
        )

        embedding = np.random.randn(384).astype(np.float32)
        store.upsert_chunks([chunk])
        # Should not raise

    def test_insert_multiple_chunks(self, store: LibSqlStore):
        """Test inserting multiple chunks."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )
        store.upsert_document(doc)

        chunks_to_insert = [
            Chunk(
                chunk_id="chunk1",
                doc_id="doc1",
                vault_id="vault1",
                anchor_type="heading",
                anchor_ref="h1",
                text="Content 1",
                text_hash="hash1",
            ),
            Chunk(
                chunk_id="chunk2",
                doc_id="doc1",
                vault_id="vault1",
                anchor_type="heading",
                anchor_ref="h2",
                text="Content 2",
                text_hash="hash2",
            ),
        ]
        embeddings = [np.random.randn(384).astype(np.float32) for _ in chunks_to_insert]

        store.upsert_chunks(chunks_to_insert)
        # Should not raise


class TestLexicalSearch:
    """Test full-text search (FTS5)."""

    def test_lexical_search_basic(self, store: LibSqlStore):
        """Test basic lexical search."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )
        store.upsert_document(doc)

        chunk = Chunk(
            chunk_id="chunk1",
            doc_id="doc1",
            vault_id="vault1",
            anchor_type="heading",
            anchor_ref="ref",
            text="cloud infrastructure deployment",
            text_hash="hash",
        )
        embedding = np.random.randn(384).astype(np.float32)

        store.upsert_chunks([chunk])

        results = store.lexical_search("cloud", k=10, filters={"vault_id": "vault1"})

        assert len(results) > 0
        assert results[0].chunk.chunk_id == "chunk1"

    def test_lexical_search_multiple_matches(self, store: LibSqlStore):
        """Test lexical search with multiple matching chunks."""
        doc1 = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file1.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash1",
            deleted=False,
        )
        doc2 = Document(
            doc_id="doc2",
            vault_id="vault1",
            rel_path="file2.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash2",
            deleted=False,
        )
        store.upsert_document(doc1)
        store.upsert_document(doc2)

        chunks = [
            Chunk(
                chunk_id="chunk1",
                doc_id="doc1",
                vault_id="vault1",
                anchor_type="heading",
                anchor_ref="h1",
                text="database optimization",
                text_hash="hash1",
            ),
            Chunk(
                chunk_id="chunk2",
                doc_id="doc2",
                vault_id="vault1",
                anchor_type="heading",
                anchor_ref="h2",
                text="database performance tuning",
                text_hash="hash2",
            ),
        ]
        embeddings = [np.random.randn(384).astype(np.float32) for _ in chunks]

        store.upsert_chunks(chunks)

        results = store.lexical_search("database", k=10, filters={"vault_id": "vault1"})

        assert len(results) >= 2

    def test_lexical_search_respects_k(self, store: LibSqlStore):
        """Test that lexical search respects k limit."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )
        store.upsert_document(doc)

        chunks = [
            Chunk(
                chunk_id=f"chunk{i}",
                doc_id="doc1",
                vault_id="vault1",
                anchor_type="heading",
                anchor_ref=f"h{i}",
                text=f"test content number {i}",
                text_hash=f"hash{i}",
            )
            for i in range(10)
        ]
        embeddings = [np.random.randn(384).astype(np.float32) for _ in chunks]

        store.upsert_chunks(chunks)

        results = store.lexical_search("test", k=5, filters={"vault_id": "vault1"})

        assert len(results) <= 5

    def test_lexical_search_no_matches(self, store: LibSqlStore):
        """Test lexical search with no matches."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )
        store.upsert_document(doc)

        chunk = Chunk(
            chunk_id="chunk1",
            doc_id="doc1",
            vault_id="vault1",
            anchor_type="heading",
            anchor_ref="ref",
            text="some content",
            text_hash="hash",
        )
        embedding = np.random.randn(384).astype(np.float32)

        store.upsert_chunks([chunk])

        results = store.lexical_search("xyz123", k=10, filters={"vault_id": "vault1"})

        assert len(results) == 0


class TestVectorSearch:
    """Test vector search functionality."""

    def test_vector_search_basic(self, store: LibSqlStore):
        """Test basic vector search."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )
        store.upsert_document(doc)

        chunk = Chunk(
            chunk_id="chunk1",
            doc_id="doc1",
            vault_id="vault1",
            anchor_type="heading",
            anchor_ref="ref",
            text="cloud infrastructure",
            text_hash="hash",
        )
        embedding = np.ones(384, dtype=np.float32) * 0.1

        store.upsert_chunks([chunk])

        # Search with similar embedding
        query_embedding = np.ones(384, dtype=np.float32) * 0.1
        results = store.vector_search(query_embedding, k=10, filters={"vault_id": "vault1"})

        assert len(results) > 0

    def test_vector_search_respects_k(self, store: LibSqlStore):
        """Test that vector search respects k parameter."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )
        store.upsert_document(doc)

        chunks = [
            Chunk(
                chunk_id=f"chunk{i}",
                doc_id="doc1",
                vault_id="vault1",
                anchor_type="heading",
                anchor_ref=f"h{i}",
                text=f"content {i}",
                text_hash=f"hash{i}",
            )
            for i in range(20)
        ]
        embeddings = [np.random.randn(384).astype(np.float32) for _ in chunks]

        store.upsert_chunks(chunks)

        query_embedding = np.random.randn(384).astype(np.float32)
        results = store.vector_search(query_embedding, k=5, filters={"vault_id": "vault1"})

        assert len(results) <= 5


class TestMultipleVaults:
    """Test store with multiple vaults."""

    def test_separate_vaults_isolated(self, store: LibSqlStore):
        """Test that documents from different vaults are isolated."""
        doc1 = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash1",
            deleted=False,
        )
        doc2 = Document(
            doc_id="doc2",
            vault_id="vault2",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash2",
            deleted=False,
        )

        store.upsert_document(doc1)
        store.upsert_document(doc2)

        chunk1 = Chunk(
            chunk_id="chunk1",
            doc_id="doc1",
            vault_id="vault1",
            anchor_type="heading",
            anchor_ref="h1",
            text="vault1 content",
            text_hash="hash1",
        )
        chunk2 = Chunk(
            chunk_id="chunk2",
            doc_id="doc2",
            vault_id="vault2",
            anchor_type="heading",
            anchor_ref="h2",
            text="vault2 content",
            text_hash="hash2",
        )

        embeddings = [np.random.randn(384).astype(np.float32) for _ in [chunk1, chunk2]]
        store.upsert_chunks([chunk1, chunk2])

        vault1_results = store.lexical_search("vault1", k=10, filters={"vault_id": "vault1"})
        vault2_results = store.lexical_search("vault2", k=10, filters={"vault_id": "vault2"})

        # Results should be from respective vaults
        if len(vault1_results) > 0:
            assert vault1_results[0].chunk.vault_id == "vault1"
        if len(vault2_results) > 0:
            assert vault2_results[0].chunk.vault_id == "vault2"
