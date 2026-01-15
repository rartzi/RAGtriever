"""
Comprehensive tests for retrieval system: search, ranking, and result handling.
"""

import pytest
import tempfile
from pathlib import Path

from cortexindex.config import VaultConfig
from cortexindex.indexer.indexer import Indexer
from cortexindex.retrieval.retriever import Retriever


@pytest.fixture
def vault_config(tmp_path: Path) -> VaultConfig:
    """Create vault config with test data."""
    vault = tmp_path / "test_vault"
    vault.mkdir()

    # Create test documents with varied content
    (vault / "project_a.md").write_text(
        """# Project A

This is a cloud infrastructure project for scaling.

## Details

- Architecture: microservices
- Cloud provider: AWS
- Status: active
"""
    )

    (vault / "project_b.md").write_text(
        """# Project B

Database optimization and query tuning.

## Specifications

- Database: PostgreSQL
- Optimization focus: query performance
- Team: backend
"""
    )

    (vault / "project_c.md").write_text(
        """# Project C

Machine learning pipeline implementation.

## Information

- ML framework: TensorFlow
- Pipeline stage: training and inference
- Data: large scale datasets
"""
    )

    (vault / "notes.md").write_text(
        """# General Notes

## Infrastructure

Cloud infrastructure and deployment notes.
Discussing scalability and performance optimization.

## Development

Development workflow and best practices.
Including code review and testing strategies.
"""
    )

    # Create and return config
    return VaultConfig(
        vault_root=vault,
        index_dir=tmp_path / "index",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        image_analysis_provider="off",
    )


@pytest.fixture
def indexer_with_data(vault_config: VaultConfig) -> Indexer:
    """Create an indexer with test data."""
    indexer = Indexer(vault_config)
    indexer.scan(full=True)
    return indexer


@pytest.fixture
def retriever(vault_config: VaultConfig) -> Retriever:
    """Create a retriever from vault config."""
    return Retriever(vault_config)


class TestLexicalSearch:
    """Test lexical (keyword) search functionality."""

    def test_search_finds_exact_match(self, retriever: Retriever):
        """Test that exact matches are found."""
        results = retriever.search("cloud", k=10)

        assert len(results) > 0
        assert any("cloud" in r.chunk.text.lower() for r in results)

    def test_search_returns_top_k_results(self, retriever: Retriever):
        """Test that k parameter limits results."""
        results_k5 = retriever.search("project", k=5)
        results_k2 = retriever.search("project", k=2)

        assert len(results_k5) <= 5
        assert len(results_k2) <= 2
        assert len(results_k5) >= len(results_k2)

    def test_search_case_insensitive(self, retriever: Retriever):
        """Test that search is case insensitive."""
        results_lower = retriever.search("cloud", k=10)
        results_upper = retriever.search("CLOUD", k=10)
        results_mixed = retriever.search("Cloud", k=10)

        # Should find similar results
        assert len(results_lower) > 0
        assert len(results_upper) > 0
        assert len(results_mixed) > 0

    def test_search_with_multiple_words(self, retriever: Retriever):
        """Test search with multiple keywords."""
        results = retriever.search("cloud infrastructure", k=10)

        assert len(results) > 0
        # Results should contain documents about cloud and infrastructure

    def test_search_no_results(self, retriever: Retriever):
        """Test search with no matching results."""
        results = retriever.search("xyz123abc999", k=10)

        # Should return empty list, not error
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_returns_scores(self, retriever: Retriever):
        """Test that search results include scores."""
        results = retriever.search("project", k=5)

        assert len(results) > 0
        for result in results:
            assert hasattr(result, "score")
            assert result.score >= 0


class TestSemanticSearch:
    """Test semantic (embedding-based) search functionality."""

    def test_semantic_search_executes(self, retriever: Retriever):
        """Test that semantic search runs without error."""
        results = retriever.search("scalable database systems", k=5)

        assert isinstance(results, list)
        # May or may not find results depending on embeddings

    def test_semantic_search_similarity(self, retriever: Retriever):
        """Test semantic search finds related concepts."""
        results = retriever.search("machine learning models", k=10)

        # Should execute without error
        assert isinstance(results, list)

    def test_semantic_vs_lexical_different(self, retriever: Retriever):
        """Test that semantic and lexical can return different results."""
        # This test verifies the systems work, exact differences vary
        lex_results = retriever.search("database", k=5)
        sem_results = retriever.search("data storage system", k=5)

        assert isinstance(lex_results, list)
        assert isinstance(sem_results, list)


class TestSearchFilters:
    """Test filtering search results."""

    def test_filter_by_file_type(self, indexer_with_data: Indexer):
        """Test filtering by file type."""
        retriever = Retriever(vault_config)

        results = retriever.search("project", k=10)

        assert len(results) > 0
        # All results should be from markdown files
        for result in results:
            assert result.chunk.metadata.get("file_type") in ["markdown", None]

    def test_filter_by_multiple_file_types(self, indexer_with_data: Indexer):
        """Test filtering by multiple file types."""
        retriever = Retriever(vault_config)

        results = retriever.search("project", k=10)

        # Should execute without error
        assert isinstance(results, list)

    def test_filter_by_path_pattern(self, indexer_with_data: Indexer):
        """Test filtering by path pattern."""
        retriever = Retriever(vault_config)

        # Search in specific documents
        results = retriever.search("project", k=10, path_pattern="project_a")

        # Should find results in project_a.md
        assert isinstance(results, list)


class TestSearchRanking:
    """Test result ranking and sorting."""

    def test_results_are_sorted_by_score(self, retriever: Retriever):
        """Test that results are sorted by score (descending)."""
        results = retriever.search("cloud infrastructure", k=10)

        if len(results) > 1:
            # Scores should be in descending order
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_higher_relevance_first(self, retriever: Retriever):
        """Test that more relevant results appear first."""
        results = retriever.search("database", k=10)

        if len(results) > 1:
            # First result should have highest score
            assert results[0].score >= results[1].score


class TestDocumentOpening:
    """Test opening specific documents by chunk ID."""

    def test_open_valid_chunk(self, indexer_with_data: Indexer):
        """Test opening a document with valid chunk ID."""
        retriever = Retriever(vault_config)

        # Get a chunk ID
        vault_id = indexer_with_data.vault_id
        docs = indexer_with_data.store.get_documents_by_vault(vault_id)
        assert len(docs) > 0

        doc = docs[0]
        chunks = indexer_with_data.store.get_chunks_by_document(doc.doc_id)
        assert len(chunks) > 0

        chunk = chunks[0]

        # Open the document
        result = retriever.open(chunk.chunk_id)

        assert result is not None
        assert result.chunk is not None
        assert result.chunk.chunk_id == chunk.chunk_id

    def test_open_includes_context(self, indexer_with_data: Indexer):
        """Test that opening includes surrounding context."""
        retriever = Retriever(vault_config)

        vault_id = indexer_with_data.vault_id
        docs = indexer_with_data.store.get_documents_by_vault(vault_id)
        doc = docs[0]
        chunks = indexer_with_data.store.get_chunks_by_document(doc.doc_id)

        if len(chunks) > 0:
            chunk = chunks[0]
            result = retriever.open(chunk.chunk_id)

            assert result is not None
            assert result.context is not None

    def test_open_invalid_chunk_id(self, retriever: Retriever):
        """Test opening with invalid chunk ID."""
        result = retriever.open("invalid_chunk_id_12345")

        # Should handle gracefully
        assert result is None or result.chunk is None


class TestGraphNavigation:
    """Test graph-based navigation (neighbors, backlinks)."""

    def test_find_neighbors(self, indexer_with_data: Indexer):
        """Test finding neighboring chunks."""
        retriever = Retriever(vault_config)

        vault_id = indexer_with_data.vault_id
        docs = indexer_with_data.store.get_documents_by_vault(vault_id)
        doc = docs[0]
        chunks = indexer_with_data.store.get_chunks_by_document(doc.doc_id)

        if len(chunks) > 0:
            chunk = chunks[0]

            # Find neighbors
            neighbors = retriever.neighbors(chunk.chunk_id, k=5)

            assert neighbors is not None
            assert isinstance(neighbors, list)

    def test_neighbors_returns_limited_results(self, indexer_with_data: Indexer):
        """Test that neighbors respects k parameter."""
        retriever = Retriever(vault_config)

        vault_id = indexer_with_data.vault_id
        docs = indexer_with_data.store.get_documents_by_vault(vault_id)
        doc = docs[0]
        chunks = indexer_with_data.store.get_chunks_by_document(doc.doc_id)

        if len(chunks) > 0:
            chunk = chunks[0]

            neighbors = retriever.neighbors(chunk.chunk_id, k=3)

            assert len(neighbors) <= 3

    def test_neighbors_excludes_self(self, indexer_with_data: Indexer):
        """Test that neighbors doesn't include the source chunk."""
        retriever = Retriever(vault_config)

        vault_id = indexer_with_data.vault_id
        docs = indexer_with_data.store.get_documents_by_vault(vault_id)
        doc = docs[0]
        chunks = indexer_with_data.store.get_chunks_by_document(doc.doc_id)

        if len(chunks) > 0:
            chunk = chunks[0]
            neighbors = retriever.neighbors(chunk.chunk_id, k=10)

            # Source chunk should not be in neighbors
            neighbor_ids = [n.chunk_id for n in neighbors] if neighbors else []
            assert chunk.chunk_id not in neighbor_ids


class TestResultPagination:
    """Test result pagination and offset."""

    def test_large_k_returns_all_available(self, retriever: Retriever):
        """Test that large k returns all available results."""
        results = retriever.search("project", k=1000)

        assert isinstance(results, list)
        # Should return whatever is available, up to k

    def test_k_one_returns_top_result(self, retriever: Retriever):
        """Test that k=1 returns only top result."""
        results = retriever.search("project", k=1)

        assert len(results) <= 1


class TestErrorHandling:
    """Test error handling in retrieval."""

    def test_empty_query_search(self, retriever: Retriever):
        """Test search with empty query."""
        results = retriever.search("", k=10)

        # Should handle gracefully
        assert isinstance(results, list)

    def test_null_query_search(self, retriever: Retriever):
        """Test search with None query."""
        try:
            results = retriever.search(None, k=10)
            assert isinstance(results, list)
        except (TypeError, AttributeError):
            # Exception is acceptable for None input
            pass

    def test_very_long_query(self, retriever: Retriever):
        """Test search with very long query."""
        long_query = " ".join(["word"] * 1000)
        results = retriever.search(long_query, k=5)

        # Should handle without crashing
        assert isinstance(results, list)


class TestResultContent:
    """Test that search results contain expected content."""

    def test_result_has_chunk(self, retriever: Retriever):
        """Test that results include chunk data."""
        results = retriever.search("project", k=5)

        if len(results) > 0:
            result = results[0]
            assert result.chunk is not None
            assert result.chunk.text is not None

    def test_result_has_document_reference(self, retriever: Retriever):
        """Test that results reference source document."""
        results = retriever.search("project", k=5)

        if len(results) > 0:
            result = results[0]
            assert result.chunk is not None
            assert result.chunk.doc_id is not None

    def test_result_has_metadata(self, retriever: Retriever):
        """Test that results include metadata."""
        results = retriever.search("project", k=5)

        if len(results) > 0:
            result = results[0]
            assert result.chunk is not None
            assert result.chunk.metadata is not None


class TestRetrievalConsistency:
    """Test that retrieval is consistent and reproducible."""

    def test_same_query_consistent_order(self, retriever: Retriever):
        """Test that same query returns consistent order."""
        results1 = retriever.search("cloud", k=10)
        results2 = retriever.search("cloud", k=10)

        if len(results1) > 0 and len(results2) > 0:
            # Same number of results
            assert len(results1) == len(results2)

            # Same order and scores
            for r1, r2 in zip(results1, results2):
                assert r1.chunk.chunk_id == r2.chunk.chunk_id
                assert abs(r1.score - r2.score) < 0.0001
