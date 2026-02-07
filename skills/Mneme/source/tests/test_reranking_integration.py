"""Integration tests for reranking with retrieval pipeline."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
import dataclasses
from mneme.config import VaultConfig
from mneme.retrieval.retriever import Retriever
from mneme.retrieval.reranker import CROSS_ENCODER_AVAILABLE
from mneme.models import SearchResult, SourceRef


@pytest.mark.skipif(not CROSS_ENCODER_AVAILABLE, reason="sentence-transformers not installed")
class TestRerankerIntegration:

    @pytest.fixture
    def minimal_config(self, tmp_path: Path):
        """Create minimal config for testing."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "test.md").write_text("# Test\n\nSome content")

        return VaultConfig(
            vault_root=vault,
            index_dir=tmp_path / "index",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_device="cpu",
            image_analysis_provider="off",
            use_rerank=False  # Start disabled
        )

    def test_reranking_disabled_by_default(self, minimal_config):
        """Test that reranker is None when use_rerank=False."""
        with patch('sentence_transformers.SentenceTransformer'):
            retriever = Retriever(minimal_config)
            assert retriever.reranker is None

    def test_reranking_enabled_creates_reranker(self, minimal_config):
        """Test that reranker is created when use_rerank=True."""
        config_with_rerank = dataclasses.replace(minimal_config, use_rerank=True)

        with patch('sentence_transformers.SentenceTransformer'), \
             patch('mneme.retrieval.reranker.CrossEncoder'):
            retriever = Retriever(config_with_rerank)
            assert retriever.reranker is not None

    def test_reranker_initialization_message(self, minimal_config, caplog):
        """Test that reranker logs initialization message."""
        config_with_rerank = dataclasses.replace(
            minimal_config,
            use_rerank=True,
            rerank_model="cross-encoder/test-model"
        )

        with patch('sentence_transformers.SentenceTransformer'), \
             patch('mneme.retrieval.reranker.CrossEncoder'), \
             caplog.at_level("INFO", logger="mneme.retrieval.retriever"):
            Retriever(config_with_rerank)
            assert "Reranker enabled (lazy)" in caplog.text
            assert "cross-encoder/test-model" in caplog.text

    def test_reranking_integrates_with_search(self, minimal_config):
        """Test that reranker is called during search when enabled."""
        config_with_rerank = dataclasses.replace(minimal_config, use_rerank=True, top_k=3)

        # Mock results from hybrid search
        mock_results = [
            SearchResult(
                chunk_id="1",
                score=0.9,
                snippet="First result",
                source_ref=SourceRef(vault_id="v1", rel_path="doc1.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="h1"),
                metadata={}
            ),
            SearchResult(
                chunk_id="2",
                score=0.8,
                snippet="Second result",
                source_ref=SourceRef(vault_id="v1", rel_path="doc2.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="h1"),
                metadata={}
            ),
        ]

        with patch('sentence_transformers.SentenceTransformer'), \
             patch('mneme.retrieval.reranker.CrossEncoder') as mock_cross_encoder:

            # Setup mocks
            mock_model = Mock()
            mock_cross_encoder.return_value = mock_model
            mock_model.predict.return_value = np.array([0.7, 0.95])  # Reverse order

            retriever = Retriever(config_with_rerank)

            # Mock the hybrid search results
            with patch.object(retriever.ranker, 'merge', return_value=mock_results):
                with patch.object(retriever.embedder, 'embed_query', return_value=np.zeros(384)):
                    results = retriever.search("test query", k=3)

            # Verify reranking was applied
            assert len(results) == 2
            # Results should be reordered (chunk_id="2" now first due to higher reranker score)
            assert results[0].chunk_id == "2"
            assert results[0].metadata["reranked"] is True
            assert results[0].metadata["original_score"] == 0.8
            assert results[0].metadata["reranker_score"] == 0.95

    def test_search_without_reranking(self, minimal_config):
        """Test that search works normally without reranking."""
        mock_results = [
            SearchResult(
                chunk_id="1",
                score=0.9,
                snippet="First result",
                source_ref=SourceRef(vault_id="v1", rel_path="doc1.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="h1"),
                metadata={}
            ),
        ]

        with patch('sentence_transformers.SentenceTransformer'):
            retriever = Retriever(minimal_config)

            # Mock the hybrid search results
            with patch.object(retriever.ranker, 'merge', return_value=mock_results):
                with patch.object(retriever.embedder, 'embed_query', return_value=np.zeros(384)):
                    results = retriever.search("test query", k=1)

            # Verify no reranking metadata
            assert len(results) == 1
            assert results[0].chunk_id == "1"
            assert "reranked" not in results[0].metadata

    def test_config_override_workflow(self, minimal_config):
        """Test CLI workflow: config has use_rerank=false, override to true."""
        # Start with reranking disabled
        with patch('sentence_transformers.SentenceTransformer'):
            retriever_no_rerank = Retriever(minimal_config)
            assert retriever_no_rerank.reranker is None

        # Override to enable (simulating CLI --rerank flag)
        config_override = dataclasses.replace(minimal_config, use_rerank=True)

        with patch('sentence_transformers.SentenceTransformer'), \
             patch('mneme.retrieval.reranker.CrossEncoder'):
            retriever_with_rerank = Retriever(config_override)
            assert retriever_with_rerank.reranker is not None
