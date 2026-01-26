"""Unit tests for cross-encoder reranker."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from mneme.retrieval.reranker import CrossEncoderReranker, CROSS_ENCODER_AVAILABLE
from mneme.models import SearchResult, SourceRef


@pytest.mark.skipif(not CROSS_ENCODER_AVAILABLE, reason="sentence-transformers not installed")
class TestCrossEncoderReranker:

    @pytest.fixture
    def reranker(self):
        """Create reranker with mocked model for fast testing without downloads."""
        with patch('mneme.retrieval.reranker.CrossEncoder') as mock_cross_encoder:
            # Create mock model
            mock_model = Mock()
            mock_cross_encoder.return_value = mock_model

            # Create reranker (will use mocked CrossEncoder)
            reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
                device="cpu"
            )

            # Replace the model with our mock
            reranker.model = mock_model

            yield reranker

    def test_reranker_basic(self, reranker):
        """Test basic reranking functionality."""
        results = [
            SearchResult(
                chunk_id="1",
                score=0.9,
                snippet="AWS Lambda deployment strategies for serverless",
                source_ref=SourceRef(vault_id="v1", rel_path="aws.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="lambda"),
                metadata={}
            ),
            SearchResult(
                chunk_id="2",
                score=0.8,
                snippet="Kubernetes deployment guide for production clusters",
                source_ref=SourceRef(vault_id="v1", rel_path="k8s.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="deploy"),
                metadata={}
            ),
        ]

        # Mock scores: Kubernetes doc gets higher score (0.95 vs 0.7)
        reranker.model.predict.return_value = np.array([0.7, 0.95])

        # Query is about kubernetes
        reranked = reranker.rerank("kubernetes deployment", results, top_k=2)

        # Assertions
        assert len(reranked) == 2
        assert reranked[0].metadata["reranked"] is True
        assert "original_score" in reranked[0].metadata
        assert "reranker_score" in reranked[0].metadata

        # Kubernetes doc should be ranked first (more relevant)
        assert reranked[0].chunk_id == "2"

    def test_reranker_empty_results(self, reranker):
        """Test reranking with empty results."""
        reranked = reranker.rerank("test query", [], top_k=10)
        assert reranked == []

    def test_reranker_top_k_limiting(self, reranker):
        """Test that reranker respects top_k parameter."""
        results = [
            SearchResult(
                chunk_id=str(i),
                score=0.5,
                snippet=f"Document {i} content",
                source_ref=SourceRef(vault_id="v1", rel_path=f"doc{i}.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="h1"),
                metadata={}
            )
            for i in range(20)
        ]

        # Mock scores for 20 documents
        reranker.model.predict.return_value = np.array([0.5 + i * 0.01 for i in range(20)])

        reranked = reranker.rerank("test query", results, top_k=5)
        assert len(reranked) == 5

    def test_reranker_preserves_metadata(self, reranker):
        """Test that reranker preserves existing metadata."""
        results = [
            SearchResult(
                chunk_id="1",
                score=0.8,
                snippet="Test content",
                source_ref=SourceRef(vault_id="v1", rel_path="test.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="h1"),
                metadata={"custom_field": "value"}
            )
        ]

        # Mock score
        reranker.model.predict.return_value = np.array([0.85])

        reranked = reranker.rerank("query", results, top_k=1)
        assert reranked[0].metadata["custom_field"] == "value"
        assert reranked[0].metadata["reranked"] is True

    def test_reranker_score_update(self, reranker):
        """Test that reranker updates the score field correctly."""
        results = [
            SearchResult(
                chunk_id="1",
                score=0.5,
                snippet="Python programming tutorial",
                source_ref=SourceRef(vault_id="v1", rel_path="python.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="tutorial"),
                metadata={}
            ),
        ]

        # Mock reranker score (different from original)
        reranker.model.predict.return_value = np.array([0.92])

        reranked = reranker.rerank("python tutorial", results, top_k=1)

        # Score should be updated to reranker score
        assert reranked[0].score != 0.5  # Should be different from original
        assert reranked[0].metadata["original_score"] == 0.5
        assert reranked[0].score == reranked[0].metadata["reranker_score"]
        assert reranked[0].score == 0.92

    def test_reranker_reordering(self, reranker):
        """Test that reranker can reorder results based on relevance."""
        results = [
            SearchResult(
                chunk_id="1",
                score=0.9,
                snippet="Machine learning with scikit-learn library",
                source_ref=SourceRef(vault_id="v1", rel_path="ml.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="sklearn"),
                metadata={}
            ),
            SearchResult(
                chunk_id="2",
                score=0.8,
                snippet="Deep learning with PyTorch neural networks",
                source_ref=SourceRef(vault_id="v1", rel_path="dl.md", file_type="md",
                                     anchor_type="md_heading", anchor_ref="pytorch"),
                metadata={}
            ),
        ]

        # Mock scores: Deep learning gets higher score despite lower original score
        reranker.model.predict.return_value = np.array([0.65, 0.93])

        # Query specifically about deep learning
        reranked = reranker.rerank("deep learning neural networks", results, top_k=2)

        # Deep learning doc should rank higher despite lower original score
        assert reranked[0].chunk_id == "2"
        assert reranked[1].chunk_id == "1"


def test_reranker_import_error():
    """Test that import error is raised when sentence-transformers is not available."""
    if not CROSS_ENCODER_AVAILABLE:
        with pytest.raises(ImportError, match="sentence-transformers required"):
            CrossEncoderReranker()
