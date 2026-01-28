"""Quality evaluation tests for reranking - measures improvement in precision.

NOTE: These tests require network access to download models on first run.
They are marked with @pytest.mark.manual for optional execution.

Run with: pytest tests/test_reranking_quality.py -v -m manual
"""

import pytest
from pathlib import Path
import dataclasses
from mneme.config import VaultConfig
from mneme.indexer.indexer import Indexer
from mneme.retrieval.retriever import Retriever
from mneme.retrieval.reranker import CROSS_ENCODER_AVAILABLE


# Test queries with known relevant documents
TEST_CASES = [
    {
        "query": "kubernetes deployment strategies",
        "relevant": ["kubernetes.md"],
        "irrelevant": ["lambda.md", "jenkins.md"]
    },
    {
        "query": "serverless AWS functions",
        "relevant": ["lambda.md"],
        "irrelevant": ["kubernetes.md"]
    }
]


@pytest.mark.skipif(not CROSS_ENCODER_AVAILABLE, reason="sentence-transformers not installed")
@pytest.mark.manual  # Mark as manual test - requires network access for model downloads
def test_reranking_improves_precision(tmp_path: Path):
    """Test that reranking improves precision@5 for test queries.

    This test creates a small test vault, indexes it, and runs test queries
    with and without reranking to verify quality improvement.
    """
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create test documents with clear semantic differences
    (vault / "kubernetes.md").write_text("""# Kubernetes Deployment Guide

Complete guide to deploying applications on Kubernetes clusters.
Kubernetes deployment strategies for production environments.
Rolling updates, blue-green deployments, and canary releases with Kubernetes.
Container orchestration and scaling with Kubernetes.
""")

    (vault / "lambda.md").write_text("""# AWS Lambda Deployment

Deploying serverless functions on AWS Lambda platform.
AWS Lambda deployment strategies and best practices.
Deployment packages, layers, and versioning for Lambda functions.
Serverless architecture with AWS Lambda functions.
""")

    (vault / "jenkins.md").write_text("""# Jenkins CI/CD Pipeline

Jenkins deployment automation and continuous integration.
Building and deploying applications with Jenkins pipelines.
Deployment strategies for Jenkins-based workflows.
Automated testing and deployment with Jenkins.
""")

    # Create config (without reranking initially)
    config_base = VaultConfig(
        vault_root=vault,
        index_dir=tmp_path / "index",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        image_analysis_provider="off",
        k_vec=10,
        k_lex=10,
        top_k=5,
        use_rerank=False,
        offline_mode=False  # Allow downloads if needed
    )

    # Index the vault once
    print("\n📦 Indexing test vault...")
    indexer = Indexer(config_base)
    indexer.scan(full=True)
    print("✓ Indexing complete")

    # Helper function to calculate precision@k
    def precision_at_k(results, test_case, k=5):
        top_paths = [r.source_ref.rel_path for r in results[:k]]
        relevant_count = sum(1 for path in top_paths
                           if any(rel in path for rel in test_case["relevant"]))
        return relevant_count / min(k, len(results)) if results else 0

    # Test each query
    improvements = []
    for test_case in TEST_CASES:
        print(f"\n🔍 Testing query: '{test_case['query']}'")

        # Search WITHOUT reranking
        retriever_no_rerank = Retriever(config_base)
        results_no_rerank = retriever_no_rerank.search(test_case["query"], k=5)
        p_no_rerank = precision_at_k(results_no_rerank, test_case)

        print(f"  Without reranking: P@5 = {p_no_rerank:.2f}")
        print(f"    Top 3 docs: {[r.source_ref.rel_path for r in results_no_rerank[:3]]}")

        # Search WITH reranking
        config_rerank = dataclasses.replace(
            config_base,
            use_rerank=True,
            rerank_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",  # Fast model for testing
            offline_mode=False
        )
        retriever_rerank = Retriever(config_rerank)
        results_rerank = retriever_rerank.search(test_case["query"], k=5)
        p_rerank = precision_at_k(results_rerank, test_case)

        print(f"  With reranking:    P@5 = {p_rerank:.2f}")
        print(f"    Top 3 docs: {[r.source_ref.rel_path for r in results_rerank[:3]]}")

        # Verify reranking metadata is present
        for result in results_rerank:
            assert result.metadata.get("reranked") is True, "Results should have reranking metadata"
            assert "original_score" in result.metadata
            assert "reranker_score" in result.metadata

        # Calculate improvement
        if p_no_rerank > 0:
            improvement = ((p_rerank - p_no_rerank) / p_no_rerank * 100)
        else:
            improvement = 0 if p_rerank == 0 else 100

        improvements.append(improvement)

        # Reranking should improve or maintain precision
        assert p_rerank >= p_no_rerank, \
            f"Reranking decreased precision for '{test_case['query']}': {p_no_rerank:.2f} -> {p_rerank:.2f}"

        if improvement > 0:
            print(f"  ✅ Improvement: +{improvement:.1f}%")
        else:
            print(f"  ✓ Maintained precision")

    # Summary
    avg_improvement = sum(improvements) / len(improvements)
    print(f"\n📊 Average improvement: +{avg_improvement:.1f}%")
    print(f"   Test cases with improvement: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")


@pytest.mark.skipif(not CROSS_ENCODER_AVAILABLE, reason="sentence-transformers not installed")
@pytest.mark.manual  # Mark as manual test - requires network access for model downloads
def test_reranking_metadata_accuracy(tmp_path: Path):
    """Test that reranking metadata is accurate and scores make sense."""
    vault = tmp_path / "vault"
    vault.mkdir()

    (vault / "test.md").write_text("""# Test Document

This is a test document about machine learning and deep learning.
Neural networks are a key component of deep learning systems.
""")

    config = VaultConfig(
        vault_root=vault,
        index_dir=tmp_path / "index",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        image_analysis_provider="off",
        use_rerank=True,
        rerank_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        offline_mode=False
    )

    # Index and search
    indexer = Indexer(config)
    indexer.scan(full=True)

    retriever = Retriever(config)
    results = retriever.search("deep learning neural networks", k=5)

    # Verify metadata
    for result in results:
        assert result.metadata["reranked"] is True
        assert "original_score" in result.metadata
        assert "reranker_score" in result.metadata

        # Scores should be in valid range
        assert -10 <= result.metadata["original_score"] <= 10, "Original score out of expected range"
        assert -10 <= result.metadata["reranker_score"] <= 10, "Reranker score out of expected range"

        # Current score should match reranker score
        assert result.score == result.metadata["reranker_score"], \
            "Result score should be updated to reranker score"
