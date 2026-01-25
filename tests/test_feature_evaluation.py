"""Comprehensive evaluation tests for new retrieval features.

This script compares:
- Baseline: No heading boost, no tag boost, no MMR diversity
- With Heading Boost: Only heading boost enabled
- With Tag Boost: Only tag boost enabled
- With MMR Diversity: Only MMR diversity enabled
- All Features: All three enabled

Metrics measured:
- Precision@5, Precision@10
- Diversity (unique documents in top-k)
- Relevance (% of results matching expected documents)
- Score improvements

NOTE: These tests require network access to download models on first run.
They are marked with @pytest.mark.manual for optional execution.

Run with: pytest tests/test_feature_evaluation.py -v -m manual
"""

import pytest
from pathlib import Path
import dataclasses
from datetime import datetime, timedelta
from ragtriever.config import VaultConfig
from ragtriever.indexer.indexer import Indexer
from ragtriever.retrieval.retriever import Retriever
from ragtriever.retrieval.diversity import MMRDiversifier, DiversityConfig


# Test queries with expected behavior
TEST_CASES = [
    {
        "name": "kubernetes_architecture_heading",
        "query": "kubernetes architecture",
        "relevant": ["kubernetes-architecture.md"],
        "expected_behavior": "Should match H1 title 'Kubernetes Architecture' with heading boost",
    },
    {
        "name": "machine_learning_deployment_tag",
        "query": "machine learning deployment",
        "relevant": ["ml-deployment.md", "ml-training.md"],
        "expected_behavior": "Should match #machine-learning tag with tag boost",
    },
    {
        "name": "api_implementation_diversity",
        "query": "API implementation details",
        "relevant": ["api-design.md", "api-security.md", "api-testing.md"],
        "expected_behavior": "Should benefit from MMR diversity (multiple docs)",
    },
    {
        "name": "docker_containers_heading_tag",
        "query": "docker container orchestration",
        "relevant": ["docker-guide.md", "kubernetes-architecture.md"],
        "expected_behavior": "Should benefit from both heading boost (H2) and tag boost",
    },
    {
        "name": "database_design_multi_chunk",
        "query": "database schema design patterns",
        "relevant": ["database-design.md"],
        "expected_behavior": "Long doc with many chunks - MMR should select diverse chunks",
    },
]


def create_test_vault(vault_path: Path) -> None:
    """Create a test vault with documents designed to test each feature.

    Documents include:
    - Clear headings (H1, H2, H3) for heading boost testing
    - Tags (#machine-learning, #kubernetes, etc.) for tag boost testing
    - Multiple chunks per document for MMR diversity testing
    - Mix of relevant/irrelevant content
    """
    vault_path.mkdir(exist_ok=True)

    # Document 1: Kubernetes Architecture (H1 title boost)
    (vault_path / "kubernetes-architecture.md").write_text("""# Kubernetes Architecture

Comprehensive guide to Kubernetes cluster architecture and components.

#kubernetes #container-orchestration #devops

## Core Components

The control plane consists of several key components that manage the cluster.
API server handles all REST operations and serves as the frontend.
Scheduler assigns pods to nodes based on resource requirements.
Controller manager runs controllers that regulate cluster state.

## Node Components

Each worker node runs a kubelet agent that manages pod lifecycle.
Container runtime (Docker, containerd) runs the actual containers.
Kube-proxy manages network routing for services.

## Networking

Kubernetes networking model ensures pods can communicate across nodes.
Services provide stable endpoints for groups of pods.
Ingress controllers manage external access to services.
""")

    # Document 2: ML Deployment (tag boost focus)
    (vault_path / "ml-deployment.md").write_text("""# Machine Learning Deployment

Best practices for deploying machine learning models to production.

#machine-learning #mlops #deployment #production

## Model Serving

Production deployment requires careful consideration of inference latency.
Model serving frameworks like TensorFlow Serving or TorchServe are essential.
API endpoints should handle preprocessing and postprocessing efficiently.

## Scaling Strategies

Horizontal scaling with multiple model replicas handles traffic spikes.
GPU acceleration improves throughput for deep learning models.
Load balancing distributes requests across model instances.

## Monitoring

Production ML systems require comprehensive monitoring and observability.
Track prediction accuracy, latency, throughput, and resource utilization.
Set up alerts for model drift and performance degradation.
""")

    # Document 3: ML Training (tag boost + multiple chunks)
    (vault_path / "ml-training.md").write_text("""# Machine Learning Training Pipeline

End-to-end guide for training machine learning models at scale.

#machine-learning #training #data-engineering

## Data Preparation

Quality training data is the foundation of successful machine learning.
Data cleaning, normalization, and feature engineering are critical steps.
Split datasets into training, validation, and test sets appropriately.

## Model Training

Select appropriate architectures for your problem domain.
Hyperparameter tuning significantly impacts model performance.
Use techniques like cross-validation to prevent overfitting.

## Distributed Training

Large models and datasets require distributed training strategies.
Data parallelism splits batches across multiple GPUs.
Model parallelism splits model layers across devices.
""")

    # Document 4: API Design (diversity testing - multiple relevant chunks)
    (vault_path / "api-design.md").write_text("""# API Design Principles

Comprehensive guide to designing robust and scalable APIs.

#api #architecture #rest

## RESTful Design

Follow REST principles for predictable and maintainable APIs.
Use standard HTTP methods: GET, POST, PUT, PATCH, DELETE.
Resource-based URLs provide clear API structure.
API implementation details should follow HTTP status codes properly.
Return 200 for success, 201 for created, 404 for not found.

## Versioning

API versioning prevents breaking changes for existing clients.
URL versioning (v1, v2) or header-based versioning are common approaches.
Implementation details require careful planning for backward compatibility.

## Request/Response Design

Clear request and response schemas improve API usability.
Use JSON for data interchange in modern APIs.
Implementation details include proper error messages and validation.
Consistent data structures across endpoints reduce integration complexity.

## Rate Limiting

Protect API infrastructure with rate limiting and throttling.
Implementation details: token bucket, sliding window, or fixed window algorithms.
Return 429 status code when rate limits are exceeded.
""")

    # Document 5: API Security (diversity testing)
    (vault_path / "api-security.md").write_text("""# API Security Best Practices

Essential security measures for protecting API endpoints.

#api #security #authentication

## Authentication

OAuth 2.0 provides robust authentication for modern APIs.
JWT tokens enable stateless authentication across services.
API implementation details should validate tokens on every request.
Never store credentials in plain text or client-side code.

## Authorization

Role-based access control (RBAC) manages user permissions.
Fine-grained authorization controls access to specific resources.
Implementation details: check permissions before processing requests.

## Input Validation

Validate all input to prevent injection attacks and data corruption.
Sanitize user input before processing or storing.
API implementation details should reject malformed requests early.
""")

    # Document 6: API Testing (diversity testing)
    (vault_path / "api-testing.md").write_text("""# API Testing Strategies

Comprehensive testing approaches for API development.

#api #testing #quality-assurance

## Unit Testing

Test individual API endpoints in isolation.
Mock external dependencies for reliable unit tests.
API implementation details should be testable without full integration.

## Integration Testing

Test API endpoints with real dependencies and databases.
Verify data flows correctly through the system.
Implementation details: use test databases and cleanup after tests.

## Performance Testing

Load testing reveals API performance under stress.
Measure response times, throughput, and resource utilization.
API implementation details should handle concurrent requests efficiently.
""")

    # Document 7: Docker Guide (heading + tag boost)
    (vault_path / "docker-guide.md").write_text("""# Docker Container Guide

Complete guide to containerization with Docker.

#docker #containers #devops

## Container Basics

Containers package applications with dependencies for consistent deployment.
Docker images define the container filesystem and configuration.

## Docker Orchestration

Container orchestration platforms manage multi-container applications.
Docker Swarm provides built-in orchestration capabilities.
Kubernetes offers more advanced orchestration features.

## Best Practices

Use multi-stage builds to optimize image size.
Layer caching speeds up image builds significantly.
""")

    # Document 8: Database Design (long document with many chunks for MMR)
    (vault_path / "database-design.md").write_text("""# Database Schema Design

Comprehensive guide to designing scalable database schemas.

#database #architecture #data-modeling

## Schema Fundamentals

Database schema design patterns are critical for application performance.
Normalize data to eliminate redundancy and ensure consistency.
Choose appropriate data types for efficient storage and indexing.
Primary keys uniquely identify records in tables.

## Relationships

One-to-many relationships are the most common database design pattern.
Foreign keys maintain referential integrity between tables.
Many-to-many relationships require junction tables.
Design patterns for relationships impact query performance.

## Indexing Strategies

Indexes dramatically improve query performance for large datasets.
B-tree indexes are the default for most database systems.
Database schema design patterns should consider read vs write patterns.
Composite indexes support queries with multiple filter conditions.

## Normalization

First normal form (1NF) eliminates repeating groups.
Second normal form (2NF) removes partial dependencies.
Third normal form (3NF) eliminates transitive dependencies.
Design patterns balance normalization with query performance.

## Denormalization

Strategic denormalization improves read performance.
Materialized views pre-compute complex queries.
Schema design patterns for analytics often favor denormalization.
Trade consistency for performance when appropriate.

## Partitioning

Horizontal partitioning splits tables by row ranges.
Vertical partitioning splits tables by column groups.
Database design patterns for large datasets rely on partitioning.
Sharding distributes data across multiple database servers.

## Migrations

Schema migrations manage database changes over time.
Version control schema changes alongside application code.
Design patterns for migrations ensure zero-downtime deployments.
Test migrations on production-like datasets before deployment.
""")

    # Document 9: Jenkins CI/CD (irrelevant document)
    (vault_path / "jenkins-cicd.md").write_text("""# Jenkins CI/CD Pipeline

Continuous integration and deployment with Jenkins.

#jenkins #cicd #automation

## Pipeline Basics

Jenkins pipelines automate build, test, and deployment workflows.
Declarative pipelines provide simple DSL for common patterns.
Scripted pipelines offer more flexibility with Groovy.

## Integration

Integrate Jenkins with version control systems like Git.
Trigger builds automatically on code commits.
Deploy artifacts to staging and production environments.
""")

    # Document 10: AWS Lambda (irrelevant document)
    (vault_path / "lambda-serverless.md").write_text("""# AWS Lambda Serverless

Serverless computing with AWS Lambda functions.

#aws #lambda #serverless

## Function Basics

Lambda functions run code without managing servers.
Event-driven architecture triggers functions automatically.
Pay only for compute time consumed by functions.

## Deployment

Package functions with dependencies in deployment archives.
Use Lambda layers for shared code and libraries.
Configure memory, timeout, and environment variables.
""")


def precision_at_k(results, test_case, k):
    """Calculate precision@k for a test case."""
    if not results:
        return 0.0
    top_paths = [r.source_ref.rel_path for r in results[:k]]
    relevant_count = sum(
        1 for path in top_paths
        if any(rel in path for rel in test_case["relevant"])
    )
    return relevant_count / min(k, len(results))


def diversity_score(results, k):
    """Calculate diversity: ratio of unique documents in top-k results."""
    if not results:
        return 0.0
    top_results = results[:k]
    doc_ids = {f"{r.source_ref.vault_id}:{r.source_ref.rel_path}" for r in top_results}
    return len(doc_ids) / len(top_results)


def relevance_percentage(results, test_case, k):
    """Calculate percentage of results matching expected documents."""
    if not results:
        return 0.0
    top_paths = [r.source_ref.rel_path for r in results[:k]]
    relevant_count = sum(
        1 for path in top_paths
        if any(rel in path for rel in test_case["relevant"])
    )
    return (relevant_count / min(k, len(results))) * 100


@pytest.mark.manual
def test_feature_comparison_comprehensive(tmp_path: Path):
    """Comprehensive comparison of baseline vs new features.

    Tests 4 configurations:
    1. Baseline: No heading boost, no tag boost, no MMR
    2. With Heading Boost: Only heading boost enabled
    3. With Tag Boost: Only tag boost enabled
    4. With MMR Diversity: Only MMR enabled
    5. All Features: All three enabled

    Measures precision@5, precision@10, diversity, and relevance.
    """
    vault = tmp_path / "vault"
    create_test_vault(vault)

    # Base config (will be modified for each configuration)
    config_base = VaultConfig(
        vault_root=vault,
        index_dir=tmp_path / "index",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        image_analysis_provider="off",
        k_vec=20,
        k_lex=20,
        top_k=10,
        use_rerank=False,
        offline_mode=True,  # Use cached models only
        # Disable all boosts initially (baseline)
        heading_boost_enabled=False,
        tag_boost_enabled=False,
        backlink_boost_enabled=False,
        recency_boost_enabled=False,
    )

    # Index the vault once
    print("\n" + "="*80)
    print("INDEXING TEST VAULT")
    print("="*80)
    indexer = Indexer(config_base)
    indexer.scan(full=True)
    print("✓ Indexing complete\n")

    # Define configurations to test
    configurations = [
        {
            "name": "Baseline",
            "config": config_base,
            "apply_mmr": False,
        },
        {
            "name": "With Heading Boost",
            "config": dataclasses.replace(
                config_base,
                heading_boost_enabled=True,
            ),
            "apply_mmr": False,
        },
        {
            "name": "With Tag Boost",
            "config": dataclasses.replace(
                config_base,
                tag_boost_enabled=True,
            ),
            "apply_mmr": False,
        },
        {
            "name": "With MMR Diversity",
            "config": config_base,
            "apply_mmr": True,
        },
        {
            "name": "All Features",
            "config": dataclasses.replace(
                config_base,
                heading_boost_enabled=True,
                tag_boost_enabled=True,
            ),
            "apply_mmr": True,
        },
    ]

    # Store results for comparison
    all_results = {}

    # Run tests for each configuration
    for config_def in configurations:
        config_name = config_def["name"]
        config = config_def["config"]
        apply_mmr = config_def["apply_mmr"]

        print("="*80)
        print(f"CONFIGURATION: {config_name}")
        print("="*80)

        retriever = Retriever(config)
        mmr_diversifier = MMRDiversifier(
            config=DiversityConfig(
                enabled=True,
                lambda_param=0.7,
                max_per_document=2,
            )
        ) if apply_mmr else None

        config_results = []

        for test_case in TEST_CASES:
            # Search
            results = retriever.search(test_case["query"], k=20)

            # Apply MMR if enabled
            if mmr_diversifier:
                results = mmr_diversifier.diversify(results, k=10)
            else:
                results = results[:10]

            # Calculate metrics
            p5 = precision_at_k(results, test_case, 5)
            p10 = precision_at_k(results, test_case, 10)
            div5 = diversity_score(results, 5)
            div10 = diversity_score(results, 10)
            rel5 = relevance_percentage(results, test_case, 5)
            rel10 = relevance_percentage(results, test_case, 10)

            config_results.append({
                "query_name": test_case["name"],
                "p5": p5,
                "p10": p10,
                "div5": div5,
                "div10": div10,
                "rel5": rel5,
                "rel10": rel10,
                "top_3_docs": [r.source_ref.rel_path for r in results[:3]],
            })

            print(f"\nQuery: {test_case['query']}")
            print(f"  P@5={p5:.2f}, P@10={p10:.2f}, Div@5={div5:.2f}, Div@10={div10:.2f}, Rel@5={rel5:.0f}%")
            print(f"  Top 3: {[r.source_ref.rel_path for r in results[:3]]}")

        all_results[config_name] = config_results

    # Generate comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    # Header
    print(f"\n{'Query':<35} {'Metric':<10} {'Baseline':<10} {'Heading':<10} {'Tag':<10} {'MMR':<10} {'All':<10}")
    print("-" * 95)

    # Get baseline results for comparison
    baseline_results = all_results["Baseline"]

    for i, test_case in enumerate(TEST_CASES):
        query_name = test_case["name"]

        # Print each metric
        for metric in ["p5", "p10", "div5", "div10"]:
            metric_label = metric.upper().replace("DIV", "Div@").replace("P", "P@")

            baseline_val = baseline_results[i][metric]
            heading_val = all_results["With Heading Boost"][i][metric]
            tag_val = all_results["With Tag Boost"][i][metric]
            mmr_val = all_results["With MMR Diversity"][i][metric]
            all_val = all_results["All Features"][i][metric]

            # Format with improvement indicators
            def fmt_with_delta(val, base):
                if val > base:
                    delta = ((val - base) / base * 100) if base > 0 else 100
                    return f"{val:.2f}(+{delta:.0f}%)"
                elif val < base:
                    delta = ((base - val) / base * 100) if base > 0 else 0
                    return f"{val:.2f}(-{delta:.0f}%)"
                else:
                    return f"{val:.2f}"

            # Only print query name on first metric
            qname = query_name if metric == "p5" else ""

            print(
                f"{qname:<35} {metric_label:<10} "
                f"{baseline_val:<10.2f} "
                f"{fmt_with_delta(heading_val, baseline_val):<10} "
                f"{fmt_with_delta(tag_val, baseline_val):<10} "
                f"{fmt_with_delta(mmr_val, baseline_val):<10} "
                f"{fmt_with_delta(all_val, baseline_val):<10}"
            )

    # Overall averages
    print("\n" + "-" * 95)
    print("AVERAGE ACROSS ALL QUERIES:")
    print("-" * 95)

    for metric in ["p5", "p10", "div5", "div10"]:
        metric_label = metric.upper().replace("DIV", "Div@").replace("P", "P@")

        baseline_avg = sum(r[metric] for r in baseline_results) / len(baseline_results)
        heading_avg = sum(r[metric] for r in all_results["With Heading Boost"]) / len(TEST_CASES)
        tag_avg = sum(r[metric] for r in all_results["With Tag Boost"]) / len(TEST_CASES)
        mmr_avg = sum(r[metric] for r in all_results["With MMR Diversity"]) / len(TEST_CASES)
        all_avg = sum(r[metric] for r in all_results["All Features"]) / len(TEST_CASES)

        def fmt_with_delta(val, base):
            if val > base:
                delta = ((val - base) / base * 100) if base > 0 else 100
                return f"{val:.2f}(+{delta:.0f}%)"
            elif val < base:
                delta = ((base - val) / base * 100) if base > 0 else 0
                return f"{val:.2f}(-{delta:.0f}%)"
            else:
                return f"{val:.2f}"

        print(
            f"{'Average':<35} {metric_label:<10} "
            f"{baseline_avg:<10.2f} "
            f"{fmt_with_delta(heading_avg, baseline_avg):<10} "
            f"{fmt_with_delta(tag_avg, baseline_avg):<10} "
            f"{fmt_with_delta(mmr_avg, baseline_avg):<10} "
            f"{fmt_with_delta(all_avg, baseline_avg):<10}"
        )

    print("\n" + "="*80)
    print("ASSERTIONS")
    print("="*80)

    # Assertions to verify improvements
    baseline_p5_avg = sum(r["p5"] for r in baseline_results) / len(baseline_results)
    all_features_p5_avg = sum(r["p5"] for r in all_results["All Features"]) / len(TEST_CASES)

    print(f"\nBaseline P@5 average: {baseline_p5_avg:.2f}")
    print(f"All Features P@5 average: {all_features_p5_avg:.2f}")

    # All Features should improve or maintain baseline precision
    assert all_features_p5_avg >= baseline_p5_avg, \
        f"All Features should improve or maintain baseline precision: {baseline_p5_avg:.2f} -> {all_features_p5_avg:.2f}"

    improvement = ((all_features_p5_avg - baseline_p5_avg) / baseline_p5_avg * 100) if baseline_p5_avg > 0 else 0
    print(f"✓ Overall improvement: {improvement:.1f}%")

    # Check specific query improvements
    # Heading boost should help kubernetes query
    kubernetes_baseline = next(r for r in baseline_results if r["query_name"] == "kubernetes_architecture_heading")
    kubernetes_heading = next(r for r in all_results["With Heading Boost"] if r["query_name"] == "kubernetes_architecture_heading")
    assert kubernetes_heading["p5"] >= kubernetes_baseline["p5"], \
        "Heading boost should improve kubernetes architecture query"
    print(f"✓ Heading boost improved kubernetes query: {kubernetes_baseline['p5']:.2f} -> {kubernetes_heading['p5']:.2f}")

    # Tag boost should help ML query
    ml_baseline = next(r for r in baseline_results if r["query_name"] == "machine_learning_deployment_tag")
    ml_tag = next(r for r in all_results["With Tag Boost"] if r["query_name"] == "machine_learning_deployment_tag")
    assert ml_tag["p5"] >= ml_baseline["p5"], \
        "Tag boost should improve machine learning query"
    print(f"✓ Tag boost improved ML query: {ml_baseline['p5']:.2f} -> {ml_tag['p5']:.2f}")

    # MMR should improve diversity for API query
    api_baseline = next(r for r in baseline_results if r["query_name"] == "api_implementation_diversity")
    api_mmr = next(r for r in all_results["With MMR Diversity"] if r["query_name"] == "api_implementation_diversity")
    assert api_mmr["div10"] >= api_baseline["div10"], \
        "MMR should improve diversity for API query"
    print(f"✓ MMR improved diversity for API query: {api_baseline['div10']:.2f} -> {api_mmr['div10']:.2f}")

    print("\n✓ All assertions passed!")


@pytest.mark.manual
def test_feature_independence(tmp_path: Path):
    """Test that features work independently without conflicts."""
    vault = tmp_path / "vault"
    create_test_vault(vault)

    config = VaultConfig(
        vault_root=vault,
        index_dir=tmp_path / "index",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        image_analysis_provider="off",
        offline_mode=False,
        heading_boost_enabled=True,
        tag_boost_enabled=True,
        backlink_boost_enabled=False,
        recency_boost_enabled=False,
    )

    # Index
    indexer = Indexer(config)
    indexer.scan(full=True)

    # Search with all features
    retriever = Retriever(config)
    results = retriever.search("machine learning kubernetes deployment", k=10)

    # Apply MMR
    mmr = MMRDiversifier(config=DiversityConfig(enabled=True))
    diverse_results = mmr.diversify(results, k=10)

    # Verify results have expected metadata
    for r in diverse_results:
        # Should have boost metadata if boosts applied
        if "heading_boost" in r.metadata or "tag_boost" in r.metadata:
            assert "original_score" in r.metadata, "Boosted results should have original_score"

        # All results should have basic metadata
        assert "full_path" in r.metadata or r.source_ref.rel_path, "Results should have path info"

    print(f"✓ Retrieved {len(diverse_results)} results with all features enabled")
    print(f"  Results with heading boost: {sum(1 for r in diverse_results if 'heading_boost' in r.metadata)}")
    print(f"  Results with tag boost: {sum(1 for r in diverse_results if 'tag_boost' in r.metadata)}")

    # Check diversity
    doc_ids = {f"{r.source_ref.vault_id}:{r.source_ref.rel_path}" for r in diverse_results}
    print(f"  Unique documents in top-10: {len(doc_ids)}")

    assert len(doc_ids) >= 3, "Should have at least 3 different documents in top-10"
    print("✓ Feature independence verified")


@pytest.mark.manual
def test_mmr_diversity_effectiveness(tmp_path: Path):
    """Test that MMR effectively increases document diversity."""
    vault = tmp_path / "vault"
    create_test_vault(vault)

    config = VaultConfig(
        vault_root=vault,
        index_dir=tmp_path / "index",
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_device="cpu",
        image_analysis_provider="off",
        offline_mode=False,
        heading_boost_enabled=False,
        tag_boost_enabled=False,
    )

    # Index
    indexer = Indexer(config)
    indexer.scan(full=True)

    # Search
    retriever = Retriever(config)
    query = "database schema design patterns"
    results = retriever.search(query, k=20)

    # Calculate diversity without MMR
    results_no_mmr = results[:10]
    doc_ids_no_mmr = {f"{r.source_ref.vault_id}:{r.source_ref.rel_path}" for r in results_no_mmr}
    diversity_no_mmr = len(doc_ids_no_mmr) / len(results_no_mmr)

    # Apply MMR
    mmr = MMRDiversifier(
        config=DiversityConfig(
            enabled=True,
            max_per_document=2,  # Max 2 chunks per document
        )
    )
    results_with_mmr = mmr.diversify(results, k=10)
    doc_ids_with_mmr = {f"{r.source_ref.vault_id}:{r.source_ref.rel_path}" for r in results_with_mmr}
    diversity_with_mmr = len(doc_ids_with_mmr) / len(results_with_mmr)

    print(f"\nQuery: {query}")
    print(f"Diversity without MMR: {diversity_no_mmr:.2f} ({len(doc_ids_no_mmr)} unique docs)")
    print(f"Diversity with MMR: {diversity_with_mmr:.2f} ({len(doc_ids_with_mmr)} unique docs)")

    # Count chunks per document
    print("\nChunks per document (without MMR):")
    from collections import Counter
    doc_counts_no_mmr = Counter([r.source_ref.rel_path for r in results_no_mmr])
    for doc, count in doc_counts_no_mmr.most_common():
        print(f"  {doc}: {count}")

    print("\nChunks per document (with MMR):")
    doc_counts_with_mmr = Counter([r.source_ref.rel_path for r in results_with_mmr])
    for doc, count in doc_counts_with_mmr.most_common():
        print(f"  {doc}: {count}")

    # MMR should increase diversity
    assert diversity_with_mmr >= diversity_no_mmr, \
        f"MMR should maintain or improve diversity: {diversity_no_mmr:.2f} -> {diversity_with_mmr:.2f}"

    # No document should have more than max_per_document chunks
    for doc, count in doc_counts_with_mmr.items():
        assert count <= 2, f"Document {doc} has {count} chunks, exceeds max_per_document=2"

    print(f"\n✓ MMR diversity improvement: {((diversity_with_mmr - diversity_no_mmr) / diversity_no_mmr * 100):.1f}%")
    print("✓ MMR effectiveness verified")
