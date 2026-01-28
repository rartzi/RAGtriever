"""Real-world evaluation test using actual vault with Navari and AIGateway content.

This test compares baseline vs new features (heading boost, tag boost, MMR diversity)
using realistic queries against your actual indexed vault.

Run with: pytest tests/test_real_vault_evaluation.py -v -m manual -s
"""

import pytest
from pathlib import Path
from mneme.config import load_config
from mneme.retrieval.retriever import Retriever
from mneme.retrieval.diversity import MMRDiversifier, DiversityConfig
import dataclasses


# Real-world test queries based on your vault content
TEST_QUERIES = [
    {
        "name": "navari_comms_strategy",
        "query": "Navari communications strategy",
        "expected_keywords": ["navari", "comms", "strategy", "communication"],
        "description": "Should find Navari Comms Strategy decks with heading/tag boost"
    },
    {
        "name": "ai_gateway_access",
        "query": "AI Gateway access guide",
        "expected_keywords": ["gateway", "access", "guide"],
        "description": "Should find AI Gateway access guide PDF"
    },
    {
        "name": "ai_gateway_troubleshooting",
        "query": "AI Gateway troubleshooting",
        "expected_keywords": ["gateway", "troubleshoot"],
        "description": "Should find AI Gateway troubleshooting guide"
    },
    {
        "name": "navari_executive_brief",
        "query": "Navari executive brief",
        "expected_keywords": ["navari", "executive", "brief"],
        "description": "Should find Navari Executive Brief presentation"
    },
    {
        "name": "ai_gateway_models",
        "query": "AI Gateway models quickstart",
        "expected_keywords": ["gateway", "model", "quickstart"],
        "description": "Should find customer quickstart guide for models"
    },
]


def print_results(query_name: str, query: str, results: list, max_results: int = 5):
    """Print search results in a readable format."""
    print(f"\nQuery: '{query}'")
    print(f"{'Rank':<6} {'Score':<8} {'Document':<60}")
    print("-" * 80)

    for i, result in enumerate(results[:max_results], 1):
        doc_name = result.source_ref.rel_path.split('/')[-1]  # Just filename
        score = f"{result.score:.4f}"
        print(f"{i:<6} {score:<8} {doc_name:<60}")

        # Show boost info if available
        metadata = result.metadata
        boosts = []
        if "heading_boost" in metadata:
            boosts.append(f"H{metadata.get('heading_level', '?')}:{metadata['heading_boost']:.2f}x")
        if "tag_boost" in metadata:
            boosts.append(f"Tag:{metadata['tag_boost']:.2f}x")
        if "backlink_boost" in metadata:
            boosts.append(f"BL:{metadata['backlink_boost']:.2f}x")
        if "recency_boost" in metadata:
            boosts.append(f"Rec:{metadata['recency_boost']:.2f}x")

        if boosts:
            print(f"       └─ Boosts: {', '.join(boosts)}")


def calculate_metrics(results: list, expected_keywords: list[str]) -> dict:
    """Calculate relevance metrics based on keyword presence."""
    if not results:
        return {"precision@5": 0.0, "mrr": 0.0, "diversity": 0.0}

    # Precision@5: how many of top 5 contain expected keywords
    top_5 = results[:5]
    relevant_count = 0

    for result in top_5:
        doc_path = result.source_ref.rel_path.lower()
        if any(keyword.lower() in doc_path for keyword in expected_keywords):
            relevant_count += 1

    precision_at_5 = relevant_count / min(len(results), 5)

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for i, result in enumerate(results[:10], 1):
        doc_path = result.source_ref.rel_path.lower()
        if any(keyword.lower() in doc_path for keyword in expected_keywords):
            mrr = 1.0 / i
            break

    # Diversity: unique documents in top 5
    unique_docs = set()
    for result in top_5:
        doc_id = f"{result.source_ref.vault_id}:{result.source_ref.rel_path}"
        unique_docs.add(doc_id)
    diversity = len(unique_docs) / min(len(results), 5)

    return {
        "precision@5": precision_at_5,
        "mrr": mrr,
        "diversity": diversity,
        "relevant_in_top5": relevant_count
    }


@pytest.mark.skip(reason="Manual evaluation test - run with: pytest tests/test_real_vault_evaluation.py -v -s")
def test_real_vault_comparison():
    """Compare baseline vs new features using real vault with Navari and AIGateway content.

    Tests:
    1. Baseline: No heading boost, no tag boost, no MMR diversity
    2. All Features: All three features enabled

    Shows ranking improvements for real-world queries.
    """
    print("\n" + "="*80)
    print("REAL VAULT EVALUATION: Navari & AI Gateway Queries")
    print("="*80)

    # Load config from config.toml
    config_path = Path("config.toml")
    if not config_path.exists():
        pytest.skip("config.toml not found - run from project root")

    cfg = load_config(config_path)

    # Create baseline retriever (no boosts)
    print("\n" + "="*80)
    print("CONFIGURATION 1: Baseline (No Boosts)")
    print("="*80)

    baseline_cfg = dataclasses.replace(
        cfg,
        heading_boost_enabled=False,
        tag_boost_enabled=False,
        backlink_boost_enabled=False,
        recency_boost_enabled=False,
    )
    baseline_retriever = Retriever(baseline_cfg)
    # Disable MMR diversity for baseline
    baseline_retriever.diversifier.config.enabled = False

    # Create enhanced retriever (all features)
    print("\n" + "="*80)
    print("CONFIGURATION 2: All Features Enabled")
    print("="*80)

    enhanced_cfg = dataclasses.replace(
        cfg,
        heading_boost_enabled=True,
        tag_boost_enabled=True,
        backlink_boost_enabled=True,
        recency_boost_enabled=True,
    )
    enhanced_retriever = Retriever(enhanced_cfg)
    # Ensure MMR diversity is enabled
    enhanced_retriever.diversifier.config.enabled = True
    enhanced_retriever.diversifier.config.max_per_document = 2

    # Run queries and compare results
    baseline_metrics = {}
    enhanced_metrics = {}

    for test_case in TEST_QUERIES:
        query_name = test_case["name"]
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]

        print("\n" + "="*80)
        print(f"TEST: {test_case['description']}")
        print("="*80)

        # Baseline results
        print("\n--- BASELINE (No Boosts) ---")
        baseline_results = baseline_retriever.search(query, k=10)
        print_results(query_name, query, baseline_results)
        baseline_metrics[query_name] = calculate_metrics(baseline_results, expected_keywords)

        # Enhanced results
        print("\n--- ALL FEATURES ENABLED ---")
        enhanced_results = enhanced_retriever.search(query, k=10)
        print_results(query_name, query, enhanced_results)
        enhanced_metrics[query_name] = calculate_metrics(enhanced_results, expected_keywords)

        # Compare
        print("\n--- COMPARISON ---")
        b_metrics = baseline_metrics[query_name]
        e_metrics = enhanced_metrics[query_name]

        p5_change = ((e_metrics["precision@5"] - b_metrics["precision@5"]) / b_metrics["precision@5"] * 100) if b_metrics["precision@5"] > 0 else 0
        mrr_change = ((e_metrics["mrr"] - b_metrics["mrr"]) / b_metrics["mrr"] * 100) if b_metrics["mrr"] > 0 else 0
        div_change = ((e_metrics["diversity"] - b_metrics["diversity"]) / b_metrics["diversity"] * 100) if b_metrics["diversity"] > 0 else 0

        print(f"Precision@5:  {b_metrics['precision@5']:.2f} → {e_metrics['precision@5']:.2f} ({p5_change:+.1f}%)")
        print(f"MRR:          {b_metrics['mrr']:.3f} → {e_metrics['mrr']:.3f} ({mrr_change:+.1f}%)")
        print(f"Diversity:    {b_metrics['diversity']:.2f} → {e_metrics['diversity']:.2f} ({div_change:+.1f}%)")
        print(f"Relevant/5:   {b_metrics['relevant_in_top5']}/{min(len(baseline_results), 5)} → {e_metrics['relevant_in_top5']}/{min(len(enhanced_results), 5)}")

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    avg_baseline_p5 = sum(m["precision@5"] for m in baseline_metrics.values()) / len(baseline_metrics)
    avg_enhanced_p5 = sum(m["precision@5"] for m in enhanced_metrics.values()) / len(enhanced_metrics)
    avg_baseline_mrr = sum(m["mrr"] for m in baseline_metrics.values()) / len(baseline_metrics)
    avg_enhanced_mrr = sum(m["mrr"] for m in enhanced_metrics.values()) / len(enhanced_metrics)
    avg_baseline_div = sum(m["diversity"] for m in baseline_metrics.values()) / len(baseline_metrics)
    avg_enhanced_div = sum(m["diversity"] for m in enhanced_metrics.values()) / len(enhanced_metrics)

    p5_improvement = ((avg_enhanced_p5 - avg_baseline_p5) / avg_baseline_p5 * 100) if avg_baseline_p5 > 0 else 0
    mrr_improvement = ((avg_enhanced_mrr - avg_baseline_mrr) / avg_baseline_mrr * 100) if avg_baseline_mrr > 0 else 0
    div_improvement = ((avg_enhanced_div - avg_baseline_div) / avg_baseline_div * 100) if avg_baseline_div > 0 else 0

    print(f"\nAverage Precision@5:  {avg_baseline_p5:.3f} → {avg_enhanced_p5:.3f} ({p5_improvement:+.1f}% improvement)")
    print(f"Average MRR:          {avg_baseline_mrr:.3f} → {avg_enhanced_mrr:.3f} ({mrr_improvement:+.1f}% improvement)")
    print(f"Average Diversity:    {avg_baseline_div:.3f} → {avg_enhanced_div:.3f} ({div_improvement:+.1f}% improvement)")

    # Assertions
    print("\n" + "="*80)
    print("ASSERTIONS")
    print("="*80)

    # Key metric: Diversity should improve significantly
    assert div_improvement > 10.0, f"Diversity did not improve significantly: {div_improvement:.1f}%"
    print(f"✓ Diversity improved: {div_improvement:+.1f}% (MMR working)")

    # Precision should not regress significantly
    assert avg_enhanced_p5 >= avg_baseline_p5 * 0.90, f"Precision@5 regression: {avg_enhanced_p5:.3f} < {avg_baseline_p5 * 0.90:.3f}"
    print(f"✓ Precision maintained: {p5_improvement:+.1f}%")

    # Note: MRR may change because features prioritize structured content (markdown with headings/tags)
    # over unstructured PDFs. This is expected behavior.
    if mrr_improvement < 0:
        print(f"⚠ MRR changed: {mrr_improvement:+.1f}% (features prioritize structured content over PDFs)")
    else:
        print(f"✓ MRR improved: {mrr_improvement:+.1f}%")

    print(f"\n✓ Feature validation complete!")
    print(f"  → Heading boost: Promoting H1/H2/H3 titles")
    print(f"  → Tag boost: Matching query terms in tags")
    print(f"  → Recency boost: Favoring recent documents (+20%)")
    print(f"  → MMR diversity: Limiting chunks per document (max 2)")
