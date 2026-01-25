"""Manual query test for real-world questions."""

import dataclasses
from pathlib import Path
from ragtriever.config import load_config
from ragtriever.retrieval.retriever import Retriever


def print_results(title: str, results: list, max_results: int = 10):
    """Print search results with boosts."""
    print(f"\n{title}")
    print("="*100)
    print(f"{'Rank':<6} {'Score':<8} {'Document':<70}")
    print("-"*100)

    for i, result in enumerate(results[:max_results], 1):
        doc_name = result.source_ref.rel_path
        score = f"{result.score:.4f}"
        print(f"{i:<6} {score:<8} {doc_name:<70}")

        # Show snippet
        snippet = result.snippet[:150].replace('\n', ' ')
        print(f"       {snippet}...")

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
        print()


def main():
    # Load config
    config_path = Path("config.toml")
    cfg = load_config(config_path)

    # Real-world query
    query = "how should AstraZeneca data sensitivity be handled with the AI Gateway"
    print("\n" + "="*100)
    print(f"QUERY: {query}")
    print("="*100)

    # Baseline (no boosts)
    print("\n" + "="*100)
    print("BASELINE CONFIGURATION (No Boosts)")
    print("="*100)

    baseline_cfg = dataclasses.replace(
        cfg,
        heading_boost_enabled=False,
        tag_boost_enabled=False,
        backlink_boost_enabled=False,
        recency_boost_enabled=False,
    )
    baseline_retriever = Retriever(baseline_cfg)
    baseline_retriever.diversifier.config.enabled = False

    baseline_results = baseline_retriever.search(query, k=10)
    print_results("BASELINE RESULTS", baseline_results)

    # Enhanced (all features)
    print("\n" + "="*100)
    print("ENHANCED CONFIGURATION (All Features Enabled)")
    print("="*100)

    enhanced_cfg = dataclasses.replace(
        cfg,
        heading_boost_enabled=True,
        tag_boost_enabled=True,
        backlink_boost_enabled=True,
        recency_boost_enabled=True,
    )
    enhanced_retriever = Retriever(enhanced_cfg)
    enhanced_retriever.diversifier.config.enabled = True
    enhanced_retriever.diversifier.config.max_per_document = 2

    enhanced_results = enhanced_retriever.search(query, k=10)
    print_results("ENHANCED RESULTS (with Heading, Tag, Recency boosts + MMR diversity)", enhanced_results)

    # Comparison
    print("\n" + "="*100)
    print("COMPARISON")
    print("="*100)

    print("\nTop 3 Documents - Baseline:")
    for i, result in enumerate(baseline_results[:3], 1):
        print(f"  {i}. {result.source_ref.rel_path}")

    print("\nTop 3 Documents - Enhanced:")
    for i, result in enumerate(enhanced_results[:3], 1):
        print(f"  {i}. {result.source_ref.rel_path}")

    # Count unique documents
    baseline_docs = set(r.source_ref.rel_path for r in baseline_results[:10])
    enhanced_docs = set(r.source_ref.rel_path for r in enhanced_results[:10])

    print(f"\nUnique documents in top 10:")
    print(f"  Baseline: {len(baseline_docs)}")
    print(f"  Enhanced: {len(enhanced_docs)} ({'+' if len(enhanced_docs) > len(baseline_docs) else ''}{len(enhanced_docs) - len(baseline_docs)})")


if __name__ == "__main__":
    main()
