"""Query with baseline config to see PDF content."""

import dataclasses
import json
from pathlib import Path
from mneme.config import load_config
from mneme.retrieval.retriever import Retriever


def main():
    config_path = Path("config.toml")
    cfg = load_config(config_path)

    # Baseline (no boosts)
    baseline_cfg = dataclasses.replace(
        cfg,
        heading_boost_enabled=False,
        tag_boost_enabled=False,
        backlink_boost_enabled=False,
        recency_boost_enabled=False,
    )
    baseline_retriever = Retriever(baseline_cfg)
    baseline_retriever.diversifier.config.enabled = False

    query = "how should AstraZeneca data sensitivity be handled with the AI Gateway"
    results = baseline_retriever.search(query, k=10)

    # Print results as JSON
    output = []
    for result in results:
        output.append({
            "rank": len(output) + 1,
            "score": result.score,
            "document": result.source_ref.rel_path,
            "snippet": result.snippet[:300],
            "heading": result.metadata.get("heading", ""),
        })

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
