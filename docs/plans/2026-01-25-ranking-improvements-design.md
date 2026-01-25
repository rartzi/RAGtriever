# RAGtriever Ranking Improvements Design

**Date:** 2026-01-25
**Status:** Draft

## Overview

This document describes architectural improvements to RAGtriever's search result ranking system. The goal is to improve search result quality by 25-40% through three complementary components:

1. **Reciprocal Rank Fusion (RRF)** - Replace weighted scoring with rank-based fusion
2. **Backlink Boost** - Leverage the existing link graph to surface authoritative documents
3. **Tiered Recency Boost** - Favor recently modified content when relevance is similar

These improvements work together to produce more relevant, contextually appropriate search results without requiring additional model inference.

## Current State

The current hybrid retrieval system in `HybridRanker` uses weighted scoring to merge vector and lexical search results:

```python
combined_score = w_vec * vector_score + w_lex * lexical_score
```

**Default weights:** `w_vec=1.0`, `w_lex=0.5`

**Problems with the current approach:**

1. **Scale mismatch** - Vector similarity scores (cosine, 0-1 range) and lexical scores (BM25-style, unbounded) operate on different scales. Direct weighted combination produces inconsistent rankings.

2. **Score sensitivity** - Small changes in weights cause large ranking shifts because the approach is sensitive to the raw score magnitudes.

3. **No context signals** - The ranker ignores valuable signals already present in the index: link graph structure and document recency.

The optional `CrossEncoderReranker` provides quality improvement through model-based reranking, but it operates after the initial merge and cannot fix fundamental issues with candidate selection.

## Proposed Architecture

The improved ranking pipeline introduces RRF fusion and boost adjustments between candidate retrieval and optional reranking:

```
Query → Vector Search ──┐
      → Lexical Search ──┼→ RRF Fusion → Boost Adjuster → [Reranker] → Results
```

**Processing stages:**

1. **Candidate Retrieval** - Vector and lexical searches run in parallel, each returning ranked candidate lists
2. **RRF Fusion** - Merge candidates using reciprocal rank fusion (rank-based, score-agnostic)
3. **Boost Adjuster** - Apply backlink and recency multipliers to fused scores
4. **Reranker (Optional)** - Cross-encoder reranking if enabled

This architecture preserves backward compatibility. Users can disable RRF to fall back to weighted scoring, and boosts can be individually toggled.

## Component 1: Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion combines ranked lists by summing reciprocal ranks across all input lists. Unlike weighted scoring, RRF is agnostic to the scale of underlying scores.

**Formula:**

```
RRF_score(d) = Σ (1 / (k + rank_i(d)))
```

Where:
- `d` is a document
- `rank_i(d)` is the rank of document `d` in list `i` (1-indexed)
- `k` is a smoothing constant (default: 60)
- Documents not present in a list receive no contribution from that list

**Why k=60?**

The constant `k` controls how much top ranks are emphasized over lower ranks. With `k=60`:
- Rank 1 contributes `1/61 ≈ 0.0164`
- Rank 10 contributes `1/70 ≈ 0.0143`
- Rank 100 contributes `1/160 ≈ 0.00625`

This provides a balanced decay that rewards top positions without completely ignoring lower-ranked candidates. The value 60 is well-established in IR literature.

**Implementation location:** `src/ragtriever/retrieval/hybrid.py`

**Key behavior:**
- Replace the weighted merge in `HybridRanker.merge()`, not add to it
- No score normalization required
- Documents appearing in both lists get higher scores than single-list documents

## Component 2: Backlink Boost

RAGtriever already maintains a `links` table tracking `[[wikilinks]]` and `![[embeds]]` between documents. Documents with many incoming links tend to be authoritative or central to the knowledge base.

**Boost formula:**

```
boosted_score = base_score * (1 + weight * min(backlinks, cap))
```

**Parameters:**
- `weight` (default: 0.1) - Contribution per backlink
- `cap` (default: 10) - Maximum backlinks considered

**Effect with defaults:**
- 0 backlinks: 1.0x (no boost)
- 5 backlinks: 1.5x boost
- 10+ backlinks: 2.0x boost (capped)

**Query pattern:**

```sql
SELECT target_path, COUNT(*) as backlink_count
FROM links
WHERE target_path IN (/* candidate doc paths */)
GROUP BY target_path
```

**Implementation location:** `src/ragtriever/retrieval/retriever.py`

This leverages the existing link graph with minimal overhead. The query runs once per search against candidate documents only.

## Component 3: Tiered Recency Boost

Knowledge bases often contain both evergreen reference material and time-sensitive notes. The recency boost provides a configurable preference for recently modified content.

**Boost tiers (default):**

| Tier | Age Threshold | Multiplier |
|------|---------------|------------|
| Fresh | < 14 days | 1.20x |
| Recent | < 60 days | 1.10x |
| Standard | < 180 days | 1.00x |
| Older | >= 180 days | 0.95x |

**Formula:**

```python
age_days = (now - modified_at).days

if age_days < fresh_days:
    multiplier = 1.20
elif age_days < recent_days:
    multiplier = 1.10
elif age_days < old_days:
    multiplier = 1.00
else:
    multiplier = 0.95
```

**Rationale:**

- Fresh content (< 2 weeks) often reflects current focus and active work
- Recent content (< 2 months) remains contextually relevant
- Standard content (< 6 months) receives no adjustment
- Older content receives slight penalty but is never excluded

**Per-vault configuration:**

Different vaults may have different recency characteristics. A daily journal vault benefits from strong recency bias, while a reference vault may want it disabled entirely.

**Implementation location:** `src/ragtriever/retrieval/retriever.py`

The `modified_at` timestamp is already stored in chunk metadata and requires no additional database queries.

## Configuration

All new features are configurable via `config.toml` under the `[retrieval]` section:

```toml
[retrieval]
# Fusion algorithm: "rrf" (recommended) or "weighted" (legacy)
fusion_algorithm = "rrf"

# RRF smoothing constant (only used when fusion_algorithm = "rrf")
rrf_k = 60

# Backlink boost settings
backlink_boost_weight = 0.1    # Per-backlink contribution (0 to disable)
backlink_boost_cap = 10        # Maximum backlinks considered

# Recency boost settings
recency_boost_enabled = true   # Enable/disable recency boost
recency_fresh_days = 14        # Days considered "fresh"
recency_recent_days = 60       # Days considered "recent"
recency_old_days = 180         # Days before "older" penalty
```

**Backward compatibility:**

- `fusion_algorithm = "weighted"` preserves current behavior
- `backlink_boost_weight = 0` disables backlink boost
- `recency_boost_enabled = false` disables recency boost

## Evaluation

The ranking improvements will be evaluated against the test vault with a fixed set of queries.

**Test setup:**
- 10 representative queries covering different search patterns
- Test vault: `test_vault_comprehensive` (or equivalent fixture)
- Baseline: Current weighted scoring without boosts

**Metrics:**

1. **Mean Reciprocal Rank (MRR)** - Average of `1/rank` for the first relevant result
2. **Precision@3** - Fraction of relevant results in top 3

**Success criteria:**
- MRR improvement >= 10% over baseline
- No regressions on any individual query
- Precision@3 maintained or improved

**Evaluation process:**

1. Generate baseline rankings for all test queries
2. Apply each component individually, measure impact
3. Apply all components together, measure combined impact
4. Document per-query and aggregate results

## Implementation Plan

The implementation follows a modular approach, adding each component independently with full test coverage.

### Phase 1: RRF Fusion

**Files:** `src/ragtriever/retrieval/hybrid.py`

1. Add `rrf_merge()` method to `HybridRanker`
2. Add `fusion_algorithm` parameter to `merge()` method
3. Update existing tests to cover both algorithms
4. Add dedicated RRF unit tests

### Phase 2: Backlink Boost

**Files:** `src/ragtriever/retrieval/retriever.py`, `src/ragtriever/store/libsql_store.py`

1. Add `get_backlink_counts()` method to store
2. Add `_apply_backlink_boost()` method to `Retriever`
3. Integrate into search pipeline after fusion
4. Add unit tests for boost calculation

### Phase 3: Recency Boost

**Files:** `src/ragtriever/retrieval/retriever.py`

1. Add `_apply_recency_boost()` method to `Retriever`
2. Integrate into search pipeline after backlink boost
3. Add unit tests for tier boundaries

### Phase 4: Configuration

**Files:** `src/ragtriever/config.py`

1. Add new config options with defaults
2. Add validation for numeric ranges
3. Update config loading to pass options to retriever
4. Update `examples/config.toml.example`

### Phase 5: Documentation

**Files:** `CLAUDE.md`

1. Update Hybrid Retrieval Strategy section
2. Document new configuration options
3. Add performance characteristics

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| RRF degrades certain query types | Medium | High | A/B testing, fallback to weighted |
| Backlink boost over-emphasizes hubs | Low | Medium | Configurable cap, per-vault tuning |
| Recency bias harms reference lookups | Medium | Medium | Disable per-vault, modest multipliers |
| Performance regression | Low | Low | All operations are O(n) on candidates |

## References

- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. SIGIR.
- RAGtriever link graph implementation: `src/ragtriever/store/libsql_store.py` (links table)
- Current ranker implementation: `src/ragtriever/retrieval/hybrid.py`
