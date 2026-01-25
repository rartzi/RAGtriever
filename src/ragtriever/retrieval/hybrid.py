from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models import SearchResult


@dataclass
class HybridRanker:
    """Merge/dedupe lexical and vector results into a single ranked list.

    Supports two fusion strategies:
    1. Reciprocal Rank Fusion (RRF) - default, score-agnostic rank-based fusion
    2. Weighted scoring - legacy method using raw scores with configurable weights

    RRF is preferred because it:
    - Works regardless of score scale differences between retrievers
    - Naturally handles the case where one retriever returns much higher scores
    - Is a well-established technique in information retrieval literature

    TODO:
    - Add recency/tag/graph boosts
    """

    # Legacy weighted scoring parameters
    w_vec: float = 1.0
    w_lex: float = 0.5

    # RRF parameters
    use_rrf: bool = True  # Default to RRF (recommended)
    rrf_k: int = 60  # Standard RRF constant (60 is the commonly used value)

    def merge(
        self, vec: Sequence[SearchResult], lex: Sequence[SearchResult], k: int
    ) -> list[SearchResult]:
        """Merge vector and lexical search results into a single ranked list.

        Args:
            vec: Vector (semantic) search results, ordered by relevance
            lex: Lexical (FTS5) search results, ordered by relevance
            k: Maximum number of results to return

        Returns:
            Merged and deduplicated results, sorted by combined score
        """
        if self.use_rrf:
            return self._merge_rrf(vec, lex, k)
        return self._merge_weighted(vec, lex, k)

    def _merge_rrf(
        self, vec: Sequence[SearchResult], lex: Sequence[SearchResult], k: int
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion - score based on rank position, not raw scores.

        RRF Formula: score = sum(1 / (k + rank)) for each result list

        Why RRF works well:
        - Score-agnostic: handles different score scales from different retrievers
        - Rank-based: a result ranked #1 in both lists gets a high combined score
        - The constant k (default 60) dampens the effect of outlier rankings
        - Results appearing in multiple lists get boosted naturally

        Reference: Cormack, Clarke, Buettcher (2009) "Reciprocal Rank Fusion
        outperforms Condorcet and individual Rank Learning Methods"

        Args:
            vec: Vector search results (ordered by relevance, best first)
            lex: Lexical search results (ordered by relevance, best first)
            k: Maximum number of results to return

        Returns:
            Merged results sorted by RRF score (highest first)
        """
        scores: dict[str, float] = {}
        results: dict[str, SearchResult] = {}

        # Score vector results by rank position
        # rank is 0-indexed, so we use (rank + 1) to get 1-based ranking
        for rank, r in enumerate(vec):
            # RRF score contribution: 1 / (k + rank + 1)
            # Lower rank (better position) = higher contribution
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (
                self.rrf_k + rank + 1
            )
            results[r.chunk_id] = r

        # Score lexical results by rank position
        for rank, r in enumerate(lex):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (
                self.rrf_k + rank + 1
            )
            # Only store if not already present (prefer vector result's metadata)
            if r.chunk_id not in results:
                results[r.chunk_id] = r

        # Sort by RRF score (descending) and return top k
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        out = []
        for chunk_id in sorted_ids[:k]:
            r = results[chunk_id]
            # Create new SearchResult with RRF score, preserving original metadata
            out.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    score=scores[chunk_id],
                    snippet=r.snippet,
                    source_ref=r.source_ref,
                    metadata={**r.metadata, "rrf_score": scores[chunk_id]},
                )
            )
        return out

    def _merge_weighted(
        self, vec: Sequence[SearchResult], lex: Sequence[SearchResult], k: int
    ) -> list[SearchResult]:
        """Weighted score fusion - combine raw scores with configurable weights.

        This is the legacy fusion method. It multiplies each result's score by
        a weight factor and sums them for results appearing in both lists.

        Note: This method is sensitive to score scale differences between
        retrievers. Use RRF (use_rrf=True) for more robust fusion.

        Args:
            vec: Vector search results
            lex: Lexical search results
            k: Maximum number of results to return

        Returns:
            Merged results sorted by weighted score
        """
        by_id: dict[str, SearchResult] = {}

        # First, add all lexical results with weighted scores
        for r in lex:
            by_id[r.chunk_id] = SearchResult(
                chunk_id=r.chunk_id,
                score=self.w_lex * r.score,
                snippet=r.snippet,
                source_ref=r.source_ref,
                metadata=r.metadata,
            )

        # Then, process vector results
        for r in vec:
            if r.chunk_id in by_id:
                # Combine scores for results appearing in both lists
                merged = by_id[r.chunk_id]
                by_id[r.chunk_id] = SearchResult(
                    chunk_id=r.chunk_id,
                    score=(self.w_vec * r.score + merged.score),
                    snippet=r.snippet or merged.snippet,
                    source_ref=r.source_ref,
                    metadata={**merged.metadata, **r.metadata},
                )
            else:
                # Vector-only result
                by_id[r.chunk_id] = SearchResult(
                    chunk_id=r.chunk_id,
                    score=self.w_vec * r.score,
                    snippet=r.snippet,
                    source_ref=r.source_ref,
                    metadata=r.metadata,
                )

        out = sorted(by_id.values(), key=lambda x: x.score, reverse=True)
        return out[:k]
