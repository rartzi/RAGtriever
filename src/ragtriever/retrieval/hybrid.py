from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from ..models import SearchResult

@dataclass
class HybridRanker:
    """Merge/dedupe lexical and vector results into a single ranked list.

    TODO:
    - Add reciprocal rank fusion (RRF) or weighted scoring
    - Add recency/tag/graph boosts
    """
    w_vec: float = 1.0
    w_lex: float = 0.5

    def merge(self, vec: Sequence[SearchResult], lex: Sequence[SearchResult], k: int) -> list[SearchResult]:
        by_id: dict[str, SearchResult] = {}
        for r in lex:
            by_id[r.chunk_id] = r
        for r in vec:
            if r.chunk_id in by_id:
                # combine scores; keep richer metadata
                merged = by_id[r.chunk_id]
                by_id[r.chunk_id] = SearchResult(
                    chunk_id=r.chunk_id,
                    score=(self.w_vec * r.score + self.w_lex * merged.score),
                    snippet=r.snippet or merged.snippet,
                    source_ref=r.source_ref,
                    metadata={**merged.metadata, **r.metadata},
                )
            else:
                by_id[r.chunk_id] = SearchResult(
                    chunk_id=r.chunk_id,
                    score=self.w_vec * r.score,
                    snippet=r.snippet,
                    source_ref=r.source_ref,
                    metadata=r.metadata,
                )
        out = sorted(by_id.values(), key=lambda x: x.score, reverse=True)
        return out[:k]
