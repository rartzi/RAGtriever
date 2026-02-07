"""Cross-encoder reranking for improved search result quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models import SearchResult

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


@dataclass
class CrossEncoderReranker:
    """Rerank search results using cross-encoder model.

    Cross-encoders read query + document together for accurate relevance scoring,
    unlike bi-encoders which encode them separately. This provides 20-30% quality
    improvement at the cost of ~100-200ms latency.

    Args:
        model_name: HuggingFace model name (default: ms-marco-MiniLM-L-6-v2)
        device: Device to run model on (cpu, cuda, or mps)
    """
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
        # Defer model loading until first rerank() call to save 500ms-2s at init
        self._model = None

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_k: int
    ) -> list[SearchResult]:
        """Rerank search results using cross-encoder scores.

        Args:
            query: Search query
            results: Search results to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked results (up to top_k), with updated scores and metadata
        """
        if not results:
            return []

        # Lazy-load model on first use
        if self._model is None:
            import warnings
            warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*leaked semaphore")
            self._model = CrossEncoder(self.model_name, device=self.device)

        # Create query-document pairs
        pairs = [(query, r.snippet) for r in results]

        # Get cross-encoder scores
        scores = self._model.predict(pairs, convert_to_numpy=True)

        # Sort by cross-encoder score
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k with updated metadata
        reranked = []
        for result, score in scored[:top_k]:
            # SearchResult is frozen, so create new instance
            reranked_result = SearchResult(
                chunk_id=result.chunk_id,
                score=float(score),
                snippet=result.snippet,
                source_ref=result.source_ref,
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "reranker_score": float(score),
                    "reranked": True
                }
            )
            reranked.append(reranked_result)

        return reranked
