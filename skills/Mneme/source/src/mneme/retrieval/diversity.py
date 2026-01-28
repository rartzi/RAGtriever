"""Result diversity using Maximal Marginal Relevance (MMR)."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import SearchResult


@dataclass
class DiversityConfig:
    """Configuration for result diversity."""

    enabled: bool = True
    lambda_param: float = 0.7  # Balance relevance vs diversity (1.0 = all relevance, 0.0 = all diversity)
    max_per_document: int = 2  # Max chunks from same document


@dataclass
class MMRDiversifier:
    """Apply Maximal Marginal Relevance to diversify search results.

    MMR balances relevance with diversity by penalizing results that are
    too similar to already-selected results. This implementation uses a
    simplified document-based diversity approach rather than full
    embedding similarity:

    1. Limits results per document (max_per_document)
    2. Preserves original relevance ordering within constraints
    3. Fills remaining slots from skipped results if needed
    """

    config: DiversityConfig

    def diversify(self, results: list[SearchResult], k: int) -> list[SearchResult]:
        """Apply MMR to select diverse results.

        Args:
            results: Search results sorted by relevance score
            k: Target number of results to return

        Returns:
            Diversified results, maintaining relevance order within
            document diversity constraints
        """
        if not self.config.enabled or not results:
            return results[:k]

        selected: list[SearchResult] = []
        doc_counts: dict[str, int] = {}  # doc_id -> count
        skipped: list[SearchResult] = []  # Results skipped due to doc limit

        for result in results:
            if len(selected) >= k:
                break

            # Extract doc_id from source_ref
            doc_id = f"{result.source_ref.vault_id}:{result.source_ref.rel_path}"

            # Check document limit
            current_count = doc_counts.get(doc_id, 0)
            if current_count >= self.config.max_per_document:
                skipped.append(result)
                continue  # Skip, too many from this doc

            # Add to selected
            selected.append(result)
            doc_counts[doc_id] = current_count + 1

        # If we don't have enough, fill with remaining (relaxing constraint)
        if len(selected) < k:
            for result in skipped:
                if len(selected) >= k:
                    break
                selected.append(result)

        return selected
