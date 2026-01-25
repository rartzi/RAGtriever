"""Score boosting based on document signals (backlinks, recency)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from ..models import SearchResult


class LinkStore(Protocol):
    """Protocol for querying link/backlink information."""

    def get_backlink_counts(self, doc_ids: list[str] | None = None) -> dict[str, int]:
        """Return mapping of doc_id -> backlink count."""
        ...


@dataclass
class BoostConfig:
    """Configuration for score boosting."""

    # Backlink boost
    backlink_enabled: bool = True
    backlink_weight: float = 0.1  # 10% per backlink
    backlink_cap: int = 10  # Max backlinks to count (caps at 2x boost)

    # Recency boost (tiered)
    recency_enabled: bool = True
    recency_fresh_days: int = 14  # < 14 days = fresh
    recency_recent_days: int = 60  # < 60 days = recent
    recency_old_days: int = 180  # > 180 days = old
    recency_fresh_boost: float = 1.20  # 20% boost for fresh
    recency_recent_boost: float = 1.10  # 10% boost for recent
    recency_old_penalty: float = 0.95  # 5% penalty for old

    # Tag boost
    tag_boost_enabled: bool = True
    tag_boost_weight: float = 0.15  # 15% boost per matching tag
    tag_boost_cap: int = 3  # Max tags counted (caps at 45% boost)


@dataclass
class BoostAdjuster:
    """Apply score boosts based on document signals."""

    config: BoostConfig

    def apply_boosts(
        self,
        results: list[SearchResult],
        backlink_counts: dict[str, int] | None = None,
        query: str | None = None,
    ) -> list[SearchResult]:
        """Apply configured boosts to search results.

        Args:
            results: Search results to boost
            backlink_counts: Pre-fetched backlink counts by doc_id (optional)
            query: Original search query for tag matching (optional)

        Returns:
            Results with adjusted scores, re-sorted by boosted score
        """
        if not results:
            return []

        boosted = []
        for r in results:
            score = r.score
            boost_info: dict[str, float | int] = {}

            # Backlink boost
            if self.config.backlink_enabled and backlink_counts:
                doc_id = self._extract_doc_id(r)
                count = backlink_counts.get(doc_id, 0)
                if count > 0:
                    capped = min(count, self.config.backlink_cap)
                    backlink_boost = 1 + (self.config.backlink_weight * capped)
                    score *= backlink_boost
                    boost_info["backlink_count"] = count
                    boost_info["backlink_boost"] = backlink_boost

            # Recency boost
            if self.config.recency_enabled:
                mtime = self._extract_mtime(r)
                if mtime:
                    recency_boost = self._calculate_recency_boost(mtime)
                    score *= recency_boost
                    boost_info["recency_boost"] = recency_boost

            # Tag boost
            if self.config.tag_boost_enabled and query:
                tag_boost, tag_matches = self._calculate_tag_boost(r, query)
                if tag_boost > 1.0:
                    score *= tag_boost
                    boost_info["tag_boost"] = tag_boost
                    boost_info["tag_matches"] = tag_matches

            # Create boosted result with updated metadata
            new_metadata = {**r.metadata, "original_score": r.score, **boost_info}

            boosted.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    score=score,
                    snippet=r.snippet,
                    source_ref=r.source_ref,
                    metadata=new_metadata,
                )
            )

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted

    def _extract_doc_id(self, result: SearchResult) -> str:
        """Extract doc_id from result metadata or construct from source_ref."""
        # Try metadata first
        if "doc_id" in result.metadata:
            return str(result.metadata["doc_id"])
        # Fallback: construct from vault_id + rel_path
        # Note: This won't match the hashed doc_id in the store, but provides
        # a consistent key for backlink lookups when doc_id isn't in metadata
        return f"{result.source_ref.vault_id}:{result.source_ref.rel_path}"

    def _extract_mtime(self, result: SearchResult) -> datetime | None:
        """Extract modification time from result metadata."""
        # Look for modified_at in ISO format
        if "modified_at" in result.metadata:
            try:
                return datetime.fromisoformat(str(result.metadata["modified_at"]))
            except (ValueError, TypeError):
                pass
        return None

    def _calculate_recency_boost(self, mtime: datetime) -> float:
        """Calculate recency boost based on document age."""
        now = datetime.now(timezone.utc)
        # Ensure mtime is timezone-aware
        if mtime.tzinfo is None:
            mtime = mtime.replace(tzinfo=timezone.utc)

        age_days = (now - mtime).days

        if age_days < self.config.recency_fresh_days:
            return self.config.recency_fresh_boost
        elif age_days < self.config.recency_recent_days:
            return self.config.recency_recent_boost
        elif age_days < self.config.recency_old_days:
            return 1.0  # Standard, no boost or penalty
        else:
            return self.config.recency_old_penalty

    def _calculate_tag_boost(self, result: SearchResult, query: str) -> tuple[float, int]:
        """Calculate boost based on tag matches.

        Args:
            result: Search result to check for tags
            query: Search query to match against tags

        Returns:
            Tuple of (boost_multiplier, number_of_matching_tags)
        """
        if not self.config.tag_boost_enabled:
            return 1.0, 0

        # Get tags from metadata - can be list or string
        tags = result.metadata.get("tags", [])
        if not tags:
            return 1.0, 0

        # Normalize tags to list if string
        if isinstance(tags, str):
            tags = [tags]

        # Tokenize query into words (lowercase)
        query_terms = set(query.lower().split())

        # Count matching tags
        matches = 0
        for tag in tags:
            # Normalize tag: remove # prefix, replace hyphens/underscores with spaces
            tag_lower = str(tag).lower().lstrip("#").replace("-", " ").replace("_", " ")
            tag_terms = set(tag_lower.split())
            if query_terms & tag_terms:  # Any overlap
                matches += 1

        if matches == 0:
            return 1.0, 0

