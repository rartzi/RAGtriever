"""Unit tests for score boosting based on backlinks and recency."""

import pytest
from datetime import datetime, timezone, timedelta
from mneme.retrieval.boosts import BoostConfig, BoostAdjuster
from mneme.models import SearchResult, SourceRef


def make_result(
    chunk_id: str,
    score: float,
    doc_id: str | None = None,
    modified_at: str | None = None,
    vault_id: str = "v1",
    rel_path: str = "test.md",
) -> SearchResult:
    """Helper to create SearchResult for testing."""
    metadata: dict = {}
    if doc_id:
        metadata["doc_id"] = doc_id
    if modified_at:
        metadata["modified_at"] = modified_at
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        snippet=f"Content for {chunk_id}",
        source_ref=SourceRef(
            vault_id=vault_id,
            rel_path=rel_path,
            file_type="md",
            anchor_type="md_heading",
            anchor_ref="heading",
        ),
        metadata=metadata,
    )


class TestBoostConfig:
    """Test BoostConfig dataclass defaults."""

    def test_default_values(self):
        """Test that default config values are sensible."""
        config = BoostConfig()
        assert config.backlink_enabled is True
        assert config.backlink_weight == 0.1
        assert config.backlink_cap == 10
        assert config.recency_enabled is True
        assert config.recency_fresh_days == 14
        assert config.recency_recent_days == 60
        assert config.recency_old_days == 180
        assert config.recency_fresh_boost == 1.10  # 10% boost for fresh
        assert config.recency_recent_boost == 1.05  # 5% boost for recent
        assert config.recency_old_penalty == 0.98  # 2% penalty for old
        # Heading and tag boosts are disabled by default
        assert config.heading_boost_enabled is False
        assert config.tag_boost_enabled is False

    def test_custom_values(self):
        """Test custom config values."""
        config = BoostConfig(
            backlink_enabled=False,
            backlink_weight=0.2,
            recency_fresh_boost=1.30,
        )
        assert config.backlink_enabled is False
        assert config.backlink_weight == 0.2
        assert config.recency_fresh_boost == 1.30


class TestBoostAdjusterBacklinks:
    """Test backlink boosting functionality."""

    def test_backlink_boost_applied(self):
        """Test that backlink boost is applied correctly."""
        config = BoostConfig(backlink_weight=0.1, recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, doc_id="doc1")]
        backlink_counts = {"doc1": 5}

        boosted = adjuster.apply_boosts(results, backlink_counts)

        # 5 backlinks * 0.1 weight = 0.5 -> boost = 1.5
        # 0.8 * 1.5 = 1.2
        assert len(boosted) == 1
        assert boosted[0].score == pytest.approx(1.2)
        assert boosted[0].metadata["original_score"] == 0.8
        assert boosted[0].metadata["backlink_count"] == 5
        assert boosted[0].metadata["backlink_boost"] == pytest.approx(1.5)

    def test_backlink_capping(self):
        """Test that backlink count is capped."""
        config = BoostConfig(backlink_weight=0.1, backlink_cap=10, recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 1.0, doc_id="doc1")]
        backlink_counts = {"doc1": 50}  # More than cap

        boosted = adjuster.apply_boosts(results, backlink_counts)

        # Capped at 10 backlinks: 10 * 0.1 = 1.0 -> boost = 2.0 (max)
        assert boosted[0].score == pytest.approx(2.0)
        assert boosted[0].metadata["backlink_count"] == 50  # Original count preserved
        assert boosted[0].metadata["backlink_boost"] == pytest.approx(2.0)

    def test_no_backlinks(self):
        """Test result with no backlinks."""
        config = BoostConfig(recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, doc_id="doc1")]
        backlink_counts = {}  # No backlinks

        boosted = adjuster.apply_boosts(results, backlink_counts)

        assert boosted[0].score == pytest.approx(0.8)
        assert "backlink_count" not in boosted[0].metadata
        assert "backlink_boost" not in boosted[0].metadata

    def test_backlink_boost_disabled(self):
        """Test that backlink boost can be disabled."""
        config = BoostConfig(backlink_enabled=False, recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, doc_id="doc1")]
        backlink_counts = {"doc1": 10}

        boosted = adjuster.apply_boosts(results, backlink_counts)

        assert boosted[0].score == pytest.approx(0.8)
        assert "backlink_boost" not in boosted[0].metadata

    def test_doc_id_from_source_ref_fallback(self):
        """Test fallback when doc_id not in metadata."""
        config = BoostConfig(backlink_weight=0.1, recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        # No doc_id in metadata
        results = [make_result("chunk1", 0.8, vault_id="vault1", rel_path="notes.md")]
        # Key uses fallback format
        backlink_counts = {"vault1:notes.md": 3}

        boosted = adjuster.apply_boosts(results, backlink_counts)

        # 3 backlinks * 0.1 = 0.3 -> boost = 1.3
        assert boosted[0].score == pytest.approx(1.04)  # 0.8 * 1.3 = 1.04
        assert boosted[0].metadata["backlink_count"] == 3


class TestBoostAdjusterRecency:
    """Test recency boosting functionality."""

    def test_fresh_document_boost(self):
        """Test boost for recently modified documents."""
        config = BoostConfig(backlink_enabled=False, recency_fresh_boost=1.20)
        adjuster = BoostAdjuster(config=config)

        # 5 days old
        fresh_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        results = [make_result("chunk1", 0.8, modified_at=fresh_date)]

        boosted = adjuster.apply_boosts(results)

        assert boosted[0].score == pytest.approx(0.96)  # 0.8 * 1.20
        assert boosted[0].metadata["recency_boost"] == pytest.approx(1.20)

    def test_recent_document_boost(self):
        """Test boost for recently modified documents (14-60 days)."""
        config = BoostConfig(backlink_enabled=False, recency_recent_boost=1.10)
        adjuster = BoostAdjuster(config=config)

        # 30 days old
        recent_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        results = [make_result("chunk1", 0.8, modified_at=recent_date)]

        boosted = adjuster.apply_boosts(results)

        assert boosted[0].score == pytest.approx(0.88)  # 0.8 * 1.10
        assert boosted[0].metadata["recency_boost"] == pytest.approx(1.10)

    def test_standard_document_no_boost(self):
        """Test no boost/penalty for mid-age documents (60-180 days)."""
        config = BoostConfig(backlink_enabled=False)
        adjuster = BoostAdjuster(config=config)

        # 100 days old
        standard_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        results = [make_result("chunk1", 0.8, modified_at=standard_date)]

        boosted = adjuster.apply_boosts(results)

        assert boosted[0].score == pytest.approx(0.8)  # No change
        assert boosted[0].metadata["recency_boost"] == pytest.approx(1.0)

    def test_old_document_penalty(self):
        """Test penalty for old documents (>180 days)."""
        config = BoostConfig(backlink_enabled=False, recency_old_penalty=0.95)
        adjuster = BoostAdjuster(config=config)

        # 200 days old
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        results = [make_result("chunk1", 0.8, modified_at=old_date)]

        boosted = adjuster.apply_boosts(results)

        assert boosted[0].score == pytest.approx(0.76)  # 0.8 * 0.95
        assert boosted[0].metadata["recency_boost"] == pytest.approx(0.95)

    def test_recency_disabled(self):
        """Test that recency boost can be disabled."""
        config = BoostConfig(backlink_enabled=False, recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        fresh_date = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        results = [make_result("chunk1", 0.8, modified_at=fresh_date)]

        boosted = adjuster.apply_boosts(results)

        assert boosted[0].score == pytest.approx(0.8)
        assert "recency_boost" not in boosted[0].metadata

    def test_missing_modified_at(self):
        """Test handling when modified_at is missing."""
        config = BoostConfig(backlink_enabled=False)
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8)]  # No modified_at

        boosted = adjuster.apply_boosts(results)

        assert boosted[0].score == pytest.approx(0.8)
        assert "recency_boost" not in boosted[0].metadata

    def test_invalid_modified_at_format(self):
        """Test handling of invalid date format."""
        config = BoostConfig(backlink_enabled=False)
        adjuster = BoostAdjuster(config=config)

        result = make_result("chunk1", 0.8)
        # Add invalid date via new result with bad metadata
        result = SearchResult(
            chunk_id="chunk1",
            score=0.8,
            snippet="Content",
            source_ref=result.source_ref,
            metadata={"modified_at": "not-a-date"},
        )

        boosted = adjuster.apply_boosts([result])

        assert boosted[0].score == pytest.approx(0.8)
        assert "recency_boost" not in boosted[0].metadata

    def test_naive_datetime_handling(self):
        """Test that naive datetimes are treated as UTC."""
        config = BoostConfig(backlink_enabled=False, recency_fresh_boost=1.20)
        adjuster = BoostAdjuster(config=config)

        # Naive datetime (no timezone) - 5 days ago
        naive_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%S")
        results = [make_result("chunk1", 0.8, modified_at=naive_date)]

        boosted = adjuster.apply_boosts(results)

        # Should still get fresh boost
        assert boosted[0].score == pytest.approx(0.96)  # 0.8 * 1.20


class TestBoostAdjusterCombined:
    """Test combined backlink and recency boosting."""

    def test_combined_boosts(self):
        """Test that both boosts are applied multiplicatively."""
        config = BoostConfig(
            backlink_weight=0.1,
            recency_fresh_boost=1.20,
        )
        adjuster = BoostAdjuster(config=config)

        fresh_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        results = [make_result("chunk1", 0.5, doc_id="doc1", modified_at=fresh_date)]
        backlink_counts = {"doc1": 5}

        boosted = adjuster.apply_boosts(results, backlink_counts)

        # Backlink boost: 1 + (0.1 * 5) = 1.5
        # Recency boost: 1.20
        # Combined: 0.5 * 1.5 * 1.20 = 0.9
        assert boosted[0].score == pytest.approx(0.9)
        assert boosted[0].metadata["backlink_boost"] == pytest.approx(1.5)
        assert boosted[0].metadata["recency_boost"] == pytest.approx(1.20)

    def test_resorting_after_boost(self):
        """Test that results are re-sorted after boosting."""
        config = BoostConfig(
            backlink_weight=0.2,
            recency_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # Original order: chunk1 (0.9), chunk2 (0.8)
        # After boost: chunk1 (0.9), chunk2 (0.8 * 1.4 = 1.12)
        results = [
            make_result("chunk1", 0.9, doc_id="doc1"),
            make_result("chunk2", 0.8, doc_id="doc2"),
        ]
        backlink_counts = {"doc2": 2}  # Only doc2 has backlinks

        boosted = adjuster.apply_boosts(results, backlink_counts)

        # chunk2 should now be first
        assert boosted[0].chunk_id == "chunk2"
        assert boosted[0].score == pytest.approx(1.12)
        assert boosted[1].chunk_id == "chunk1"
        assert boosted[1].score == pytest.approx(0.9)


class TestBoostAdjusterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_results(self):
        """Test handling of empty results list."""
        config = BoostConfig()
        adjuster = BoostAdjuster(config=config)

        boosted = adjuster.apply_boosts([], {})

        assert boosted == []

    def test_none_backlink_counts(self):
        """Test handling when backlink_counts is None."""
        config = BoostConfig(recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, doc_id="doc1")]

        boosted = adjuster.apply_boosts(results, None)

        assert boosted[0].score == pytest.approx(0.8)
        assert "backlink_boost" not in boosted[0].metadata

    def test_preserves_existing_metadata(self):
        """Test that existing metadata is preserved."""
        config = BoostConfig(recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        result = SearchResult(
            chunk_id="chunk1",
            score=0.8,
            snippet="Content",
            source_ref=SourceRef(
                vault_id="v1",
                rel_path="test.md",
                file_type="md",
                anchor_type="md_heading",
                anchor_ref="h1",
            ),
            metadata={"custom_field": "custom_value", "doc_id": "doc1"},
        )

        boosted = adjuster.apply_boosts([result], {"doc1": 3})

        assert boosted[0].metadata["custom_field"] == "custom_value"
        assert boosted[0].metadata["doc_id"] == "doc1"
        assert boosted[0].metadata["original_score"] == 0.8

    def test_zero_backlinks_no_boost(self):
        """Test that zero backlinks don't add boost metadata."""
        config = BoostConfig(recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, doc_id="doc1")]
        backlink_counts = {"doc1": 0}

        boosted = adjuster.apply_boosts(results, backlink_counts)

        assert boosted[0].score == pytest.approx(0.8)
        assert "backlink_count" not in boosted[0].metadata
        assert "backlink_boost" not in boosted[0].metadata

    def test_multiple_chunks_same_doc(self):
        """Test handling multiple chunks from the same document."""
        config = BoostConfig(backlink_weight=0.1, recency_enabled=False)
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result("chunk1", 0.9, doc_id="doc1"),
            make_result("chunk2", 0.8, doc_id="doc1"),
            make_result("chunk3", 0.7, doc_id="doc2"),
        ]
        backlink_counts = {"doc1": 5}  # doc1 has 5 backlinks

        boosted = adjuster.apply_boosts(results, backlink_counts)

        # Both chunks from doc1 get the same boost
        assert boosted[0].score == pytest.approx(1.35)  # 0.9 * 1.5
        assert boosted[1].score == pytest.approx(1.2)  # 0.8 * 1.5
        assert boosted[2].score == pytest.approx(0.7)  # doc2, no backlinks
