"""Unit tests for title/heading boost functionality."""

import pytest
from ragtriever.retrieval.boosts import BoostConfig, BoostAdjuster
from ragtriever.models import SearchResult, SourceRef


def make_result(
    chunk_id: str,
    score: float,
    level: int | None = None,
    anchor_type: str = "md_heading",
    vault_id: str = "v1",
    rel_path: str = "test.md",
) -> SearchResult:
    """Helper to create SearchResult with heading metadata for testing."""
    metadata: dict = {}
    if level is not None:
        metadata["level"] = level
    metadata["anchor_type"] = anchor_type

    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        snippet=f"Content for {chunk_id}",
        source_ref=SourceRef(
            vault_id=vault_id,
            rel_path=rel_path,
            file_type="md",
            anchor_type=anchor_type,
            anchor_ref="heading",
        ),
        metadata=metadata,
    )


class TestHeadingBoostConfig:
    """Test heading boost configuration."""

    def test_default_values(self):
        """Test that default heading boost config values are sensible."""
        config = BoostConfig()
        assert config.heading_boost_enabled is True
        assert config.heading_h1_boost == 1.30
        assert config.heading_h2_boost == 1.15
        assert config.heading_h3_boost == 1.08

    def test_custom_values(self):
        """Test custom heading boost values."""
        config = BoostConfig(
            heading_boost_enabled=False,
            heading_h1_boost=1.50,
            heading_h2_boost=1.25,
            heading_h3_boost=1.10,
        )
        assert config.heading_boost_enabled is False
        assert config.heading_h1_boost == 1.50
        assert config.heading_h2_boost == 1.25
        assert config.heading_h3_boost == 1.10


class TestHeadingBoostH1:
    """Test H1 (title) boosting."""

    def test_h1_boost_applied(self):
        """Test that H1 boost is applied correctly (30% = 1.30x)."""
        config = BoostConfig(
            heading_h1_boost=1.30,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, level=1)]
        boosted = adjuster.apply_boosts(results)

        # 0.8 * 1.30 = 1.04
        assert len(boosted) == 1
        assert boosted[0].score == pytest.approx(1.04)
        assert boosted[0].metadata["original_score"] == 0.8
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.30)
        assert boosted[0].metadata["heading_level"] == 1

    def test_h1_boost_multiple_results(self):
        """Test H1 boost with multiple results."""
        config = BoostConfig(
            heading_h1_boost=1.30,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result("chunk1", 0.6, level=1),
            make_result("chunk2", 0.9, level=1),
            make_result("chunk3", 0.7, level=1),
        ]
        boosted = adjuster.apply_boosts(results)

        # All get H1 boost
        assert boosted[0].score == pytest.approx(1.17)  # 0.9 * 1.30
        assert boosted[1].score == pytest.approx(0.91)  # 0.7 * 1.30
        assert boosted[2].score == pytest.approx(0.78)  # 0.6 * 1.30
        # Verify they're re-sorted by boosted score
        assert boosted[0].chunk_id == "chunk2"
        assert boosted[1].chunk_id == "chunk3"
        assert boosted[2].chunk_id == "chunk1"


class TestHeadingBoostH2:
    """Test H2 boosting."""

    def test_h2_boost_applied(self):
        """Test that H2 boost is applied correctly (15% = 1.15x)."""
        config = BoostConfig(
            heading_h2_boost=1.15,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, level=2)]
        boosted = adjuster.apply_boosts(results)

        # 0.8 * 1.15 = 0.92
        assert boosted[0].score == pytest.approx(0.92)
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.15)
        assert boosted[0].metadata["heading_level"] == 2

    def test_h2_boost_custom_value(self):
        """Test H2 boost with custom value."""
        config = BoostConfig(
            heading_h2_boost=1.25,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 1.0, level=2)]
        boosted = adjuster.apply_boosts(results)

        assert boosted[0].score == pytest.approx(1.25)
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.25)


class TestHeadingBoostH3:
    """Test H3 boosting."""

    def test_h3_boost_applied(self):
        """Test that H3 boost is applied correctly (8% = 1.08x)."""
        config = BoostConfig(
            heading_h3_boost=1.08,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, level=3)]
        boosted = adjuster.apply_boosts(results)

        # 0.8 * 1.08 = 0.864
        assert boosted[0].score == pytest.approx(0.864)
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.08)
        assert boosted[0].metadata["heading_level"] == 3

    def test_h3_boost_minimal_effect(self):
        """Test that H3 boost has minimal effect (as designed)."""
        config = BoostConfig(
            heading_h3_boost=1.08,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 1.0, level=3)]
        boosted = adjuster.apply_boosts(results)

        # Only 8% boost
        assert boosted[0].score == pytest.approx(1.08)


class TestHeadingBoostMixedLevels:
    """Test boosting with mixed heading levels."""

    def test_mixed_heading_levels(self):
        """Test that different heading levels get different boosts."""
        config = BoostConfig(
            heading_h1_boost=1.30,
            heading_h2_boost=1.15,
            heading_h3_boost=1.08,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result("chunk_h1", 0.7, level=1),
            make_result("chunk_h2", 0.8, level=2),
            make_result("chunk_h3", 0.9, level=3),
        ]
        boosted = adjuster.apply_boosts(results)

        # H1: 0.7 * 1.30 = 0.91
        # H2: 0.8 * 1.15 = 0.92
        # H3: 0.9 * 1.08 = 0.972
        assert len(boosted) == 3
        # Re-sorted by boosted score
        assert boosted[0].chunk_id == "chunk_h3"
        assert boosted[0].score == pytest.approx(0.972)
        assert boosted[1].chunk_id == "chunk_h2"
        assert boosted[1].score == pytest.approx(0.92)
        assert boosted[2].chunk_id == "chunk_h1"
        assert boosted[2].score == pytest.approx(0.91)

    def test_heading_boost_ranking_change(self):
        """Test that heading boost can change result ordering."""
        config = BoostConfig(
            heading_h1_boost=1.30,
            heading_h2_boost=1.15,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # H1 has lower original score but should rank higher after boost
        results = [
            make_result("chunk_h2", 0.9, level=2),  # Higher original
            make_result("chunk_h1", 0.7, level=1),  # Lower original
        ]
        boosted = adjuster.apply_boosts(results)

        # H2: 0.9 * 1.15 = 1.035
        # H1: 0.7 * 1.30 = 0.91
        # H2 still wins due to much higher original score
        assert boosted[0].chunk_id == "chunk_h2"
        assert boosted[1].chunk_id == "chunk_h1"

    def test_heading_boost_significant_ranking_change(self):
        """Test heading boost with closer original scores."""
        config = BoostConfig(
            heading_h1_boost=1.30,
            heading_h2_boost=1.15,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # Closer original scores - H1 boost should win
        results = [
            make_result("chunk_h2", 0.88, level=2),  # Higher original
            make_result("chunk_h1", 0.85, level=1),  # Lower original
        ]
        boosted = adjuster.apply_boosts(results)

        # H2: 0.88 * 1.15 = 1.012
        # H1: 0.85 * 1.30 = 1.105
        # H1 wins after boost
        assert boosted[0].chunk_id == "chunk_h1"
        assert boosted[0].score == pytest.approx(1.105)
        assert boosted[1].chunk_id == "chunk_h2"
        assert boosted[1].score == pytest.approx(1.012)


class TestHeadingBoostFallback:
    """Test fallback for headings without level metadata."""

    def test_fallback_heading_anchor_type(self):
        """Test fallback boost for chunks with 'heading' in anchor_type but no level."""
        config = BoostConfig(
            heading_h3_boost=1.08,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # No level metadata, but anchor_type contains "heading"
        result = SearchResult(
            chunk_id="chunk1",
            score=0.8,
            snippet="Content",
            source_ref=SourceRef(
                vault_id="v1",
                rel_path="test.md",
                file_type="md",
                anchor_type="md_heading",
                anchor_ref="h",
            ),
            metadata={"anchor_type": "md_heading"},  # No level
        )

        boosted = adjuster.apply_boosts([result])

        # Should get H3 boost as fallback
        assert boosted[0].score == pytest.approx(0.864)  # 0.8 * 1.08
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.08)
        assert boosted[0].metadata["heading_level"] is None

    def test_fallback_case_insensitive(self):
        """Test that fallback is case-insensitive for 'heading'."""
        config = BoostConfig(
            heading_h3_boost=1.08,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        result = SearchResult(
            chunk_id="chunk1",
            score=0.9,
            snippet="Content",
            source_ref=SourceRef(
                vault_id="v1",
                rel_path="test.md",
                file_type="md",
                anchor_type="MD_HEADING",
                anchor_ref="h",
            ),
            metadata={"anchor_type": "MD_HEADING"},
        )

        boosted = adjuster.apply_boosts([result])

        # Should get fallback boost
        assert boosted[0].score == pytest.approx(0.972)  # 0.9 * 1.08
        assert boosted[0].metadata["heading_level"] is None

    def test_no_fallback_for_non_heading(self):
        """Test no boost for non-heading chunks without level."""
        config = BoostConfig(
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        result = SearchResult(
            chunk_id="chunk1",
            score=0.8,
            snippet="Content",
            source_ref=SourceRef(
                vault_id="v1",
                rel_path="test.md",
                file_type="md",
                anchor_type="md_block",
                anchor_ref="para",
            ),
            metadata={"anchor_type": "md_block"},  # Not a heading
        )

        boosted = adjuster.apply_boosts([result])

        # No boost applied
        assert boosted[0].score == pytest.approx(0.8)
        assert "heading_boost" not in boosted[0].metadata
        assert "heading_level" not in boosted[0].metadata


class TestHeadingBoostDisabled:
    """Test disabled heading boost."""

    def test_heading_boost_disabled_h1(self):
        """Test that H1 boost can be disabled."""
        config = BoostConfig(
            heading_boost_enabled=False,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, level=1)]
        boosted = adjuster.apply_boosts(results)

        # No boost applied
        assert boosted[0].score == pytest.approx(0.8)
        assert "heading_boost" not in boosted[0].metadata
        assert "heading_level" not in boosted[0].metadata

    def test_heading_boost_disabled_all_levels(self):
        """Test that all heading levels are unaffected when disabled."""
        config = BoostConfig(
            heading_boost_enabled=False,
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result("chunk1", 0.7, level=1),
            make_result("chunk2", 0.8, level=2),
            make_result("chunk3", 0.9, level=3),
        ]
        boosted = adjuster.apply_boosts(results)

        # No boosts applied, sorted by original score
        assert boosted[0].chunk_id == "chunk3"
        assert boosted[0].score == pytest.approx(0.9)
        assert boosted[1].chunk_id == "chunk2"
        assert boosted[1].score == pytest.approx(0.8)
        assert boosted[2].chunk_id == "chunk1"
        assert boosted[2].score == pytest.approx(0.7)

        for result in boosted:
            assert "heading_boost" not in result.metadata
            assert "heading_level" not in result.metadata


class TestHeadingBoostEdgeCases:
    """Test edge cases for heading boost."""

    def test_invalid_level_type(self):
        """Test handling of invalid level type (non-integer)."""
        config = BoostConfig(
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
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
                anchor_ref="h",
            ),
            metadata={"level": "not-a-number"},
        )

        boosted = adjuster.apply_boosts([result])

        # Should fall back to anchor_type check
        assert boosted[0].score == pytest.approx(0.864)  # 0.8 * 1.08 (H3 fallback)
        assert boosted[0].metadata["heading_level"] is None

    def test_level_out_of_range(self):
        """Test handling of heading levels beyond H3."""
        config = BoostConfig(
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result("chunk_h4", 0.8, level=4),
            make_result("chunk_h5", 0.8, level=5),
            make_result("chunk_h6", 0.8, level=6),
        ]
        boosted = adjuster.apply_boosts(results)

        # Levels beyond H3 don't get boost (not in config)
        for result in boosted:
            assert result.score == pytest.approx(0.8)
            assert "heading_boost" not in result.metadata

    def test_level_zero(self):
        """Test handling of level=0 (invalid)."""
        config = BoostConfig(
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, level=0)]
        boosted = adjuster.apply_boosts(results)

        # Level 0 doesn't match any boost config
        assert boosted[0].score == pytest.approx(0.8)
        assert "heading_boost" not in boosted[0].metadata

    def test_preserves_existing_metadata(self):
        """Test that heading boost preserves other metadata."""
        config = BoostConfig(
            backlink_enabled=False,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
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
                anchor_ref="h",
            ),
            metadata={"level": 1, "custom_field": "custom_value", "tags": ["ai", "ml"]},
        )

        boosted = adjuster.apply_boosts([result])

        assert boosted[0].metadata["custom_field"] == "custom_value"
        assert boosted[0].metadata["tags"] == ["ai", "ml"]
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.30)
        assert boosted[0].metadata["level"] == 1


class TestHeadingBoostCombined:
    """Test heading boost combined with other boosts."""

    def test_heading_with_backlink_boost(self):
        """Test heading boost combined with backlink boost."""
        config = BoostConfig(
            heading_h1_boost=1.30,
            backlink_weight=0.1,
            recency_enabled=False,
            tag_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.5, level=1, vault_id="v1", rel_path="doc.md")]
        # Add doc_id to metadata for backlink matching
        results[0] = SearchResult(
            chunk_id="chunk1",
            score=0.5,
            snippet="Content",
            source_ref=results[0].source_ref,
            metadata={"level": 1, "doc_id": "doc1"},
        )
        backlink_counts = {"doc1": 5}

        boosted = adjuster.apply_boosts(results, backlink_counts)

        # Backlink boost: 1 + (0.1 * 5) = 1.5
        # Heading boost: 1.30
        # Combined: 0.5 * 1.5 * 1.30 = 0.975
        assert boosted[0].score == pytest.approx(0.975)
        assert boosted[0].metadata["backlink_boost"] == pytest.approx(1.5)
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.30)

    def test_heading_with_all_boosts(self):
        """Test heading boost combined with all other boosts."""
        from datetime import datetime, timezone, timedelta

        config = BoostConfig(
            heading_h2_boost=1.15,
            backlink_weight=0.1,
            recency_fresh_boost=1.20,
            tag_boost_enabled=False,  # Exclude tag boost for this test
        )
        adjuster = BoostAdjuster(config=config)

        fresh_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        result = SearchResult(
            chunk_id="chunk1",
            score=0.5,
            snippet="Content",
            source_ref=SourceRef(
                vault_id="v1",
                rel_path="doc.md",
                file_type="md",
                anchor_type="md_heading",
                anchor_ref="h",
            ),
            metadata={"level": 2, "doc_id": "doc1", "modified_at": fresh_date},
        )
        backlink_counts = {"doc1": 3}

        boosted = adjuster.apply_boosts([result], backlink_counts)

        # Backlink boost: 1 + (0.1 * 3) = 1.3
        # Recency boost: 1.20
        # Heading boost: 1.15
        # Combined: 0.5 * 1.3 * 1.20 * 1.15 = 0.897
        assert boosted[0].score == pytest.approx(0.897)
        assert boosted[0].metadata["backlink_boost"] == pytest.approx(1.3)
        assert boosted[0].metadata["recency_boost"] == pytest.approx(1.20)
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.15)
