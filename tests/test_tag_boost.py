"""Unit tests for tag match boost functionality."""

import pytest
from mneme.retrieval.boosts import BoostConfig, BoostAdjuster
from mneme.models import SearchResult, SourceRef


def make_result(
    chunk_id: str,
    score: float,
    tags: list[str] | str | None = None,
    vault_id: str = "v1",
    rel_path: str = "test.md",
) -> SearchResult:
    """Helper to create SearchResult with tags for testing."""
    metadata: dict = {}
    if tags is not None:
        metadata["tags"] = tags

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


class TestTagBoostConfig:
    """Test tag boost configuration."""

    def test_default_values(self):
        """Test that default tag boost config values are sensible."""
        config = BoostConfig()
        assert config.tag_boost_enabled is True
        assert config.tag_boost_weight == 0.15
        assert config.tag_boost_cap == 3

    def test_custom_values(self):
        """Test custom tag boost values."""
        config = BoostConfig(
            tag_boost_enabled=False,
            tag_boost_weight=0.20,
            tag_boost_cap=5,
        )
        assert config.tag_boost_enabled is False
        assert config.tag_boost_weight == 0.20
        assert config.tag_boost_cap == 5


class TestSingleTagMatch:
    """Test single tag matching."""

    def test_single_tag_exact_match(self):
        """Test boost for single exact tag match."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python"])]
        boosted = adjuster.apply_boosts(results, query="python tutorial")

        # 1 match * 0.15 = 0.15 -> boost = 1.15
        # 0.8 * 1.15 = 0.92
        assert len(boosted) == 1
        assert boosted[0].score == pytest.approx(0.92)
        assert boosted[0].metadata["original_score"] == 0.8
        assert boosted[0].metadata["tag_boost"] == pytest.approx(1.15)
        assert boosted[0].metadata["tag_matches"] == 1

    def test_single_tag_case_insensitive(self):
        """Test tag matching is case-insensitive."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # Tag in uppercase, query in lowercase
        results = [make_result("chunk1", 0.8, tags=["PYTHON"])]
        boosted = adjuster.apply_boosts(results, query="python tutorial")

        assert boosted[0].score == pytest.approx(0.92)  # 0.8 * 1.15
        assert boosted[0].metadata["tag_matches"] == 1

    def test_single_tag_with_hash_prefix(self):
        """Test tag matching with # prefix."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # Tag with # prefix (Obsidian style)
        results = [make_result("chunk1", 0.8, tags=["#python"])]
        boosted = adjuster.apply_boosts(results, query="python")

        # Should still match (# is stripped)
        assert boosted[0].score == pytest.approx(0.92)  # 0.8 * 1.15
        assert boosted[0].metadata["tag_matches"] == 1

    def test_single_tag_no_match(self):
        """Test no boost when tag doesn't match query."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python"])]
        boosted = adjuster.apply_boosts(results, query="javascript tutorial")

        # No match, no boost
        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_boost" not in boosted[0].metadata
        assert "tag_matches" not in boosted[0].metadata

    def test_single_tag_string_format(self):
        """Test tag as string instead of list."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # Tag as string (should be normalized to list)
        results = [make_result("chunk1", 0.8, tags="python")]
        boosted = adjuster.apply_boosts(results, query="python")

        assert boosted[0].score == pytest.approx(0.92)
        assert boosted[0].metadata["tag_matches"] == 1


class TestMultipleTagMatches:
    """Test multiple tag matching and capping."""

    def test_two_tag_matches(self):
        """Test boost for two matching tags."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python", "machine-learning", "data"])]
        boosted = adjuster.apply_boosts(results, query="python machine learning")

        # 2 matches * 0.15 = 0.30 -> boost = 1.30
        # 0.8 * 1.30 = 1.04
        assert boosted[0].score == pytest.approx(1.04)
        assert boosted[0].metadata["tag_boost"] == pytest.approx(1.30)
        assert boosted[0].metadata["tag_matches"] == 2

    def test_three_tag_matches(self):
        """Test boost for three matching tags (at cap)."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            tag_boost_cap=3,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result("chunk1", 0.8, tags=["python", "ai", "ml", "data", "nlp"])
        ]
        boosted = adjuster.apply_boosts(results, query="python ai ml data")

        # 4 matches, but capped at 3: 3 * 0.15 = 0.45 -> boost = 1.45
        # 0.8 * 1.45 = 1.16
        assert boosted[0].score == pytest.approx(1.16)
        assert boosted[0].metadata["tag_boost"] == pytest.approx(1.45)
        assert boosted[0].metadata["tag_matches"] == 4  # Actual matches preserved

    def test_tag_match_capping_at_max(self):
        """Test that tag matches are capped at max (3 tags = 45%)."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            tag_boost_cap=3,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # 10 matching tags, but should cap at 3
        results = [
            make_result(
                "chunk1",
                1.0,
                tags=["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"],
            )
        ]
        boosted = adjuster.apply_boosts(
            results, query="t1 t2 t3 t4 t5 t6 t7 t8 t9 t10"
        )

        # Capped at 3 matches: 3 * 0.15 = 0.45 -> boost = 1.45
        assert boosted[0].score == pytest.approx(1.45)
        assert boosted[0].metadata["tag_boost"] == pytest.approx(1.45)
        assert boosted[0].metadata["tag_matches"] == 10  # Actual matches preserved

    def test_custom_cap_value(self):
        """Test custom tag cap value."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            tag_boost_cap=5,  # Custom cap
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result("chunk1", 1.0, tags=["t1", "t2", "t3", "t4", "t5", "t6"])
        ]
        boosted = adjuster.apply_boosts(results, query="t1 t2 t3 t4 t5 t6")

        # Capped at 5 matches: 5 * 0.15 = 0.75 -> boost = 1.75
        assert boosted[0].score == pytest.approx(1.75)
        assert boosted[0].metadata["tag_matches"] == 6


class TestPartialWordMatches:
    """Test partial word matches in tags."""

    def test_partial_word_match_hyphenated(self):
        """Test query 'machine' matches tag '#machine-learning'."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["machine-learning"])]
        boosted = adjuster.apply_boosts(results, query="machine")

        # Hyphen converted to space, "machine" matches
        assert boosted[0].score == pytest.approx(0.92)  # 0.8 * 1.15
        assert boosted[0].metadata["tag_matches"] == 1

    def test_partial_word_match_underscored(self):
        """Test query 'deep' matches tag 'deep_learning'."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["deep_learning"])]
        boosted = adjuster.apply_boosts(results, query="deep networks")

        # Underscore converted to space, "deep" matches
        assert boosted[0].score == pytest.approx(0.92)
        assert boosted[0].metadata["tag_matches"] == 1

    def test_partial_word_no_match(self):
        """Test partial word that doesn't match."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["machine-learning"])]
        boosted = adjuster.apply_boosts(results, query="python")

        # No overlap
        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_matches" not in boosted[0].metadata

    def test_multiple_partial_matches(self):
        """Test multiple partial matches from same tag."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # Query has both words from hyphenated tag
        results = [make_result("chunk1", 0.8, tags=["machine-learning"])]
        boosted = adjuster.apply_boosts(results, query="machine learning algorithms")

        # Only counts as 1 tag match (not 2, even though both words match)
        assert boosted[0].score == pytest.approx(0.92)  # 0.8 * 1.15
        assert boosted[0].metadata["tag_matches"] == 1

    def test_complex_multi_word_tags(self):
        """Test complex multi-word tags with various separators."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result(
                "chunk1",
                0.8,
                tags=[
                    "natural-language-processing",
                    "computer_vision",
                    "reinforcement-learning",
                ],
            )
        ]
        boosted = adjuster.apply_boosts(results, query="natural language computer")

        # "natural" matches first tag, "computer" matches second tag = 2 matches
        assert boosted[0].score == pytest.approx(1.04)  # 0.8 * 1.30
        assert boosted[0].metadata["tag_matches"] == 2


class TestTagBoostDisabled:
    """Test disabled tag boost."""

    def test_tag_boost_disabled(self):
        """Test that tag boost can be disabled."""
        config = BoostConfig(
            tag_boost_enabled=False,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python"])]
        boosted = adjuster.apply_boosts(results, query="python")

        # No boost applied
        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_boost" not in boosted[0].metadata
        assert "tag_matches" not in boosted[0].metadata

    def test_tag_boost_disabled_with_multiple_tags(self):
        """Test disabled boost with multiple matching tags."""
        config = BoostConfig(
            tag_boost_enabled=False,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python", "ml", "ai"])]
        boosted = adjuster.apply_boosts(results, query="python ml ai")

        # No boost even with 3 matches
        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_boost" not in boosted[0].metadata


class TestTagBoostEdgeCases:
    """Test edge cases for tag boost."""

    def test_no_tags_metadata(self):
        """Test handling when tags metadata is missing."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8)]  # No tags
        boosted = adjuster.apply_boosts(results, query="python")

        # No boost
        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_boost" not in boosted[0].metadata

    def test_empty_tags_list(self):
        """Test handling of empty tags list."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=[])]
        boosted = adjuster.apply_boosts(results, query="python")

        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_boost" not in boosted[0].metadata

    def test_no_query_provided(self):
        """Test handling when query is None."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python"])]
        boosted = adjuster.apply_boosts(results, query=None)

        # No boost without query
        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_boost" not in boosted[0].metadata

    def test_empty_query_string(self):
        """Test handling of empty query string."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python"])]
        boosted = adjuster.apply_boosts(results, query="")

        # No boost with empty query
        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_boost" not in boosted[0].metadata

    def test_whitespace_only_query(self):
        """Test handling of whitespace-only query."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python"])]
        boosted = adjuster.apply_boosts(results, query="   ")

        assert boosted[0].score == pytest.approx(0.8)
        assert "tag_boost" not in boosted[0].metadata

    def test_special_characters_in_tags(self):
        """Test tags with special characters."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["c++", "c#", ".net"])]
        boosted = adjuster.apply_boosts(results, query="c++ programming")

        # Should match (basic word tokenization)
        assert boosted[0].score == pytest.approx(0.92)
        assert boosted[0].metadata["tag_matches"] == 1

    def test_numeric_tags(self):
        """Test tags with numbers."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [make_result("chunk1", 0.8, tags=["python3", "web2.0"])]
        boosted = adjuster.apply_boosts(results, query="python3")

        assert boosted[0].score == pytest.approx(0.92)
        assert boosted[0].metadata["tag_matches"] == 1

    def test_preserves_existing_metadata(self):
        """Test that tag boost preserves other metadata."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
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
            metadata={
                "tags": ["python"],
                "custom_field": "custom_value",
                "level": 1,
            },
        )

        boosted = adjuster.apply_boosts([result], query="python")

        assert boosted[0].metadata["custom_field"] == "custom_value"
        assert boosted[0].metadata["level"] == 1
        assert boosted[0].metadata["tag_boost"] == pytest.approx(1.15)


class TestTagBoostRanking:
    """Test tag boost effect on result ranking."""

    def test_tag_boost_changes_ranking(self):
        """Test that tag boost can change result ordering."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        # chunk2 has lower score but matching tags
        results = [
            make_result("chunk1", 0.9, tags=["javascript"]),
            make_result("chunk2", 0.7, tags=["python", "ml", "ai"]),
        ]
        boosted = adjuster.apply_boosts(results, query="python machine learning ai")

        # chunk1: 0.9 (no boost)
        # chunk2: 0.7 * 1.45 = 1.015 (3 matches capped at 3)
        # chunk2 should now rank first
        assert boosted[0].chunk_id == "chunk2"
        assert boosted[0].score == pytest.approx(1.015)
        assert boosted[1].chunk_id == "chunk1"
        assert boosted[1].score == pytest.approx(0.9)

    def test_multiple_results_resorting(self):
        """Test re-sorting of multiple results after tag boost."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_enabled=False,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

        results = [
            make_result("chunk1", 0.9, tags=["unrelated"]),
            make_result("chunk2", 0.8, tags=["python"]),
            make_result("chunk3", 0.7, tags=["python", "ml"]),
        ]
        boosted = adjuster.apply_boosts(results, query="python ml")

        # chunk1: 0.9 (no boost)
        # chunk2: 0.8 * 1.15 = 0.92 (1 match)
        # chunk3: 0.7 * 1.30 = 0.91 (2 matches)
        # Order should be: chunk2, chunk3, chunk1
        assert boosted[0].chunk_id == "chunk2"
        assert boosted[0].score == pytest.approx(0.92)
        assert boosted[1].chunk_id == "chunk3"
        assert boosted[1].score == pytest.approx(0.91)
        assert boosted[2].chunk_id == "chunk1"
        assert boosted[2].score == pytest.approx(0.9)


class TestTagBoostCombined:
    """Test tag boost combined with other boosts."""

    def test_tag_with_backlink_boost(self):
        """Test tag boost combined with backlink boost."""
        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_weight=0.1,
            recency_enabled=False,
            heading_boost_enabled=False,
        )
        adjuster = BoostAdjuster(config=config)

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
            metadata={"tags": ["python", "ml"], "doc_id": "doc1"},
        )
        backlink_counts = {"doc1": 3}

        boosted = adjuster.apply_boosts([result], backlink_counts, query="python ml")

        # Tag boost: 1 + (0.15 * 2) = 1.30
        # Backlink boost: 1 + (0.1 * 3) = 1.3
        # Combined: 0.5 * 1.30 * 1.3 = 0.845
        assert boosted[0].score == pytest.approx(0.845)
        assert boosted[0].metadata["tag_boost"] == pytest.approx(1.30)
        assert boosted[0].metadata["backlink_boost"] == pytest.approx(1.3)

    def test_tag_with_all_boosts(self):
        """Test tag boost combined with all other boosts."""
        from datetime import datetime, timezone, timedelta

        config = BoostConfig(
            tag_boost_weight=0.15,
            backlink_weight=0.1,
            recency_fresh_boost=1.20,
            heading_h1_boost=1.30,
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
            metadata={
                "tags": ["python", "ai"],
                "doc_id": "doc1",
                "modified_at": fresh_date,
                "level": 1,
            },
        )
        backlink_counts = {"doc1": 2}

        boosted = adjuster.apply_boosts([result], backlink_counts, query="python ai")

        # Tag boost: 1 + (0.15 * 2) = 1.30
        # Backlink boost: 1 + (0.1 * 2) = 1.2
        # Recency boost: 1.20
        # Heading boost: 1.30
        # Combined: 0.5 * 1.30 * 1.2 * 1.20 * 1.30 = 1.2168
        assert boosted[0].score == pytest.approx(1.2168)
        assert boosted[0].metadata["tag_boost"] == pytest.approx(1.30)
        assert boosted[0].metadata["backlink_boost"] == pytest.approx(1.2)
        assert boosted[0].metadata["recency_boost"] == pytest.approx(1.20)
        assert boosted[0].metadata["heading_boost"] == pytest.approx(1.30)
