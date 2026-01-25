"""Unit tests for MMR diversity functionality."""

import pytest
from ragtriever.retrieval.diversity import DiversityConfig, MMRDiversifier
from ragtriever.models import SearchResult, SourceRef


def make_result(
    chunk_id: str,
    score: float,
    vault_id: str = "v1",
    rel_path: str = "doc.md",
) -> SearchResult:
    """Helper to create SearchResult for testing."""
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
        metadata={},
    )


class TestDiversityConfig:
    """Test diversity configuration."""

    def test_default_values(self):
        """Test that default diversity config values are sensible."""
        config = DiversityConfig()
        assert config.enabled is True
        assert config.lambda_param == 0.7
        assert config.max_per_document == 2

    def test_custom_values(self):
        """Test custom diversity values."""
        config = DiversityConfig(
            enabled=False,
            lambda_param=0.5,
            max_per_document=3,
        )
        assert config.enabled is False
        assert config.lambda_param == 0.5
        assert config.max_per_document == 3


class TestMaxChunksPerDocument:
    """Test max chunks per document constraint."""

    def test_max_2_chunks_per_document(self):
        """Test that max 2 chunks per document are returned."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        # 4 chunks from doc1, 2 chunks from doc2
        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
            make_result("chunk3", 0.8, rel_path="doc1.md"),
            make_result("chunk4", 0.7, rel_path="doc1.md"),
            make_result("chunk5", 0.6, rel_path="doc2.md"),
            make_result("chunk6", 0.5, rel_path="doc2.md"),
        ]

        diversified = diversifier.diversify(results, k=6)

        # Should get: chunk1, chunk2 (from doc1), chunk3, chunk4 (from doc1 - wait no!)
        # Actually: chunk1, chunk2 (from doc1 - max 2), chunk5, chunk6 (from doc2 - max 2)
        # Then backfill: chunk3, chunk4
        assert len(diversified) == 6

        # Count chunks per document
        doc_counts = {}
        for i, r in enumerate(diversified[:4]):  # First 4 should respect constraint
            doc_id = f"{r.source_ref.vault_id}:{r.source_ref.rel_path}"
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        # In first pass (before backfill), each doc should have max 2
        assert diversified[0].chunk_id == "chunk1"
        assert diversified[1].chunk_id == "chunk2"
        assert diversified[2].chunk_id == "chunk5"
        assert diversified[3].chunk_id == "chunk6"
        # Backfill adds more from doc1
        assert diversified[4].chunk_id == "chunk3"
        assert diversified[5].chunk_id == "chunk4"

    def test_max_2_chunks_enforced_strictly(self):
        """Test max 2 chunks strictly enforced during first pass."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        # 5 chunks from same document
        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
            make_result("chunk3", 0.8, rel_path="doc1.md"),
            make_result("chunk4", 0.7, rel_path="doc1.md"),
            make_result("chunk5", 0.6, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=2)

        # With k=2, should only get first 2 chunks (no backfill needed)
        assert len(diversified) == 2
        assert diversified[0].chunk_id == "chunk1"
        assert diversified[1].chunk_id == "chunk2"

    def test_max_3_chunks_custom_value(self):
        """Test custom max_per_document value."""
        config = DiversityConfig(max_per_document=3)
        diversifier = MMRDiversifier(config=config)

        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
            make_result("chunk3", 0.8, rel_path="doc1.md"),
            make_result("chunk4", 0.7, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=3)

        # Should get first 3 chunks (respecting max_per_document=3)
        assert len(diversified) == 3
        assert diversified[0].chunk_id == "chunk1"
        assert diversified[1].chunk_id == "chunk2"
        assert diversified[2].chunk_id == "chunk3"

    def test_multiple_documents_interleaved(self):
        """Test that chunks from multiple documents are properly diversified."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        # Interleaved chunks from 3 documents
        results = [
            make_result("c1", 1.0, rel_path="doc1.md"),
            make_result("c2", 0.95, rel_path="doc2.md"),
            make_result("c3", 0.9, rel_path="doc3.md"),
            make_result("c4", 0.85, rel_path="doc1.md"),
            make_result("c5", 0.8, rel_path="doc2.md"),
            make_result("c6", 0.75, rel_path="doc3.md"),
            make_result("c7", 0.7, rel_path="doc1.md"),  # 3rd from doc1
            make_result("c8", 0.65, rel_path="doc2.md"),  # 3rd from doc2
        ]

        diversified = diversifier.diversify(results, k=6)

        # Should get first 2 from each doc (total 6)
        assert len(diversified) == 6
        assert diversified[0].chunk_id == "c1"  # doc1 #1
        assert diversified[1].chunk_id == "c2"  # doc2 #1
        assert diversified[2].chunk_id == "c3"  # doc3 #1
        assert diversified[3].chunk_id == "c4"  # doc1 #2 (max reached)
        assert diversified[4].chunk_id == "c5"  # doc2 #2 (max reached)
        assert diversified[5].chunk_id == "c6"  # doc3 #2 (max reached)


class TestBackfillWhenNotEnoughDiverse:
    """Test backfill mechanism when diversity constraint prevents reaching k."""

    def test_backfill_from_same_document(self):
        """Test backfilling from skipped results when not enough diverse results."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        # Only 1 document, 5 chunks, k=4
        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
            make_result("chunk3", 0.8, rel_path="doc1.md"),
            make_result("chunk4", 0.7, rel_path="doc1.md"),
            make_result("chunk5", 0.6, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=4)

        # First pass: 2 chunks (max_per_document)
        # Backfill: 2 more chunks to reach k=4
        assert len(diversified) == 4
        assert diversified[0].chunk_id == "chunk1"
        assert diversified[1].chunk_id == "chunk2"
        assert diversified[2].chunk_id == "chunk3"  # Backfilled
        assert diversified[3].chunk_id == "chunk4"  # Backfilled

    def test_backfill_partial(self):
        """Test partial backfill when some diversity constraint is met."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        # 2 docs: doc1 has 4 chunks, doc2 has 1 chunk
        results = [
            make_result("c1", 1.0, rel_path="doc1.md"),
            make_result("c2", 0.9, rel_path="doc1.md"),
            make_result("c3", 0.8, rel_path="doc2.md"),
            make_result("c4", 0.7, rel_path="doc1.md"),
            make_result("c5", 0.6, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=5)

        # First pass: c1, c2 (doc1 max), c3 (doc2)
        # Backfill: c4, c5 to reach k=5
        assert len(diversified) == 5
        assert diversified[0].chunk_id == "c1"
        assert diversified[1].chunk_id == "c2"
        assert diversified[2].chunk_id == "c3"
        assert diversified[3].chunk_id == "c4"  # Backfilled
        assert diversified[4].chunk_id == "c5"  # Backfilled

    def test_backfill_maintains_score_order(self):
        """Test that backfill maintains relevance score ordering."""
        config = DiversityConfig(max_per_document=1)
        diversifier = MMRDiversifier(config=config)

        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
            make_result("chunk3", 0.8, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=3)

        # First pass: chunk1 (max 1 per doc)
        # Backfill: chunk2, chunk3 in score order
        assert len(diversified) == 3
        assert diversified[0].chunk_id == "chunk1"
        assert diversified[1].chunk_id == "chunk2"
        assert diversified[2].chunk_id == "chunk3"

    def test_no_backfill_needed(self):
        """Test case where no backfill is needed."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        # 3 docs, 1 chunk each, k=3
        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc2.md"),
            make_result("chunk3", 0.8, rel_path="doc3.md"),
        ]

        diversified = diversifier.diversify(results, k=3)

        # No backfill needed, diversity constraint satisfied
        assert len(diversified) == 3
        assert diversified[0].chunk_id == "chunk1"
        assert diversified[1].chunk_id == "chunk2"
        assert diversified[2].chunk_id == "chunk3"


class TestDiversityDisabled:
    """Test disabled diversity."""

    def test_diversity_disabled(self):
        """Test that diversity can be disabled."""
        config = DiversityConfig(enabled=False)
        diversifier = MMRDiversifier(config=config)

        # 5 chunks from same document
        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
            make_result("chunk3", 0.8, rel_path="doc1.md"),
            make_result("chunk4", 0.7, rel_path="doc1.md"),
            make_result("chunk5", 0.6, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=3)

        # With diversity disabled, should just return first k results
        assert len(diversified) == 3
        assert diversified[0].chunk_id == "chunk1"
        assert diversified[1].chunk_id == "chunk2"
        assert diversified[2].chunk_id == "chunk3"

    def test_diversity_disabled_no_constraint(self):
        """Test that disabled diversity allows all chunks from same doc."""
        config = DiversityConfig(enabled=False, max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
            make_result("chunk3", 0.8, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=3)

        # All 3 chunks from same doc (no constraint)
        assert len(diversified) == 3
        # Count chunks from doc1
        doc1_count = sum(
            1 for r in diversified
            if r.source_ref.rel_path == "doc1.md"
        )
        assert doc1_count == 3


class TestEdgeCases:
    """Test edge cases for diversity."""

    def test_empty_results(self):
        """Test handling of empty results list."""
        config = DiversityConfig()
        diversifier = MMRDiversifier(config=config)

        diversified = diversifier.diversify([], k=10)

        assert diversified == []

    def test_k_greater_than_results(self):
        """Test k > len(results)."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=10)

        # Should return all available results (2)
        assert len(diversified) == 2
        assert diversified[0].chunk_id == "chunk1"
        assert diversified[1].chunk_id == "chunk2"

    def test_k_zero(self):
        """Test k=0."""
        config = DiversityConfig()
        diversifier = MMRDiversifier(config=config)

        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
        ]

        diversified = diversifier.diversify(results, k=0)

        assert len(diversified) == 0

    def test_k_one(self):
        """Test k=1."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        results = [
            make_result("chunk1", 1.0, rel_path="doc1.md"),
            make_result("chunk2", 0.9, rel_path="doc1.md"),
            make_result("chunk3", 0.8, rel_path="doc2.md"),
        ]

        diversified = diversifier.diversify(results, k=1)

        # Should return only the top result
        assert len(diversified) == 1
        assert diversified[0].chunk_id == "chunk1"

    def test_single_result(self):
        """Test with single result."""
        config = DiversityConfig()
        diversifier = MMRDiversifier(config=config)

        results = [make_result("chunk1", 1.0, rel_path="doc1.md")]

        diversified = diversifier.diversify(results, k=5)

        assert len(diversified) == 1
        assert diversified[0].chunk_id == "chunk1"

    def test_different_vault_ids_same_relpath(self):
        """Test that different vault IDs are treated as different documents."""
        config = DiversityConfig(max_per_document=1)
        diversifier = MMRDiversifier(config=config)

        # Same rel_path but different vault_ids
        results = [
            make_result("chunk1", 1.0, vault_id="v1", rel_path="doc.md"),
            make_result("chunk2", 0.9, vault_id="v2", rel_path="doc.md"),
            make_result("chunk3", 0.8, vault_id="v1", rel_path="doc.md"),
        ]

        diversified = diversifier.diversify(results, k=2)

        # Should get one from v1:doc.md and one from v2:doc.md
        assert len(diversified) == 2
        assert diversified[0].chunk_id == "chunk1"  # v1:doc.md
        assert diversified[1].chunk_id == "chunk2"  # v2:doc.md

    def test_preserves_metadata(self):
        """Test that diversity preserves result metadata."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        result = SearchResult(
            chunk_id="chunk1",
            score=1.0,
            snippet="Content",
            source_ref=SourceRef(
                vault_id="v1",
                rel_path="doc.md",
                file_type="md",
                anchor_type="md_heading",
                anchor_ref="h",
            ),
            metadata={"custom_field": "value", "tags": ["ai"]},
        )

        diversified = diversifier.diversify([result], k=1)

        assert diversified[0].metadata["custom_field"] == "value"
        assert diversified[0].metadata["tags"] == ["ai"]

    def test_preserves_all_searchresult_fields(self):
        """Test that all SearchResult fields are preserved."""
        config = DiversityConfig()
        diversifier = MMRDiversifier(config=config)

        result = make_result("chunk1", 0.95, vault_id="vault1", rel_path="notes.md")

        diversified = diversifier.diversify([result], k=1)

        assert diversified[0].chunk_id == "chunk1"
        assert diversified[0].score == 0.95
        assert diversified[0].snippet == "Content for chunk1"
        assert diversified[0].source_ref.vault_id == "vault1"
        assert diversified[0].source_ref.rel_path == "notes.md"


class TestRealWorldScenarios:
    """Test realistic diversity scenarios."""

    def test_search_results_with_clusters(self):
        """Test realistic search results with document clusters."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        # Simulating search where top results cluster around 2 documents
        results = [
            make_result("c1", 1.0, rel_path="popular_doc.md"),
            make_result("c2", 0.98, rel_path="popular_doc.md"),
            make_result("c3", 0.96, rel_path="popular_doc.md"),
            make_result("c4", 0.94, rel_path="popular_doc.md"),
            make_result("c5", 0.92, rel_path="another_doc.md"),
            make_result("c6", 0.90, rel_path="another_doc.md"),
            make_result("c7", 0.88, rel_path="third_doc.md"),
            make_result("c8", 0.86, rel_path="third_doc.md"),
        ]

        diversified = diversifier.diversify(results, k=6)

        # Should get diverse results, not just from popular_doc
        assert len(diversified) == 6
        assert diversified[0].chunk_id == "c1"  # popular_doc #1
        assert diversified[1].chunk_id == "c2"  # popular_doc #2 (max)
        assert diversified[2].chunk_id == "c5"  # another_doc #1
        assert diversified[3].chunk_id == "c6"  # another_doc #2 (max)
        assert diversified[4].chunk_id == "c7"  # third_doc #1
        assert diversified[5].chunk_id == "c8"  # third_doc #2 (max)

    def test_small_k_large_results(self):
        """Test small k with large result set."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        results = [
            make_result(f"chunk{i}", 1.0 - i * 0.01, rel_path=f"doc{i // 5}.md")
            for i in range(50)
        ]

        diversified = diversifier.diversify(results, k=5)

        # Should get 5 most relevant, respecting diversity
        assert len(diversified) == 5
        # Check first result is highest scored
        assert diversified[0].chunk_id == "chunk0"
        assert diversified[0].score == 1.0

    def test_many_documents_one_chunk_each(self):
        """Test many documents with one chunk each."""
        config = DiversityConfig(max_per_document=2)
        diversifier = MMRDiversifier(config=config)

        results = [
            make_result(f"chunk{i}", 1.0 - i * 0.01, rel_path=f"doc{i}.md")
            for i in range(20)
        ]

        diversified = diversifier.diversify(results, k=10)

        # Should get top 10, no constraint hit
        assert len(diversified) == 10
        for i in range(10):
            assert diversified[i].chunk_id == f"chunk{i}"
