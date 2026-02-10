"""Tests for link persistence (Feature 1: upsert_links)."""
from __future__ import annotations

from pathlib import Path

import pytest

from mneme.store.libsql_store import LibSqlStore


@pytest.fixture
def store(tmp_path: Path) -> LibSqlStore:
    """Create a fresh LibSqlStore for each test."""
    s = LibSqlStore(tmp_path / "test.sqlite")
    s.init()
    return s


VAULT = "testvault"


class TestUpsertLinksBasic:
    """Core upsert_links behavior."""

    def test_upsert_links_basic(self, store: LibSqlStore):
        """Insert links, verify via neighbors()."""
        links = [("note-b", "wikilink"), ("note-c", "wikilink")]
        store.upsert_links(VAULT, "note-a.md", links)

        result = store.neighbors(VAULT, "note-a.md")
        assert set(result["outlinks"]) == {"note-b", "note-c"}

    def test_upsert_links_replaces_old(self, store: LibSqlStore):
        """Re-upserting replaces previous links."""
        store.upsert_links(VAULT, "note-a.md", [("old-target", "wikilink")])
        store.upsert_links(VAULT, "note-a.md", [("new-target", "wikilink")])

        result = store.neighbors(VAULT, "note-a.md")
        assert result["outlinks"] == ["new-target"]

    def test_upsert_links_empty_list(self, store: LibSqlStore):
        """Empty list clears all outgoing links."""
        store.upsert_links(VAULT, "note-a.md", [("note-b", "wikilink")])
        store.upsert_links(VAULT, "note-a.md", [])

        result = store.neighbors(VAULT, "note-a.md")
        assert result["outlinks"] == []

    def test_upsert_links_executemany(self, store: LibSqlStore):
        """Multiple links in single call (executemany path)."""
        links = [(f"target-{i}", "wikilink") for i in range(20)]
        store.upsert_links(VAULT, "hub.md", links)

        result = store.neighbors(VAULT, "hub.md")
        assert len(result["outlinks"]) == 20

    def test_upsert_links_no_commit(self, store: LibSqlStore):
        """_commit=False defers commit (for use inside transactions)."""
        store.begin_transaction()
        store.upsert_links(VAULT, "note-a.md", [("note-b", "wikilink")], _commit=False)
        store.commit_transaction()

        result = store.neighbors(VAULT, "note-a.md")
        assert result["outlinks"] == ["note-b"]


class TestBacklinkCounts:
    """Backlink counts after link persistence."""

    def _setup_docs(self, store: LibSqlStore):
        """Create minimal documents for backlink queries."""
        from mneme.models import Document

        for name in ("note-a.md", "note-b.md", "note-c.md"):
            doc_id = f"doc_{name}"
            store.upsert_document(Document(
                doc_id=doc_id,
                vault_id=VAULT,
                rel_path=name,
                file_type="markdown",
                mtime=1000,
                size=100,
                content_hash="abc",
                deleted=False,
                metadata={"extractor_version": "v1"},
            ))

    def test_backlink_counts_after_persist(self, store: LibSqlStore):
        """get_backlink_counts returns correct counts after upsert_links."""
        self._setup_docs(store)

        # note-a links to note-b and note-c
        store.upsert_links(VAULT, "note-a.md", [("note-b.md", "wikilink"), ("note-c.md", "wikilink")])
        # note-b also links to note-c
        store.upsert_links(VAULT, "note-b.md", [("note-c.md", "wikilink")])

        counts = store.get_backlink_counts()
        # note-c has 2 backlinks (from note-a and note-b)
        assert counts.get("doc_note-c.md", 0) == 2
        # note-b has 1 backlink (from note-a)
        assert counts.get("doc_note-b.md", 0) == 1
        # note-a has 0 backlinks
        assert counts.get("doc_note-a.md", 0) == 0

    def test_delete_document_clears_links(self, store: LibSqlStore):
        """delete_document removes outgoing links (existing behavior preserved)."""
        self._setup_docs(store)
        store.upsert_links(VAULT, "note-a.md", [("note-b.md", "wikilink")])

        # Verify link exists
        result = store.neighbors(VAULT, "note-a.md")
        assert result["outlinks"] == ["note-b.md"]

        # Delete the document
        store.delete_document(VAULT, "note-a.md")

        # Verify links are gone
        result = store.neighbors(VAULT, "note-a.md")
        assert result["outlinks"] == []
