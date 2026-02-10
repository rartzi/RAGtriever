"""Tests for agentic search tools (list_docs, text_search, backlinks).

Covers MCP tool functions, query server actions, and backward compatibility.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mneme.hashing import blake2b_hex
from mneme.models import Document, Chunk
from mneme.store.libsql_store import LibSqlStore

VAULT_ID = "testvault01"  # 11 chars so [:12] doesn't truncate it

# blake2b_hex mock: must return a string where [:12] == VAULT_ID
# Pad to 64 hex chars. [:12] of this is "testvault01a" — but we use VAULT_ID directly
# for data insertion. For the blake2b mock, we need [:12] == VAULT_ID.
# VAULT_ID is 11 chars, so we pad with one more char then the rest.
BLAKE_MOCK_RETURN = VAULT_ID + "0" + "a" * 52  # [:12] = "testvault010"

# Actually, let's make this simpler: use a 12-char VAULT_ID so [:12] matches exactly.
VAULT_ID = "tvault000001"  # exactly 12 chars
BLAKE_MOCK_RETURN = VAULT_ID + "a" * 52  # [:12] = VAULT_ID exactly


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> LibSqlStore:
    """Create a test store with sample documents, chunks, and links."""
    db = LibSqlStore(tmp_path / "test.sqlite")
    db.init()

    # Insert 5 documents
    docs = [
        ("projects/alpha.md", "markdown"),
        ("projects/beta.md", "markdown"),
        ("notes/daily.md", "markdown"),
        ("notes/ideas.md", "markdown"),
        ("readme.md", "markdown"),
    ]
    for rel_path, ftype in docs:
        doc_id = blake2b_hex(f"{VAULT_ID}:{rel_path}".encode("utf-8"))[:24]
        db.upsert_document(Document(
            doc_id=doc_id,
            vault_id=VAULT_ID,
            rel_path=rel_path,
            file_type=ftype,
            mtime=1000,
            size=100,
            content_hash=f"hash_{rel_path}",
            deleted=False,
        ))

        # Insert a chunk per document with searchable text
        chunk_id = blake2b_hex(f"{doc_id}:chunk0".encode("utf-8"))[:32]
        text_map = {
            "projects/alpha.md": "Alpha project uses agentic workflows for automation",
            "projects/beta.md": "Beta project focuses on retrieval augmented generation",
            "notes/daily.md": "Daily standup notes for the team meeting",
            "notes/ideas.md": "Ideas about improving search with semantic embeddings",
            "readme.md": "README file with project overview and setup instructions",
        }
        db.upsert_chunks([Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            vault_id=VAULT_ID,
            anchor_type="heading",
            anchor_ref="h1",
            text=text_map[rel_path],
            text_hash=blake2b_hex(text_map[rel_path].encode("utf-8")),
            metadata={"rel_path": rel_path},
        )])

    # Insert wikilinks: alpha -> beta, alpha -> readme, daily -> alpha, ideas -> alpha
    conn = db._get_conn()
    links = [
        (VAULT_ID, "projects/alpha.md", "projects/beta.md", "wikilink"),
        (VAULT_ID, "projects/alpha.md", "readme.md", "wikilink"),
        (VAULT_ID, "notes/daily.md", "projects/alpha.md", "wikilink"),
        (VAULT_ID, "notes/ideas.md", "projects/alpha.md", "wikilink"),
    ]
    for vid, src, dst, ltype in links:
        conn.execute(
            "INSERT INTO links(vault_id, src_rel_path, dst_target, link_type) VALUES (?, ?, ?, ?)",
            (vid, src, dst, ltype),
        )
    conn.commit()

    return db


@pytest.fixture
def vault_id() -> str:
    return VAULT_ID


@pytest.fixture
def retriever(store: LibSqlStore, tmp_path: Path) -> MagicMock:
    """Create a mock retriever with the real store attached."""
    r = MagicMock()
    r.store = store
    r.cfg = MagicMock()
    r.cfg.vault_root = Path("/fake/vault")
    r.cfg.top_k = 10
    return r


def _patch_vault_id():
    """Patch _get_vault_id to return our test vault_id."""
    return patch("mneme.mcp.tools._get_vault_id", return_value=VAULT_ID)


def _patch_blake2b():
    """Patch blake2b_hex in query_server so [:12] returns VAULT_ID."""
    return patch("mneme.query_server.blake2b_hex", return_value=BLAKE_MOCK_RETURN)


# ── MCP Tool Tests ──────────────────────────────────────────────────


class TestToolListDocs:
    def test_list_all(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_list_docs
        with _patch_vault_id():
            result = tool_list_docs(retriever, {})
        assert result["count"] == 5
        assert "projects/alpha.md" in result["files"]
        assert "readme.md" in result["files"]
        assert result["path_filter"] is None
        assert result["files"] == sorted(result["files"])

    def test_with_path_filter(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_list_docs
        with _patch_vault_id():
            result = tool_list_docs(retriever, {"path": "projects/"})
        assert result["count"] == 2
        assert all(f.startswith("projects/") for f in result["files"])
        assert result["path_filter"] == "projects/"

    def test_empty_vault(self, tmp_path: Path):
        from mneme.mcp.tools import tool_list_docs
        db = LibSqlStore(tmp_path / "empty.sqlite")
        db.init()
        r = MagicMock()
        r.store = db
        with patch("mneme.mcp.tools._get_vault_id", return_value="emptyvault"):
            result = tool_list_docs(r, {})
        assert result["count"] == 0
        assert result["files"] == []


class TestToolTextSearch:
    def test_exact_match(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_text_search
        with _patch_vault_id():
            result = tool_text_search(retriever, {"query": "agentic workflows"})
        assert len(result["results"]) >= 1
        snippets = [r["snippet"] for r in result["results"]]
        assert any("agentic" in s.lower() for s in snippets)

    def test_with_path_filter(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_text_search
        with _patch_vault_id():
            result = tool_text_search(retriever, {"query": "search", "path": "notes/"})
        for r in result["results"]:
            src = r["source_ref"]
            assert src["rel_path"].startswith("notes/")

    def test_no_results(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_text_search
        with _patch_vault_id():
            result = tool_text_search(retriever, {"query": "xyznonexistent"})
        assert result["results"] == []

    def test_custom_k(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_text_search
        with _patch_vault_id():
            result = tool_text_search(retriever, {"query": "project", "k": 1})
        assert len(result["results"]) <= 1


class TestToolBacklinks:
    def test_all_backlinks(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_backlinks
        result = tool_backlinks(retriever, {})
        # alpha.md has 2 backlinks (from daily.md and ideas.md)
        assert result["count"] > 0
        assert any("alpha" in path for path in result["backlinks"])

    def test_with_limit(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_backlinks
        result = tool_backlinks(retriever, {"limit": 1})
        assert result["count"] <= 1

    def test_specific_paths(self, retriever: MagicMock, store: LibSqlStore):
        from mneme.mcp.tools import tool_backlinks
        result = tool_backlinks(retriever, {"paths": ["projects/alpha.md"]})
        assert result["count"] >= 0

    def test_no_links(self, tmp_path: Path):
        from mneme.mcp.tools import tool_backlinks
        db = LibSqlStore(tmp_path / "nolinks.sqlite")
        db.init()
        r = MagicMock()
        r.store = db
        result = tool_backlinks(r, {})
        assert result["count"] == 0
        assert result["backlinks"] == {}


# ── Query Server Tests ──────────────────────────────────────────────


class TestQueryServerActions:
    def test_list_docs_action(self, retriever: MagicMock, store: LibSqlStore, tmp_path: Path):
        from mneme.query_server import QueryServer

        retriever.cfg.vault_root = tmp_path / "vault"
        server = QueryServer(retriever, tmp_path / "test.sock")

        with _patch_blake2b():
            resp = server._handle_request({"action": "list_docs"})
        assert "files" in resp
        assert resp["source"] == "watcher"
        assert resp["count"] == 5

    def test_text_search_action(self, retriever: MagicMock, store: LibSqlStore, tmp_path: Path):
        from mneme.query_server import QueryServer

        retriever.cfg.vault_root = tmp_path / "vault"
        server = QueryServer(retriever, tmp_path / "test.sock")

        with _patch_blake2b():
            resp = server._handle_request({"action": "text_search", "query": "agentic"})
        assert "results" in resp
        assert resp["source"] == "watcher"

    def test_backlinks_action(self, retriever: MagicMock, store: LibSqlStore, tmp_path: Path):
        from mneme.query_server import QueryServer

        server = QueryServer(retriever, tmp_path / "test.sock")
        resp = server._handle_request({"action": "backlinks"})
        assert "backlinks" in resp
        assert resp["source"] == "watcher"

    def test_ping_still_works(self, retriever: MagicMock, tmp_path: Path):
        from mneme.query_server import QueryServer

        server = QueryServer(retriever, tmp_path / "test.sock")
        resp = server._handle_request({"action": "ping"})
        assert resp["status"] == "ok"
        assert resp["source"] == "watcher"

    def test_unknown_action_error(self, retriever: MagicMock, tmp_path: Path):
        from mneme.query_server import QueryServer

        server = QueryServer(retriever, tmp_path / "test.sock")
        resp = server._handle_request({"action": "invalid_action"})
        assert "error" in resp
        assert "Unknown action" in resp["error"]


class TestQueryServerClient:
    def test_request_via_socket_returns_none_no_socket(self, tmp_path: Path):
        from mneme.query_server import request_via_socket
        result = request_via_socket(tmp_path / "nonexistent.sock", {"action": "ping"})
        assert result is None

    def test_query_via_socket_backward_compat(self, tmp_path: Path):
        """query_via_socket still works as before (wraps request_via_socket)."""
        from mneme.query_server import query_via_socket
        result = query_via_socket(tmp_path / "nonexistent.sock", "test query")
        assert result is None

    def test_request_via_socket_roundtrip(self, retriever: MagicMock):
        """Test actual socket communication."""
        from mneme.query_server import QueryServer, request_via_socket
        import time

        # Use /tmp directly to avoid AF_UNIX path length limit
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            sock_path = Path(td) / "rt.sock"
            server = QueryServer(retriever, sock_path)
            server.start()
            time.sleep(0.1)

            try:
                resp = request_via_socket(sock_path, {"action": "ping"})
                assert resp is not None
                assert resp["status"] == "ok"
            finally:
                server.stop()
