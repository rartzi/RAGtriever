"""Tests for transaction boundaries and atomic writes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mneme.store.libsql_store import LibSqlStore
from mneme.models import Document, Chunk


@pytest.fixture
def store(tmp_path: Path) -> LibSqlStore:
    s = LibSqlStore(tmp_path / "test.sqlite")
    s.init()
    return s


def _make_doc(doc_id: str = "doc1", vault_id: str = "v1") -> Document:
    return Document(
        doc_id=doc_id,
        vault_id=vault_id,
        rel_path="test.md",
        file_type="markdown",
        mtime=1000,
        size=100,
        content_hash="abc123",
        deleted=False,
        metadata={"extractor_version": "v1"},
    )


def _make_chunk(chunk_id: str = "c1", doc_id: str = "doc1", vault_id: str = "v1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        vault_id=vault_id,
        anchor_type="heading",
        anchor_ref="intro",
        text="Hello world",
        text_hash="hash1",
        metadata={"rel_path": "test.md"},
    )


def test_atomic_batch_write(store: LibSqlStore) -> None:
    """All writes in a transaction succeed together."""
    doc = _make_doc()
    chunk = _make_chunk()
    vectors = np.random.randn(1, 384).astype(np.float32)

    store.begin_transaction()
    store.upsert_document(doc, _commit=False)
    store.upsert_chunks([chunk], _commit=False)
    store.upsert_embeddings(["c1"], model_id="test", vectors=vectors, _commit=False)
    store.upsert_manifest(
        vault_id="v1", rel_path="test.md", file_hash="abc123",
        chunk_count=1, mtime=1000, size=100,
    )
    store.commit_transaction()

    # Verify all data is present
    conn = store._get_conn()
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM manifest").fetchone()[0] == 1


def test_rollback_leaves_db_unchanged(store: LibSqlStore) -> None:
    """Rollback after error leaves DB in original state."""
    # Verify DB is empty first
    conn = store._get_conn()
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0

    doc = _make_doc()
    chunk = _make_chunk()

    store.begin_transaction()
    store.upsert_document(doc, _commit=False)
    store.upsert_chunks([chunk], _commit=False)
    # Simulate error: rollback instead of commit
    store.rollback_transaction()

    # Verify nothing was written
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_manifest_written_with_document(store: LibSqlStore) -> None:
    """Manifest row is written alongside document in same transaction."""
    doc = _make_doc()
    chunk = _make_chunk()

    store.begin_transaction()
    store.upsert_document(doc, _commit=False)
    store.upsert_chunks([chunk], _commit=False)
    store.upsert_manifest(
        vault_id="v1", rel_path="test.md", file_hash="abc123",
        chunk_count=1, mtime=1000, size=100,
    )
    store.commit_transaction()

    conn = store._get_conn()
    row = conn.execute(
        "SELECT * FROM manifest WHERE vault_id=? AND rel_path=?",
        ("v1", "test.md"),
    ).fetchone()
    assert row is not None
    assert row["content_hash"] == "abc123"
    assert row["mtime"] == 1000
    assert row["size"] == 100


def test_individual_writes_auto_commit(store: LibSqlStore) -> None:
    """Without explicit transaction, individual writes auto-commit by default."""
    doc = _make_doc()
    store.upsert_document(doc)  # _commit=True by default

    conn = store._get_conn()
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_manifest_upsert_updates_existing(store: LibSqlStore) -> None:
    """Upserting manifest for same file updates existing row."""
    store.begin_transaction()
    store.upsert_manifest(
        vault_id="v1", rel_path="test.md", file_hash="hash1",
        chunk_count=1, mtime=1000, size=100,
    )
    store.commit_transaction()

    store.begin_transaction()
    store.upsert_manifest(
        vault_id="v1", rel_path="test.md", file_hash="hash2",
        chunk_count=2, mtime=2000, size=200,
    )
    store.commit_transaction()

    conn = store._get_conn()
    rows = conn.execute("SELECT * FROM manifest WHERE vault_id='v1'").fetchall()
    assert len(rows) == 1
    assert rows[0]["content_hash"] == "hash2"
