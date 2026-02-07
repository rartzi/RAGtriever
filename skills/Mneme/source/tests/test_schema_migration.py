"""Tests for schema migration system."""
from __future__ import annotations

import sqlite3

from mneme.store.schema_manager import SchemaManager, CURRENT_SCHEMA_VERSION


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_fresh_db_gets_current_version():
    """A brand-new database should be stamped with CURRENT_SCHEMA_VERSION."""
    conn = _make_conn()
    mgr = SchemaManager(conn)
    mgr.ensure_current()

    row = conn.execute(
        "SELECT value FROM schema_metadata WHERE key='schema_version'"
    ).fetchone()
    assert row is not None
    assert int(row["value"]) == CURRENT_SCHEMA_VERSION


def test_existing_db_detected_and_migrated():
    """A pre-migration database (has documents table, no schema_metadata) should
    be detected and stamped as v1."""
    conn = _make_conn()
    # Simulate pre-migration DB: create documents table but no schema_metadata
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            vault_id TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            mtime INTEGER NOT NULL,
            size INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            extractor_version TEXT NOT NULL,
            deleted INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT
        )
    """)
    conn.commit()

    mgr = SchemaManager(conn)
    mgr.ensure_current()

    row = conn.execute(
        "SELECT value FROM schema_metadata WHERE key='schema_version'"
    ).fetchone()
    assert row is not None
    assert int(row["value"]) == 1


def test_idempotent_ensure_current():
    """Calling ensure_current() twice should be safe."""
    conn = _make_conn()
    mgr = SchemaManager(conn)
    mgr.ensure_current()
    mgr.ensure_current()

    rows = conn.execute(
        "SELECT value FROM schema_metadata WHERE key='schema_version'"
    ).fetchall()
    assert len(rows) == 1
    assert int(rows[0]["value"]) == CURRENT_SCHEMA_VERSION


def test_version_starts_at_zero_without_metadata():
    """Without schema_metadata table entries, version should be 0."""
    conn = _make_conn()
    mgr = SchemaManager(conn)
    mgr._ensure_metadata_table()
    assert mgr._get_version() == 0


def test_has_existing_schema_false_on_empty():
    """Empty DB should not be detected as having existing schema."""
    conn = _make_conn()
    mgr = SchemaManager(conn)
    assert mgr._has_existing_schema() is False


def test_has_existing_schema_true_with_documents():
    """DB with documents table should be detected as existing schema."""
    conn = _make_conn()
    conn.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY)")
    conn.commit()
    mgr = SchemaManager(conn)
    assert mgr._has_existing_schema() is True
