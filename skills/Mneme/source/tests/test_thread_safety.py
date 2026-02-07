"""Tests for thread safety: connection isolation, concurrent writes, WAL mode."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from mneme.store.libsql_store import LibSqlStore
from mneme.models import Document, Chunk


@pytest.fixture
def store(tmp_path: Path) -> LibSqlStore:
    s = LibSqlStore(tmp_path / "test.sqlite")
    s.init()
    return s


def _make_doc(i: int, vault_id: str = "v1") -> Document:
    return Document(
        doc_id=f"doc{i}",
        vault_id=vault_id,
        rel_path=f"file{i}.md",
        file_type="markdown",
        mtime=1000 + i,
        size=100 + i,
        content_hash=f"hash{i}",
        deleted=False,
        metadata={"extractor_version": "v1"},
    )


def _make_chunk(i: int, doc_id: str = "doc1", vault_id: str = "v1") -> Chunk:
    return Chunk(
        chunk_id=f"chunk{i}",
        doc_id=doc_id,
        vault_id=vault_id,
        anchor_type="heading",
        anchor_ref=f"section{i}",
        text=f"Content for chunk {i}",
        text_hash=f"thash{i}",
        metadata={"rel_path": f"file{i}.md"},
    )


def test_each_thread_gets_own_connection(store: LibSqlStore) -> None:
    """Each thread should get its own connection via _get_conn()."""
    conn_ids: list[int] = []
    lock = threading.Lock()

    def get_conn_id():
        conn = store._get_conn()
        with lock:
            conn_ids.append(id(conn))

    threads = [threading.Thread(target=get_conn_id) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All connection IDs should be unique (different objects per thread)
    assert len(set(conn_ids)) == 5


def test_concurrent_writes_no_corruption(store: LibSqlStore) -> None:
    """Multiple threads writing documents concurrently should not corrupt DB."""
    errors: list[Exception] = []
    lock = threading.Lock()

    def write_docs(start: int, count: int):
        try:
            for i in range(start, start + count):
                doc = _make_doc(i)
                store.upsert_document(doc)
        except Exception as e:
            with lock:
                errors.append(e)

    # 4 threads, each writing 10 documents
    threads = [
        threading.Thread(target=write_docs, args=(i * 10, 10))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent writes: {errors}"

    # Verify all 40 documents were written
    conn = store._get_conn()
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 40


def test_concurrent_reads_during_writes(store: LibSqlStore) -> None:
    """Reads should not block during writes (WAL mode)."""
    # Pre-populate with some data
    for i in range(10):
        store.upsert_document(_make_doc(i))

    read_results: list[int] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def reader():
        try:
            conn = store._get_conn()
            count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            with lock:
                read_results.append(count)
        except Exception as e:
            with lock:
                errors.append(e)

    def writer():
        try:
            for i in range(10, 20):
                store.upsert_document(_make_doc(i))
                time.sleep(0.01)  # Small delay to interleave with reads
        except Exception as e:
            with lock:
                errors.append(e)

    # Start writer and multiple readers concurrently
    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader) for _ in range(5)]

    writer_thread.start()
    for t in reader_threads:
        t.start()

    writer_thread.join()
    for t in reader_threads:
        t.join()

    assert errors == [], f"Errors during concurrent read/write: {errors}"
    # All readers should have gotten a valid count (10 or more)
    for count in read_results:
        assert count >= 10


def test_wal_mode_enabled(store: LibSqlStore) -> None:
    """WAL mode should be enabled on connections."""
    conn = store._get_conn()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_busy_timeout_set(store: LibSqlStore) -> None:
    """Busy timeout should be set to handle contention."""
    conn = store._get_conn()
    timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout == 5000


def test_close_cleans_up_connections(tmp_path: Path) -> None:
    """close() should clean up all tracked connections."""
    store = LibSqlStore(tmp_path / "test.sqlite")
    store.init()

    # Create connections from multiple threads
    def touch_conn(event: threading.Event):
        store._get_conn()
        event.set()

    events = [threading.Event() for _ in range(3)]
    threads = [threading.Thread(target=touch_conn, args=(e,)) for e in events]
    for t in threads:
        t.start()
    for e in events:
        e.wait()
    for t in threads:
        t.join()

    assert len(store._connections) >= 3  # At least 3 thread connections + possibly main

    store.close()
    assert len(store._connections) == 0


def test_concurrent_chunk_writes(store: LibSqlStore) -> None:
    """Concurrent chunk writes from multiple threads succeed."""
    # Create document first
    store.upsert_document(_make_doc(0))

    errors: list[Exception] = []
    lock = threading.Lock()

    def write_chunks(start: int, count: int):
        try:
            for i in range(start, start + count):
                chunk = _make_chunk(i, doc_id="doc0")
                store.upsert_chunks([chunk])
        except Exception as e:
            with lock:
                errors.append(e)

    threads = [
        threading.Thread(target=write_chunks, args=(i * 5, 5))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent chunk writes: {errors}"

    conn = store._get_conn()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 20
