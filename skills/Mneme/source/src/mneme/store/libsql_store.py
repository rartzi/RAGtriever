from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Any, Sequence, Optional
import json
import numpy as np

from ..models import Document, Chunk, SearchResult, SourceRef, OpenResult
from .faiss_index import FAISSIndex, FAISS_AVAILABLE
from .schema_manager import SchemaManager

logger = logging.getLogger(__name__)


def _escape_fts5_query(query: str) -> str:
    """Escape special characters in FTS5 queries.

    FTS5 has special syntax for operators like -, /, AND, OR, NOT, etc.
    To search for literal text containing these characters, we wrap the
    query in double quotes and escape any quotes within.
    """
    # Escape double quotes by doubling them (SQLite convention)
    escaped = query.replace('"', '""')
    # Wrap in double quotes to treat as phrase query
    return f'"{escaped}"'


class _JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime and date objects."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


def _json_dumps(obj: Any) -> str:
    """JSON serialize with datetime support."""
    return json.dumps(obj, cls=_JSONEncoder)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS vaults (
  vault_id TEXT PRIMARY KEY,
  root_path TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS documents (
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
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_vault_path ON documents(vault_id, rel_path);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  vault_id TEXT NOT NULL,
  anchor_type TEXT NOT NULL,
  anchor_ref TEXT NOT NULL,
  text TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_vault ON chunks(vault_id);

CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  dims INTEGER NOT NULL,
  vector BLOB NOT NULL
);

-- Lexical index: FTS5
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
  chunk_id UNINDEXED,
  vault_id UNINDEXED,
  rel_path UNINDEXED,
  text,
  tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS links (
  vault_id TEXT NOT NULL,
  src_rel_path TEXT NOT NULL,
  dst_target TEXT NOT NULL,
  link_type TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_links_src ON links(vault_id, src_rel_path);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(vault_id, dst_target);

CREATE TABLE IF NOT EXISTS manifest (
  vault_id TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  mtime INTEGER NOT NULL,
  size INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  last_indexed_at TEXT DEFAULT (datetime('now')),
  last_error TEXT,
  PRIMARY KEY (vault_id, rel_path)
);
"""

def _vec_to_blob(vec: np.ndarray) -> bytes:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    return vec.tobytes()

def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

class LibSqlStore:
    """SQLite/libSQL-backed store with optional FAISS vector index.

    Vector search can use:
    - Brute-force cosine similarity (default, works for <10K chunks)
    - FAISS approximate nearest neighbor (optional, for >10K chunks)
    """

    def __init__(
        self,
        db_path: Path,
        use_faiss: bool = False,
        faiss_index_type: str = "IVF",
        faiss_nlist: int = 100,
        faiss_nprobe: int = 10,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for per-thread connections
        self._local = threading.local()
        # Track all connections for cleanup
        self._connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()

        # FAISS setup
        self.use_faiss = use_faiss
        self.faiss_index: Optional[FAISSIndex] = None
        self.faiss_index_type = faiss_index_type
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self._faiss_lock = threading.Lock()
        self._brute_force_warned = False

        if use_faiss:
            if not FAISS_AVAILABLE:
                raise ImportError(
                    "FAISS requested but not installed. Install with: pip install faiss-cpu or pip install faiss-gpu"
                )

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection.

        Each thread gets its own connection with WAL mode and busy timeout.
        Connections are tracked for cleanup via close().
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
            with self._conn_lock:
                self._connections.append(conn)
        return conn

    def close(self) -> None:
        """Close all thread-local connections."""
        with self._conn_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()

    def begin_transaction(self) -> None:
        """Begin an explicit transaction on the current thread's connection."""
        self._get_conn().execute("BEGIN IMMEDIATE")

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        self._get_conn().commit()

    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        self._get_conn().rollback()

    def init(self) -> None:
        self._get_conn().executescript(SCHEMA_SQL)
        self._get_conn().commit()

        # Run schema migrations
        SchemaManager(self._get_conn()).ensure_current()

        # Initialize FAISS index if enabled
        if self.use_faiss:
            self._initialize_faiss()

    def _initialize_faiss(self) -> None:
        """Initialize FAISS index from existing embeddings (if any)."""
        with self._faiss_lock:
            if not self.use_faiss or self.faiss_index is not None:
                return

            # Check if we have any embeddings to determine dimensions
            cursor = self._get_conn().execute("SELECT dims FROM embeddings LIMIT 1")
            row = cursor.fetchone()
            if row:
                embedding_dim = row["dims"]
                self.faiss_index = FAISSIndex(
                    embedding_dim=embedding_dim,
                    index_type=self.faiss_index_type,
                    nlist=self.faiss_nlist,
                    nprobe=self.faiss_nprobe,
                    metric="cosine",
                )

                # Try to load existing FAISS index
                faiss_path = self.db_path.parent / "faiss"
                if (faiss_path / "faiss.index").exists():
                    try:
                        self.faiss_index.load(faiss_path)
                        logger.info(f"Loaded FAISS index with {self.faiss_index.size()} vectors")
                    except Exception as e:
                        logger.warning(f"Failed to load FAISS index: {e}")
                        logger.info("Building new FAISS index from embeddings...")
                        self._build_faiss_index_locked()
                else:
                    # Build index from existing embeddings
                    logger.info("Building FAISS index from embeddings...")
                    self._build_faiss_index_locked()

    def _build_faiss_index_locked(self) -> None:
        """Build FAISS index from all embeddings in database.

        Must be called while holding self._faiss_lock.
        """
        if not self.faiss_index:
            return

        logger.info("Loading embeddings from database...")
        cursor = self._get_conn().execute("""
            SELECT chunk_id, vector
            FROM embeddings
        """)

        chunk_ids = []
        vectors = []

        for row in cursor:
            chunk_ids.append(row["chunk_id"])
            vectors.append(_blob_to_vec(row["vector"]))

        if not vectors:
            logger.info("No embeddings found in database")
            return

        vectors_array = np.vstack(vectors)
        logger.info(f"Adding {len(vectors)} vectors to FAISS index...")
        self.faiss_index.add(chunk_ids, vectors_array)

        # Save index
        faiss_path = self.db_path.parent / "faiss"
        self.faiss_index.save(faiss_path)
        logger.info(f"FAISS index saved to {faiss_path}")

    def upsert_document(self, doc: Document, *, _commit: bool = True) -> None:
        self._get_conn().execute(
            """INSERT INTO documents(doc_id, vault_id, rel_path, file_type, mtime, size, content_hash, extractor_version, deleted, metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 vault_id=excluded.vault_id, rel_path=excluded.rel_path, file_type=excluded.file_type,
                 mtime=excluded.mtime, size=excluded.size, content_hash=excluded.content_hash,
                 extractor_version=excluded.extractor_version, deleted=excluded.deleted, metadata_json=excluded.metadata_json
            """,
            (doc.doc_id, doc.vault_id, doc.rel_path, doc.file_type, doc.mtime, doc.size, doc.content_hash, doc.metadata.get("extractor_version","v1"), int(doc.deleted), _json_dumps(doc.metadata or {})),
        )
        if _commit:
            self._get_conn().commit()

    def upsert_chunks(self, chunks: Sequence[Chunk], *, _commit: bool = True) -> None:
        cur = self._get_conn().cursor()

        # Batch upsert chunks with executemany
        chunk_rows = [
            (ch.chunk_id, ch.doc_id, ch.vault_id, ch.anchor_type, ch.anchor_ref,
             ch.text, ch.text_hash, _json_dumps(ch.metadata or {}))
            for ch in chunks
        ]
        cur.executemany(
            """INSERT INTO chunks(chunk_id, doc_id, vault_id, anchor_type, anchor_ref, text, text_hash, metadata_json)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(chunk_id) DO UPDATE SET
                 doc_id=excluded.doc_id, vault_id=excluded.vault_id, anchor_type=excluded.anchor_type,
                 anchor_ref=excluded.anchor_ref, text=excluded.text, text_hash=excluded.text_hash,
                 metadata_json=excluded.metadata_json
            """,
            chunk_rows,
        )

        # Batch FTS upsert: delete then insert
        fts_ids = [(ch.chunk_id,) for ch in chunks]
        cur.executemany("DELETE FROM fts_chunks WHERE chunk_id = ?", fts_ids)

        fts_rows = [
            (ch.chunk_id, ch.vault_id, ch.metadata.get("rel_path", ""), ch.text)
            for ch in chunks
        ]
        cur.executemany(
            "INSERT INTO fts_chunks(chunk_id, vault_id, rel_path, text) VALUES(?,?,?,?)",
            fts_rows,
        )

        if _commit:
            self._get_conn().commit()

    def delete_document(self, vault_id: str, rel_path: str) -> None:
        # Find doc_id
        row = self._get_conn().execute("SELECT doc_id FROM documents WHERE vault_id=? AND rel_path=? AND deleted=0", (vault_id, rel_path)).fetchone()
        if not row:
            return
        doc_id = row["doc_id"]

        # Collect chunk_ids for batch deletion
        chunk_rows = self._get_conn().execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
        chunk_ids = [r["chunk_id"] for r in chunk_rows]

        if chunk_ids:
            # Batch delete embeddings and FTS entries using IN clause
            placeholders = ",".join("?" * len(chunk_ids))
            self._get_conn().execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
            self._get_conn().execute(f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders})", chunk_ids)

        self._get_conn().execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self._get_conn().execute("UPDATE documents SET deleted=1 WHERE doc_id=?", (doc_id,))
        # Clean up links table (outgoing links from this file)
        self._get_conn().execute("DELETE FROM links WHERE vault_id=? AND src_rel_path=?", (vault_id, rel_path))
        # Clean up manifest table
        self._get_conn().execute("DELETE FROM manifest WHERE vault_id=? AND rel_path=?", (vault_id, rel_path))
        self._get_conn().commit()

    def get_indexed_files(self, vault_id: str) -> set[str]:
        """Get set of rel_paths for all non-deleted documents in vault."""
        rows = self._get_conn().execute(
            "SELECT rel_path FROM documents WHERE vault_id=? AND deleted=0",
            (vault_id,)
        ).fetchall()
        return {row["rel_path"] for row in rows}

    def get_manifest_mtimes(self, vault_id: str) -> dict[str, int]:
        """Get mtime from manifest for all indexed files in vault.

        Returns:
            Dict mapping rel_path to mtime (unix timestamp) from last index.
            Used by watcher to detect files modified while stopped.
        """
        rows = self._get_conn().execute(
            "SELECT rel_path, mtime FROM manifest WHERE vault_id=?",
            (vault_id,)
        ).fetchall()
        return {row["rel_path"]: row["mtime"] for row in rows}

    def get_manifest_entries(self, vault_id: str) -> dict[str, tuple[int, int]]:
        """Get (mtime, size) from manifest for all indexed files in vault.

        Returns:
            Dict mapping rel_path to (mtime, size) tuple from last index.
            Used by scan to skip unchanged files.
        """
        rows = self._get_conn().execute(
            "SELECT rel_path, mtime, size FROM manifest WHERE vault_id=?",
            (vault_id,)
        ).fetchall()
        return {row["rel_path"]: (row["mtime"], row["size"]) for row in rows}

    def get_files_under_path(self, vault_id: str, path_prefix: str) -> list[str]:
        """Get all indexed files under a directory path prefix.

        Args:
            vault_id: The vault identifier
            path_prefix: Directory path (e.g., "folder/subfolder")

        Returns:
            List of rel_paths for files under the directory
        """
        # Ensure path_prefix ends with / for proper matching
        if path_prefix and not path_prefix.endswith("/"):
            path_prefix = path_prefix + "/"

        # Query files where rel_path starts with the prefix
        rows = self._get_conn().execute(
            "SELECT rel_path FROM documents WHERE vault_id=? AND deleted=0 AND rel_path LIKE ?",
            (vault_id, f"{path_prefix}%")
        ).fetchall()
        return [row["rel_path"] for row in rows]

    def upsert_embeddings(self, chunk_ids: Sequence[str], model_id: str, vectors: np.ndarray, *, _commit: bool = True) -> None:
        cur = self._get_conn().cursor()

        # Batch upsert embeddings with executemany
        emb_rows = []
        for cid, vec in zip(chunk_ids, vectors, strict=False):
            blob = _vec_to_blob(vec)
            dims = int(np.asarray(vec).size)
            emb_rows.append((cid, model_id, dims, blob))

        cur.executemany(
            """INSERT INTO embeddings(chunk_id, model_id, dims, vector)
               VALUES(?,?,?,?)
               ON CONFLICT(chunk_id) DO UPDATE SET model_id=excluded.model_id, dims=excluded.dims, vector=excluded.vector
            """,
            emb_rows,
        )

        if _commit:
            self._get_conn().commit()

        # FAISS operations under lock
        if self.use_faiss and len(chunk_ids) > 0:
            with self._faiss_lock:
                # Initialize FAISS index if needed (on first embeddings)
                if self.faiss_index is None:
                    embedding_dim = vectors.shape[1]
                    self.faiss_index = FAISSIndex(
                        embedding_dim=embedding_dim,
                        index_type=self.faiss_index_type,
                        nlist=self.faiss_nlist,
                        nprobe=self.faiss_nprobe,
                        metric="cosine",
                    )
                    logger.info(f"Initialized FAISS {self.faiss_index_type} index (dim={embedding_dim})")

                # Add to FAISS index
                if self.faiss_index:
                    self.faiss_index.add(list(chunk_ids), vectors)

                    # Periodically save FAISS index (every 5000 vectors)
                    if self.faiss_index.size() % 5000 == 0:
                        faiss_path = self.db_path.parent / "faiss"
                        self.faiss_index.save(faiss_path)
                        logger.info(f"FAISS index checkpoint ({self.faiss_index.size()} vectors)")

    def save_faiss_index(self) -> None:
        """Save the FAISS index to disk. Call at end of scan to ensure final state is persisted."""
        if self.use_faiss and self.faiss_index and self.faiss_index.size() > 0:
            with self._faiss_lock:
                faiss_path = self.db_path.parent / "faiss"
                self.faiss_index.save(faiss_path)
                logger.info(f"FAISS index saved ({self.faiss_index.size()} vectors)")

    def upsert_manifest(
        self,
        vault_id: str,
        rel_path: str,
        file_hash: str,
        chunk_count: int,
        mtime: int = 0,
        size: int = 0,
        status: str = "ok",
    ) -> None:
        """Write or update a manifest entry for an indexed file.

        Called inside the same transaction as document/chunk writes
        to ensure atomicity.
        """
        self._get_conn().execute(
            """INSERT INTO manifest (vault_id, rel_path, mtime, size, content_hash, last_indexed_at, last_error)
               VALUES (?, ?, ?, ?, ?, datetime('now'), ?)
               ON CONFLICT(vault_id, rel_path) DO UPDATE SET
                 mtime=excluded.mtime, size=excluded.size, content_hash=excluded.content_hash,
                 last_indexed_at=excluded.last_indexed_at, last_error=excluded.last_error
            """,
            (vault_id, rel_path, mtime, size, file_hash, None if status == "ok" else status),
        )

    def lexical_search(self, query: str, k: int, filters: dict[str, Any]) -> list[SearchResult]:
        vault_id = filters.get("vault_id")
        vault_ids = filters.get("vault_ids")  # Support list of vault_ids for multi-vault
        path_prefix = filters.get("path_prefix", "")
        params: list[Any] = []
        where = "1=1"

        # Handle single vault_id or list of vault_ids
        if vault_ids:
            placeholders = ",".join("?" * len(vault_ids))
            where += f" AND f.vault_id IN ({placeholders})"
            params.extend(vault_ids)
        elif vault_id:
            where += " AND f.vault_id=?"
            params.append(vault_id)

        # Join with chunks table to get full metadata
        sql = f"""
        SELECT f.chunk_id, bm25(fts_chunks) AS rank, f.rel_path, f.text, f.vault_id, c.metadata_json
        FROM fts_chunks f
        LEFT JOIN chunks c ON f.chunk_id = c.chunk_id
        WHERE {where} AND fts_chunks MATCH ?
        ORDER BY rank
        LIMIT ?
        """
        # Escape special characters in query for FTS5
        escaped_query = _escape_fts5_query(query)
        params2 = params + [escaped_query, k]
        rows = self._get_conn().execute(sql, params2).fetchall()

        results: list[SearchResult] = []
        for r in rows:
            # Parse full metadata from chunks table
            meta = json.loads(r["metadata_json"] or "{}")
            rel = meta.get("rel_path", r["rel_path"] or "")
            if path_prefix and not rel.startswith(path_prefix):
                continue
            snippet = (r["text"] or "")[:600]
            row_vault_id = r["vault_id"] if "vault_id" in r.keys() else (vault_id or "")
            sr = SourceRef(
                vault_id=row_vault_id,
                rel_path=rel,
                file_type=meta.get("file_type", "unknown"),
                anchor_type=meta.get("anchor_type", "chunk"),
                anchor_ref=meta.get("anchor_ref", r["chunk_id"]),
                locator=meta.get("locator", {}),
            )
            results.append(SearchResult(
                chunk_id=r["chunk_id"],
                score=float(-r["rank"]),
                snippet=snippet,
                source_ref=sr,
                metadata={**meta, "vault_id": row_vault_id}
            ))
        return results[:k]

    def get_total_chunk_count(self) -> int:
        """Get total number of chunks across all vaults."""
        row = self._get_conn().execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def vector_search(self, query_vec: np.ndarray, k: int, filters: dict[str, Any]) -> list[SearchResult]:
        """Vector search with optional FAISS acceleration.

        Uses FAISS approximate nearest neighbor if enabled, otherwise brute-force.
        Supports single vault_id or list of vault_ids for multi-vault search.
        Warns when brute-force is used on >10K chunks (FAISS recommended).
        """
        vault_id = filters.get("vault_id")
        vault_ids = filters.get("vault_ids")  # Support list of vault_ids for multi-vault
        path_prefix = filters.get("path_prefix", "")

        # Use FAISS if enabled
        if self.use_faiss and self.faiss_index and self.faiss_index.size() > 0:
            return self._faiss_vector_search(query_vec, k, vault_id, vault_ids, path_prefix)
        else:
            # Warn if brute-force on large index
            if not self._brute_force_warned:
                total = self.get_total_chunk_count()
                if total > 10_000:
                    logger.warning(
                        f"Brute-force vector search on {total:,} chunks. "
                        f"Enable FAISS (use_faiss=true) for 100-1000x faster queries."
                    )
                    self._brute_force_warned = True
            return self._brute_force_vector_search(query_vec, k, vault_id, vault_ids, path_prefix)

    def _faiss_vector_search(
        self,
        query_vec: np.ndarray,
        k: int,
        vault_id: Optional[str],
        vault_ids: Optional[list[str]],
        path_prefix: str,
    ) -> list[SearchResult]:
        """FAISS-accelerated vector search with multi-vault support."""
        # FAISS search (returns chunk_ids and scores)
        assert self.faiss_index is not None  # Caller ensures this
        with self._faiss_lock:
            chunk_ids, scores = self.faiss_index.search(query_vec, k * 2)  # Over-fetch for filtering

        if not chunk_ids:
            return []

        # Fetch chunk details from SQLite
        placeholders = ",".join("?" * len(chunk_ids))
        query_sql = f"""
            SELECT c.chunk_id, c.text, c.metadata_json, c.vault_id
            FROM chunks c
            WHERE c.chunk_id IN ({placeholders})
        """
        params: list[Any] = list(chunk_ids)

        # Handle single vault_id or list of vault_ids
        if vault_ids:
            vault_placeholders = ",".join("?" * len(vault_ids))
            query_sql += f" AND c.vault_id IN ({vault_placeholders})"
            params.extend(vault_ids)
        elif vault_id:
            query_sql += " AND c.vault_id = ?"
            params.append(vault_id)

        rows = self._get_conn().execute(query_sql, params).fetchall()

        # Build chunk_id to row mapping
        chunk_data = {r["chunk_id"]: r for r in rows}

        # Build results maintaining FAISS score order
        results: list[SearchResult] = []
        for chunk_id, score in zip(chunk_ids, scores):
            if chunk_id not in chunk_data:
                continue

            r = chunk_data[chunk_id]
            meta = json.loads(r["metadata_json"] or "{}")
            rel = meta.get("rel_path", "")

            # Apply path prefix filter
            if path_prefix and rel and not str(rel).startswith(path_prefix):
                continue

            snippet = (r["text"] or "")[:600]
            row_vault_id = r["vault_id"]
            sr = SourceRef(
                vault_id=row_vault_id,
                rel_path=rel,
                file_type=meta.get("file_type", "unknown"),
                anchor_type=meta.get("anchor_type", "chunk"),
                anchor_ref=meta.get("anchor_ref", chunk_id),
                locator=meta.get("locator", {}),
            )
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=score,
                snippet=snippet,
                source_ref=sr,
                metadata={**meta, "vault_id": row_vault_id}
            ))

            if len(results) >= k:
                break

        return results

    def _brute_force_vector_search(
        self,
        query_vec: np.ndarray,
        k: int,
        vault_id: Optional[str],
        vault_ids: Optional[list[str]],
        path_prefix: str,
    ) -> list[SearchResult]:
        """Brute-force cosine similarity vector search with multi-vault support."""
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        qn = np.linalg.norm(q) + 1e-12

        # Build query based on vault filtering
        base_sql = """
            SELECT e.chunk_id, e.vector, c.text, c.metadata_json, c.vault_id
            FROM embeddings e JOIN chunks c ON c.chunk_id=e.chunk_id
        """

        if vault_ids:
            # Filter by list of vault_ids
            placeholders = ",".join("?" * len(vault_ids))
            sql = base_sql + f" WHERE c.vault_id IN ({placeholders})"
            rows = self._get_conn().execute(sql, vault_ids).fetchall()
        elif vault_id:
            # Filter by single vault_id
            sql = base_sql + " WHERE c.vault_id=?"
            rows = self._get_conn().execute(sql, (vault_id,)).fetchall()
        else:
            # No vault filter - search all
            rows = self._get_conn().execute(base_sql).fetchall()

        scored = []
        for r in rows:
            meta = json.loads(r["metadata_json"] or "{}")
            rel = meta.get("rel_path", "")
            if path_prefix and rel and not str(rel).startswith(path_prefix):
                continue
            v = _blob_to_vec(r["vector"])
            vn = np.linalg.norm(v) + 1e-12
            sim = float(np.dot(q, v) / (qn * vn))
            scored.append((sim, r, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchResult] = []
        for sim, r, meta in scored[:k]:
            rel = meta.get("rel_path", "")
            snippet = (r["text"] or "")[:600]
            row_vault_id = r["vault_id"] if "vault_id" in r.keys() else (vault_id or "")
            sr = SourceRef(
                vault_id=row_vault_id,
                rel_path=rel,
                file_type=meta.get("file_type", "unknown"),
                anchor_type=meta.get("anchor_type", "chunk"),
                anchor_ref=meta.get("anchor_ref", r["chunk_id"]),
                locator=meta.get("locator", {}),
            )
            results.append(SearchResult(
                chunk_id=r["chunk_id"],
                score=sim,
                snippet=snippet,
                source_ref=sr,
                metadata={**meta, "vault_id": row_vault_id}
            ))
        return results

    def open(self, source_ref: SourceRef) -> OpenResult:
        # In v1, open by chunk_id stored as anchor_ref or locator. Agent should implement anchor-based open.
        cid = source_ref.anchor_ref
        row = self._get_conn().execute("SELECT text, metadata_json FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
        if not row:
            return OpenResult(content="", source_ref=source_ref, metadata={"error": "not_found"})
        meta = json.loads(row["metadata_json"] or "{}")
        return OpenResult(content=row["text"], source_ref=source_ref, metadata=meta)

    def status(self, vault_id: str) -> dict[str, Any]:
        files = self._get_conn().execute("SELECT COUNT(*) AS n FROM documents WHERE vault_id=? AND deleted=0", (vault_id,)).fetchone()["n"]
        chunks = self._get_conn().execute("SELECT COUNT(*) AS n FROM chunks WHERE vault_id=?", (vault_id,)).fetchone()["n"]
        return {"vault_id": vault_id, "indexed_files": int(files), "indexed_chunks": int(chunks), "last_scan_at": None, "errors": []}

    def neighbors(self, vault_id: str, rel_path: str, depth: int = 1) -> dict[str, Any]:
        outlinks = [r["dst_target"] for r in self._get_conn().execute("SELECT dst_target FROM links WHERE vault_id=? AND src_rel_path=?", (vault_id, rel_path)).fetchall()]
        backlinks = [r["src_rel_path"] for r in self._get_conn().execute("SELECT src_rel_path FROM links WHERE vault_id=? AND dst_target=?", (vault_id, rel_path)).fetchall()]
        return {"outlinks": outlinks, "backlinks": backlinks}

    def get_backlink_counts(self, doc_ids: list[str] | None = None) -> dict[str, int]:
        """Get count of incoming links (backlinks) for documents.

        Args:
            doc_ids: Optional list of doc_ids to filter. If None, returns all.

        Returns:
            Dict mapping doc_id -> number of documents linking to it
        """
        # Query the links table - group by target, count distinct sources
        # The 'dst_target' column stores the wikilink target (rel_path typically)
        # Need to join with documents to get doc_id
        query = """
            SELECT d.doc_id, COUNT(DISTINCT l.src_rel_path) as backlink_count
            FROM links l
            JOIN documents d ON l.dst_target = d.rel_path AND l.vault_id = d.vault_id
            WHERE d.deleted = 0
            GROUP BY d.doc_id
        """

        cursor = self._get_conn().execute(query)
        result = {row["doc_id"]: row["backlink_count"] for row in cursor.fetchall()}

        if doc_ids:
            return {k: v for k, v in result.items() if k in doc_ids}
        return result
