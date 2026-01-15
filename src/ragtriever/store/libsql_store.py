from __future__ import annotations

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Any, Sequence, Optional
import json
import numpy as np

from ..models import Document, Chunk, SearchResult, SourceRef, OpenResult


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
    """SQLite/libSQL-backed store.

    NOTE: Vector search is placeholder in this skeleton:
    - embeddings are stored
    - vector_search currently does brute-force cosine over filtered rows

    A coding agent can replace this with:
    - libSQL native vector indexing (if available locally)
    - or external vector store adapter
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

    def init(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def upsert_document(self, doc: Document) -> None:
        self._conn.execute(
            """INSERT INTO documents(doc_id, vault_id, rel_path, file_type, mtime, size, content_hash, extractor_version, deleted, metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 vault_id=excluded.vault_id, rel_path=excluded.rel_path, file_type=excluded.file_type,
                 mtime=excluded.mtime, size=excluded.size, content_hash=excluded.content_hash,
                 extractor_version=excluded.extractor_version, deleted=excluded.deleted, metadata_json=excluded.metadata_json
            """,
            (doc.doc_id, doc.vault_id, doc.rel_path, doc.file_type, doc.mtime, doc.size, doc.content_hash, doc.metadata.get("extractor_version","v1"), int(doc.deleted), _json_dumps(doc.metadata or {})),
        )
        self._conn.commit()

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        cur = self._conn.cursor()
        for ch in chunks:
            cur.execute(
                """INSERT INTO chunks(chunk_id, doc_id, vault_id, anchor_type, anchor_ref, text, text_hash, metadata_json)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                     doc_id=excluded.doc_id, vault_id=excluded.vault_id, anchor_type=excluded.anchor_type,
                     anchor_ref=excluded.anchor_ref, text=excluded.text, text_hash=excluded.text_hash,
                     metadata_json=excluded.metadata_json
                """,
                (ch.chunk_id, ch.doc_id, ch.vault_id, ch.anchor_type, ch.anchor_ref, ch.text, ch.text_hash, _json_dumps(ch.metadata or {})),
            )
            # FTS upsert: easiest is delete then insert
            cur.execute("DELETE FROM fts_chunks WHERE chunk_id = ?", (ch.chunk_id,))
            cur.execute("INSERT INTO fts_chunks(chunk_id, vault_id, rel_path, text) VALUES(?,?,?,?)",
                        (ch.chunk_id, ch.vault_id, ch.metadata.get("rel_path",""), ch.text))
        self._conn.commit()

    def delete_document(self, vault_id: str, rel_path: str) -> None:
        # Find doc_id
        row = self._conn.execute("SELECT doc_id FROM documents WHERE vault_id=? AND rel_path=? AND deleted=0", (vault_id, rel_path)).fetchone()
        if not row:
            return
        doc_id = row["doc_id"]
        # Delete chunks and embeddings and fts
        chunk_rows = self._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
        for r in chunk_rows:
            cid = r["chunk_id"]
            self._conn.execute("DELETE FROM embeddings WHERE chunk_id=?", (cid,))
            self._conn.execute("DELETE FROM fts_chunks WHERE chunk_id=?", (cid,))
        self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self._conn.execute("UPDATE documents SET deleted=1 WHERE doc_id=?", (doc_id,))
        self._conn.commit()

    def upsert_embeddings(self, chunk_ids: Sequence[str], model_id: str, vectors: np.ndarray) -> None:
        cur = self._conn.cursor()
        for cid, vec in zip(chunk_ids, vectors, strict=False):
            blob = _vec_to_blob(vec)
            dims = int(np.asarray(vec).size)
            cur.execute(
                """INSERT INTO embeddings(chunk_id, model_id, dims, vector)
                   VALUES(?,?,?,?)
                   ON CONFLICT(chunk_id) DO UPDATE SET model_id=excluded.model_id, dims=excluded.dims, vector=excluded.vector
                """,
                (cid, model_id, dims, blob),
            )
        self._conn.commit()

    def lexical_search(self, query: str, k: int, filters: dict[str, Any]) -> list[SearchResult]:
        vault_id = filters.get("vault_id")
        path_prefix = filters.get("path_prefix", "")
        params = []
        where = "1=1"
        if vault_id:
            where += " AND vault_id=?"
            params.append(vault_id)

        sql = f"""
        SELECT chunk_id, bm25(fts_chunks) AS rank, rel_path, text
        FROM fts_chunks
        WHERE {where} AND fts_chunks MATCH ?
        ORDER BY rank
        LIMIT ?
        """
        params2 = params + [query, k]
        rows = self._conn.execute(sql, params2).fetchall()

        results: list[SearchResult] = []
        for r in rows:
            rel = r["rel_path"] or ""
            if path_prefix and not rel.startswith(path_prefix):
                continue
            snippet = (r["text"] or "")[:600]
            sr = SourceRef(
                vault_id=vault_id or "",
                rel_path=rel,
                file_type="unknown",
                anchor_type="chunk",
                anchor_ref=r["chunk_id"],
                locator={},
            )
            results.append(SearchResult(chunk_id=r["chunk_id"], score=float(-r["rank"]), snippet=snippet, source_ref=sr, metadata={"rel_path": rel}))
        return results[:k]

    def vector_search(self, query_vec: np.ndarray, k: int, filters: dict[str, Any]) -> list[SearchResult]:
        # Brute-force cosine similarity over embeddings. Replace with ANN/vector index in production.
        vault_id = filters.get("vault_id")
        path_prefix = filters.get("path_prefix", "")
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        qn = np.linalg.norm(q) + 1e-12

        rows = self._conn.execute(
            """SELECT e.chunk_id, e.vector, c.text, c.metadata_json
               FROM embeddings e JOIN chunks c ON c.chunk_id=e.chunk_id
               WHERE c.vault_id=?
            """,
            (vault_id,) if vault_id else ("",),
        ).fetchall() if vault_id else self._conn.execute(
            """SELECT e.chunk_id, e.vector, c.text, c.metadata_json, c.vault_id
               FROM embeddings e JOIN chunks c ON c.chunk_id=e.chunk_id
            """).fetchall()

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
            sr = SourceRef(
                vault_id=vault_id or meta.get("vault_id",""),
                rel_path=rel,
                file_type=meta.get("file_type","unknown"),
                anchor_type=meta.get("anchor_type","chunk"),
                anchor_ref=meta.get("anchor_ref", r["chunk_id"]),
                locator=meta.get("locator", {}),
            )
            results.append(SearchResult(chunk_id=r["chunk_id"], score=sim, snippet=snippet, source_ref=sr, metadata=meta))
        return results

    def open(self, source_ref: SourceRef) -> OpenResult:
        # In v1, open by chunk_id stored as anchor_ref or locator. Agent should implement anchor-based open.
        cid = source_ref.anchor_ref
        row = self._conn.execute("SELECT text, metadata_json FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
        if not row:
            return OpenResult(content="", source_ref=source_ref, metadata={"error": "not_found"})
        meta = json.loads(row["metadata_json"] or "{}")
        return OpenResult(content=row["text"], source_ref=source_ref, metadata=meta)

    def status(self, vault_id: str) -> dict[str, Any]:
        files = self._conn.execute("SELECT COUNT(*) AS n FROM documents WHERE vault_id=? AND deleted=0", (vault_id,)).fetchone()["n"]
        chunks = self._conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE vault_id=?", (vault_id,)).fetchone()["n"]
        return {"vault_id": vault_id, "indexed_files": int(files), "indexed_chunks": int(chunks), "last_scan_at": None, "errors": []}

    def neighbors(self, vault_id: str, rel_path: str, depth: int = 1) -> dict[str, Any]:
        outlinks = [r["dst_target"] for r in self._conn.execute("SELECT dst_target FROM links WHERE vault_id=? AND src_rel_path=?", (vault_id, rel_path)).fetchall()]
        backlinks = [r["src_rel_path"] for r in self._conn.execute("SELECT src_rel_path FROM links WHERE vault_id=? AND dst_target=?", (vault_id, rel_path)).fetchall()]
        return {"outlinks": outlinks, "backlinks": backlinks}
