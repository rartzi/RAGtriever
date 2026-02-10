"""Lightweight Unix socket query server for the watcher.

When the watcher is running, this server accepts query requests over a unix
domain socket so CLI queries can skip Python + model startup (~2s savings).

Protocol: newline-delimited JSON
  Request:  {"action":"query","query":"...","k":10,"filters":{},"vault_names":null}\n
  Response: {"results":[...],"elapsed":0.5,"source":"watcher"}\n
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Any

from .hashing import blake2b_hex

logger = logging.getLogger(__name__)

SOCKET_FILENAME = "query.sock"


def get_socket_path(index_dir: Path) -> Path:
    """Return the socket path for a given index directory."""
    return index_dir / SOCKET_FILENAME


class QueryServer:
    """Unix domain socket server that serves queries using a warm Retriever."""

    def __init__(self, retriever: Any, socket_path: Path) -> None:
        self.retriever = retriever
        self.socket_path = socket_path
        self._server_socket: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the query server in a daemon thread."""
        # Remove stale socket file
        if self.socket_path.exists():
            self.socket_path.unlink()

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(str(self.socket_path))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)  # Allow periodic check for shutdown
        self._running = True

        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()
        logger.info(f"Query server listening on {self.socket_path}")

    def stop(self) -> None:
        """Stop the query server and clean up the socket file."""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        logger.info("Query server stopped")

    def _serve_loop(self) -> None:
        """Accept connections and handle queries."""
        while self._running:
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            # Handle each connection in a thread to avoid blocking
            threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single client connection."""
        try:
            conn.settimeout(30.0)
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            if not data:
                return

            request = json.loads(data.decode("utf-8").strip())
            response = self._handle_request(request)
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

        except Exception as e:
            try:
                error_resp = {"error": str(e), "source": "watcher"}
                conn.sendall((json.dumps(error_resp) + "\n").encode("utf-8"))
            except OSError:
                pass
            logger.debug(f"Query server error: {e}")
        finally:
            conn.close()

    def _handle_request(self, request: dict) -> dict:
        """Process a request and return results."""
        action = request.get("action", "query")

        if action == "ping":
            return {"status": "ok", "source": "watcher"}
        elif action == "query":
            return self._handle_query(request)
        elif action == "list_docs":
            return self._handle_list_docs(request)
        elif action == "text_search":
            return self._handle_text_search(request)
        elif action == "backlinks":
            return self._handle_backlinks(request)
        else:
            return {"error": f"Unknown action: {action}", "source": "watcher"}

    def _handle_query(self, request: dict) -> dict:
        """Process a query request and return results."""
        query = request.get("query", "")
        k = request.get("k", 10)
        filters = request.get("filters") or {}
        vault_names = request.get("vault_names")

        t0 = time.monotonic()

        # Use the appropriate search method based on retriever type
        if vault_names is not None and hasattr(self.retriever, "search") and "vault_names" in self.retriever.search.__code__.co_varnames:
            hits = self.retriever.search(query, k=k, vault_names=vault_names, filters=filters)
        else:
            hits = self.retriever.search(query, k=k, filters=filters)

        elapsed = time.monotonic() - t0

        results = []
        for h in hits:
            result_dict = {
                "chunk_id": h.chunk_id,
                "score": h.score,
                "snippet": h.snippet,
                "source_ref": h.source_ref.__dict__,
                "metadata": h.metadata,
            }
            if h.metadata.get("reranked"):
                result_dict["reranking"] = {
                    "original_score": h.metadata.get("original_score"),
                    "reranker_score": h.metadata.get("reranker_score"),
                    "reranked": True,
                }
            results.append(result_dict)

        return {"results": results, "elapsed": round(elapsed, 3), "source": "watcher"}

    def _handle_list_docs(self, request: dict) -> dict:
        """List indexed documents."""

        path_prefix = request.get("path", "")
        t0 = time.monotonic()

        vault_id = blake2b_hex(str(self.retriever.cfg.vault_root).encode("utf-8"))[:12]
        if path_prefix:
            files = sorted(self.retriever.store.get_files_under_path(vault_id, path_prefix))
        else:
            files = sorted(self.retriever.store.get_indexed_files(vault_id))

        elapsed = time.monotonic() - t0
        return {
            "files": files,
            "count": len(files),
            "path_filter": path_prefix or None,
            "elapsed": round(elapsed, 3),
            "source": "watcher",
        }

    def _handle_text_search(self, request: dict) -> dict:
        """Lexical (BM25) text search."""

        query = request.get("query", "")
        k = request.get("k", 20)
        path_prefix = request.get("path", "")

        filters: dict[str, Any] = {}
        vault_id = blake2b_hex(str(self.retriever.cfg.vault_root).encode("utf-8"))[:12]
        filters["vault_id"] = vault_id
        if path_prefix:
            filters["path_prefix"] = path_prefix

        t0 = time.monotonic()
        results = self.retriever.store.lexical_search(query, k, filters)
        elapsed = time.monotonic() - t0

        return {
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "snippet": r.snippet,
                    "source_ref": r.source_ref.__dict__,
                    "metadata": r.metadata,
                } for r in results
            ],
            "elapsed": round(elapsed, 3),
            "source": "watcher",
        }

    def _handle_backlinks(self, request: dict) -> dict:
        """Get backlink counts."""
        paths = request.get("paths")
        limit = request.get("limit", 20)

        t0 = time.monotonic()
        counts = self.retriever.store.get_backlink_counts(doc_ids=paths)

        # Map doc_id (hash) back to rel_path
        conn = self.retriever.store._get_conn()
        if counts:
            placeholders = ",".join("?" * len(counts))
            rows = conn.execute(
                f"SELECT doc_id, rel_path FROM documents WHERE doc_id IN ({placeholders}) AND deleted=0",
                list(counts.keys()),
            ).fetchall()
            id_to_path = {r["doc_id"]: r["rel_path"] for r in rows}
        else:
            id_to_path = {}

        backlinks = {
            id_to_path.get(did, did): count
            for did, count in counts.items()
        }
        sorted_backlinks = dict(
            sorted(backlinks.items(), key=lambda x: x[1], reverse=True)[:limit]
        )

        elapsed = time.monotonic() - t0
        return {
            "backlinks": sorted_backlinks,
            "count": len(sorted_backlinks),
            "elapsed": round(elapsed, 3),
            "source": "watcher",
        }


def request_via_socket(socket_path: Path, request: dict) -> dict | None:
    """Send a generic request to a running watcher's query server.

    Returns the response dict, or None if the socket is unavailable.
    """
    if not socket_path.exists():
        return None

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(30.0)
        sock.connect(str(socket_path))

        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))

        # Read response
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        sock.close()

        if not data:
            return None

        return json.loads(data.decode("utf-8").strip())

    except (ConnectionRefusedError, FileNotFoundError, OSError):
        # Socket exists but watcher is dead — stale socket
        return None


def query_via_socket(socket_path: Path, query: str, k: int = 10,
                     filters: dict | None = None,
                     vault_names: list[str] | None = None) -> dict | None:
    """Send a query to a running watcher's query server.

    Returns the response dict, or None if the socket is unavailable.
    """
    return request_via_socket(socket_path, {
        "action": "query",
        "query": query,
        "k": k,
        "filters": filters or {},
        "vault_names": vault_names,
    })
