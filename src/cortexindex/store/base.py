from __future__ import annotations

from typing import Protocol, Any, Sequence
import numpy as np
from ..models import Document, Chunk, SearchResult, SourceRef, OpenResult

class Store(Protocol):
    """Storage adapter interface.

    Implementations may use:
    - SQLite/libSQL only
    - SQLite + external vector store
    """

    def init(self) -> None:
        ...

    def upsert_document(self, doc: Document) -> None:
        ...

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        ...

    def delete_document(self, vault_id: str, rel_path: str) -> None:
        ...

    def upsert_embeddings(self, chunk_ids: Sequence[str], model_id: str, vectors: np.ndarray) -> None:
        ...

    def lexical_search(self, query: str, k: int, filters: dict[str, Any]) -> list[SearchResult]:
        ...

    def vector_search(self, query_vec: np.ndarray, k: int, filters: dict[str, Any]) -> list[SearchResult]:
        ...

    def open(self, source_ref: SourceRef) -> OpenResult:
        ...

    def status(self, vault_id: str) -> dict[str, Any]:
        ...

    def neighbors(self, vault_id: str, rel_path: str, depth: int = 1) -> dict[str, Any]:
        ...
