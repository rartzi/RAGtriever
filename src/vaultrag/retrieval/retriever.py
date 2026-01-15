from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..config import VaultConfig
from ..models import SearchResult, SourceRef, OpenResult
from ..embeddings.sentence_transformers import SentenceTransformersEmbedder
from ..embeddings.ollama import OllamaEmbedder
from ..store.libsql_store import LibSqlStore
from .hybrid import HybridRanker

@dataclass
class Retriever:
    cfg: VaultConfig

    def __post_init__(self) -> None:
        # Store: SQLite file under index_dir
        db_path = self.cfg.index_dir / "vaultrag.sqlite"
        self.store = LibSqlStore(db_path)
        self.store.init()

        # Embedder selection
        if self.cfg.embedding_provider == "sentence_transformers":
            self.embedder = SentenceTransformersEmbedder(
                model_id=self.cfg.embedding_model,
                device=self.cfg.embedding_device,
                batch_size=self.cfg.embedding_batch_size,
            )
        elif self.cfg.embedding_provider == "ollama":
            self.embedder = OllamaEmbedder(model_id=self.cfg.embedding_model)
        else:
            raise ValueError(f"Unknown embedding provider: {self.cfg.embedding_provider}")

        self.ranker = HybridRanker()

    def search(self, query: str, k: int | None = None, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        k = k or self.cfg.top_k
        filters = filters or {}
        qv = self.embedder.embed_query(query)

        vec_hits = self.store.vector_search(qv, k=self.cfg.k_vec, filters=filters)
        lex_hits = self.store.lexical_search(query, k=self.cfg.k_lex, filters=filters)

        return self.ranker.merge(vec_hits, lex_hits, k=k)

    def open(self, source_ref: SourceRef) -> OpenResult:
        return self.store.open(source_ref)

    def neighbors(self, path_or_doc_id: str, vault_id: str = "", depth: int = 1) -> dict[str, Any]:
        # For v1 skeleton, interpret as rel_path
        return self.store.neighbors(vault_id=vault_id, rel_path=path_or_doc_id, depth=depth)

    def status(self, vault_id: str) -> dict[str, Any]:
        return self.store.status(vault_id=vault_id)
