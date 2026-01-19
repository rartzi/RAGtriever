from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..config import VaultConfig, MultiVaultConfig
from ..hashing import blake2b_hex
from ..models import SearchResult, SourceRef, OpenResult
from ..embeddings.sentence_transformers import SentenceTransformersEmbedder
from ..embeddings.ollama import OllamaEmbedder
from ..store.libsql_store import LibSqlStore
from .hybrid import HybridRanker
from .reranker import CrossEncoderReranker, CROSS_ENCODER_AVAILABLE

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

        # Initialize reranker if enabled
        self.reranker: Optional[CrossEncoderReranker] = None
        if self.cfg.use_rerank:
            if not CROSS_ENCODER_AVAILABLE:
                import warnings
                warnings.warn(
                    "use_rerank=true but sentence-transformers not available. "
                    "Install with: pip install sentence-transformers"
                )
            else:
                self.reranker = CrossEncoderReranker(
                    model_name=self.cfg.rerank_model,
                    device=self.cfg.rerank_device
                )
                print(f"✓ Reranker initialized: {self.cfg.rerank_model}")

    def search(self, query: str, k: int | None = None, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        k = k or self.cfg.top_k
        filters = filters or {}
        qv = self.embedder.embed_query(query)

        vec_hits = self.store.vector_search(qv, k=self.cfg.k_vec, filters=filters)
        lex_hits = self.store.lexical_search(query, k=self.cfg.k_lex, filters=filters)

        # Merge with RRF
        merged = self.ranker.merge(vec_hits, lex_hits, k=k)

        # Rerank if enabled
        if self.reranker:
            merged = self.reranker.rerank(query, merged, top_k=k)

        return merged

    def open(self, source_ref: SourceRef) -> OpenResult:
        return self.store.open(source_ref)

    def neighbors(self, path_or_doc_id: str, vault_id: str = "", depth: int = 1) -> dict[str, Any]:
        # For v1 skeleton, interpret as rel_path
        return self.store.neighbors(vault_id=vault_id, rel_path=path_or_doc_id, depth=depth)

    def status(self, vault_id: str) -> dict[str, Any]:
        return self.store.status(vault_id=vault_id)


class MultiVaultRetriever:
    """Retriever supporting search across multiple vaults.

    Uses a shared store and can search all vaults or specific ones.
    """

    def __init__(self, cfg: MultiVaultConfig) -> None:
        self.cfg = cfg

        # Shared store
        db_path = cfg.index_dir / "vaultrag.sqlite"
        self.store = LibSqlStore(db_path)
        self.store.init()

        # Shared embedder
        if cfg.embedding_provider == "sentence_transformers":
            self.embedder = SentenceTransformersEmbedder(
                model_id=cfg.embedding_model,
                device=cfg.embedding_device,
                batch_size=cfg.embedding_batch_size,
            )
        elif cfg.embedding_provider == "ollama":
            self.embedder = OllamaEmbedder(model_id=cfg.embedding_model)
        else:
            raise ValueError(f"Unknown embedding provider: {cfg.embedding_provider}")

        self.ranker = HybridRanker()

        # Initialize reranker if enabled
        self.reranker: Optional[CrossEncoderReranker] = None
        if cfg.use_rerank:
            if not CROSS_ENCODER_AVAILABLE:
                import warnings
                warnings.warn(
                    "use_rerank=true but sentence-transformers not available. "
                    "Install with: pip install sentence-transformers"
                )
            else:
                self.reranker = CrossEncoderReranker(
                    model_name=cfg.rerank_model,
                    device=cfg.rerank_device
                )
                print(f"✓ Reranker initialized: {cfg.rerank_model}")

        # Build vault_id lookup
        self._vault_ids: dict[str, str] = {}  # name -> vault_id
        for vault in cfg.vaults:
            if vault.enabled:
                vault_id = blake2b_hex(str(vault.root).encode("utf-8"))[:12]
                self._vault_ids[vault.name] = vault_id

    def get_vault_names(self) -> list[str]:
        """Get list of configured vault names."""
        return list(self._vault_ids.keys())

    def get_vault_ids(self, vault_names: list[str] | None = None) -> list[str]:
        """Get vault_ids for specified vault names, or all if None."""
        if vault_names is None:
            return list(self._vault_ids.values())

        return [self._vault_ids[name] for name in vault_names if name in self._vault_ids]

    def search(
        self,
        query: str,
        k: int | None = None,
        vault_names: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search across specified vaults or all if None.

        Args:
            query: Search query string
            k: Number of results to return (default: cfg.top_k)
            vault_names: List of vault names to search, or None for all
            filters: Additional filters (e.g., path_prefix)

        Returns:
            List of SearchResult objects, merged from all searched vaults
        """
        k = k or self.cfg.top_k
        filters = filters or {}

        # Resolve vault_ids from names
        vault_ids = self.get_vault_ids(vault_names)
        if vault_ids:
            filters["vault_ids"] = vault_ids

        # Embed query
        qv = self.embedder.embed_query(query)

        # Search
        vec_hits = self.store.vector_search(qv, k=self.cfg.k_vec, filters=filters)
        lex_hits = self.store.lexical_search(query, k=self.cfg.k_lex, filters=filters)

        # Merge with RRF
        merged = self.ranker.merge(vec_hits, lex_hits, k=k)

        # Rerank if enabled
        if self.reranker:
            merged = self.reranker.rerank(query, merged, top_k=k)

        return merged

    def open(self, source_ref: SourceRef) -> OpenResult:
        """Open a source reference."""
        return self.store.open(source_ref)

    def neighbors(self, path_or_doc_id: str, vault_id: str = "", depth: int = 1) -> dict[str, Any]:
        """Get neighbors (links) for a document."""
        return self.store.neighbors(vault_id=vault_id, rel_path=path_or_doc_id, depth=depth)

    def status(self, vault_names: list[str] | None = None) -> dict[str, Any]:
        """Get status for specified vaults or all.

        Args:
            vault_names: List of vault names, or None for all

        Returns:
            Combined status dict with per-vault stats
        """
        vault_ids = self.get_vault_ids(vault_names)

        if not vault_ids:
            return {"error": "No vaults found"}

        # Aggregate status across vaults
        total_files = 0
        total_chunks = 0
        vault_stats = []

        for name, vault_id in self._vault_ids.items():
            if vault_ids and vault_id not in vault_ids:
                continue

            vault_status = self.store.status(vault_id=vault_id)
            total_files += vault_status.get("indexed_files", 0)
            total_chunks += vault_status.get("indexed_chunks", 0)
            vault_stats.append({
                "name": name,
                "vault_id": vault_id,
                **vault_status
            })

        return {
            "total_indexed_files": total_files,
            "total_indexed_chunks": total_chunks,
            "vaults": vault_stats,
        }
