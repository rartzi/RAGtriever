"""Mneme — local-only vault indexer + hybrid RAG retrieval system.

A powerful retrieval-augmented generation system for indexing Obsidian-compatible vaults
with semantic search, lexical search, and link-graph awareness. All data stays local.

Public API:
- VaultConfig
- Indexer
- Retriever
"""


def __getattr__(name: str):
    """Lazy-load heavy modules to avoid importing torch/numpy on every CLI invocation."""
    if name == "VaultConfig":
        from .config import VaultConfig
        return VaultConfig
    if name == "Indexer":
        from .indexer.indexer import Indexer
        return Indexer
    if name == "Retriever":
        from .retrieval.retriever import Retriever
        return Retriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VaultConfig", "Indexer", "Retriever"]
