"""RAGtriever — local-only vault indexer + hybrid RAG retrieval system.

A powerful retrieval-augmented generation system for indexing Obsidian-compatible vaults
with semantic search, lexical search, and link-graph awareness. All data stays local.

Public API:
- VaultConfig
- Indexer
- Retriever
"""

from .config import VaultConfig
from .indexer.indexer import Indexer
from .retrieval.retriever import Retriever

__all__ = ["VaultConfig", "Indexer", "Retriever"]
