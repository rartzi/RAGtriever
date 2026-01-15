"""VaultRAG Local — local-only vault indexer + retriever.

This package is a **skeleton** intended to be completed by a coding agent.

Public API:
- VaultConfig
- Indexer
- Retriever
"""

from .config import VaultConfig
from .indexer.indexer import Indexer
from .retrieval.retriever import Retriever

__all__ = ["VaultConfig", "Indexer", "Retriever"]
