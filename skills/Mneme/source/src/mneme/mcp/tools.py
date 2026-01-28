from __future__ import annotations

from typing import Any
from ..retrieval.retriever import Retriever, MultiVaultRetriever
from ..models import SourceRef

# Type alias for retriever (single or multi-vault)
RetrieverType = Retriever | MultiVaultRetriever

def tool_search(retriever: RetrieverType, params: dict[str, Any]) -> dict[str, Any]:
    """Search vault(s) for content.

    Params:
        query: Search query string
        k: Number of results (optional)
        filters: Optional filter dict
        vaults: List of vault names to search (multi-vault only, optional)
    """
    query = params.get("query", "")
    k = int(params.get("k", retriever.cfg.top_k))
    filters = params.get("filters") or {}
    vault_names = params.get("vaults")  # For multi-vault

    if isinstance(retriever, MultiVaultRetriever):
        results = retriever.search(query, k=k, vault_names=vault_names, filters=filters)
    else:
        results = retriever.search(query, k=k, filters=filters)

    return {"results": [
        {
            "chunk_id": r.chunk_id,
            "score": r.score,
            "snippet": r.snippet,
            "source_ref": r.source_ref.__dict__,
            "metadata": r.metadata,
        } for r in results
    ]}

def tool_open(retriever: RetrieverType, params: dict[str, Any]) -> dict[str, Any]:
    """Open a source reference to get full content."""
    sr = params.get("source_ref") or {}
    source_ref = SourceRef(
        vault_id=sr.get("vault_id",""),
        rel_path=sr["rel_path"],
        file_type=sr["file_type"],
        anchor_type=sr["anchor_type"],
        anchor_ref=sr["anchor_ref"],
        locator=sr.get("locator") or {},
    )
    opened = retriever.open(source_ref)
    return {"content": opened.content, "source_ref": opened.source_ref.__dict__, "metadata": opened.metadata}

def tool_neighbors(retriever: RetrieverType, params: dict[str, Any]) -> dict[str, Any]:
    """Get link neighbors for a document."""
    path_or_doc_id = params.get("path_or_doc_id")
    depth = int(params.get("depth", 1))
    vault_id = params.get("vault_id","")
    return retriever.neighbors(path_or_doc_id, vault_id=vault_id, depth=depth)

def tool_status(retriever: RetrieverType, params: dict[str, Any]) -> dict[str, Any]:
    """Get indexing status.

    Params:
        vault_id: Vault ID (single-vault only)
        vaults: List of vault names (multi-vault only, optional)
    """
    if isinstance(retriever, MultiVaultRetriever):
        vault_names = params.get("vaults")
        return retriever.status(vault_names=vault_names)
    else:
        vault_id = params.get("vault_id","")
        return retriever.status(vault_id=vault_id)

def tool_list_vaults(retriever: RetrieverType, params: dict[str, Any]) -> dict[str, Any]:
    """List configured vaults (multi-vault only)."""
    if isinstance(retriever, MultiVaultRetriever):
        return {"vaults": retriever.get_vault_names()}
    else:
        return {"error": "Single-vault configuration - no vault list available"}
