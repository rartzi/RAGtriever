from __future__ import annotations

from typing import Any
from ..retrieval.retriever import Retriever, MultiVaultRetriever
from ..models import SourceRef
from ..hashing import blake2b_hex

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


def _get_vault_id(retriever: RetrieverType) -> str:
    """Derive vault_id for a single-vault retriever."""
    return blake2b_hex(str(retriever.cfg.vault_root).encode("utf-8"))[:12]


def tool_list_docs(retriever: RetrieverType, params: dict[str, Any]) -> dict[str, Any]:
    """List indexed documents in the vault.

    Params:
        path: Optional path prefix filter (e.g. "projects/")
        vaults: List of vault names (multi-vault only, optional)
    """
    path_prefix = params.get("path", "")

    if isinstance(retriever, MultiVaultRetriever):
        vault_names = params.get("vaults")
        vault_ids = retriever.get_vault_ids(vault_names)
        all_files: list[str] = []
        for vid in vault_ids:
            if path_prefix:
                all_files.extend(retriever.store.get_files_under_path(vid, path_prefix))
            else:
                all_files.extend(sorted(retriever.store.get_indexed_files(vid)))
        files = sorted(set(all_files))
    else:
        vault_id = _get_vault_id(retriever)
        if path_prefix:
            files = sorted(retriever.store.get_files_under_path(vault_id, path_prefix))
        else:
            files = sorted(retriever.store.get_indexed_files(vault_id))

    return {"files": files, "count": len(files), "path_filter": path_prefix or None}


def tool_text_search(retriever: RetrieverType, params: dict[str, Any]) -> dict[str, Any]:
    """Search vault using lexical (BM25) search only.

    Bypasses semantic search, boosts, and reranking for exact phrase matching.

    Params:
        query: Search query string (required)
        k: Number of results (default 20)
        path: Optional path prefix filter
        vaults: List of vault names (multi-vault only, optional)
    """
    query = params.get("query", "")
    k = int(params.get("k", 20))
    path_prefix = params.get("path", "")

    filters: dict[str, Any] = {}
    if path_prefix:
        filters["path_prefix"] = path_prefix

    if isinstance(retriever, MultiVaultRetriever):
        vault_names = params.get("vaults")
        vault_ids = retriever.get_vault_ids(vault_names)
        filters["vault_ids"] = vault_ids
    else:
        filters["vault_id"] = _get_vault_id(retriever)

    results = retriever.store.lexical_search(query, k, filters)

    return {"results": [
        {
            "chunk_id": r.chunk_id,
            "score": r.score,
            "snippet": r.snippet,
            "source_ref": r.source_ref.__dict__,
            "metadata": r.metadata,
        } for r in results
    ]}


def tool_backlinks(retriever: RetrieverType, params: dict[str, Any]) -> dict[str, Any]:
    """Get backlink counts for documents.

    Params:
        paths: Optional list of rel_paths to check
        limit: Max results when listing all (default 20)
    """
    paths = params.get("paths")
    limit = int(params.get("limit", 20))

    counts = retriever.store.get_backlink_counts(doc_ids=paths)

    # Map doc_id (hash) back to rel_path for readability
    # Query documents table for the mapping
    conn = retriever.store._get_conn()
    if counts:
        placeholders = ",".join("?" * len(counts))
        rows = conn.execute(
            f"SELECT doc_id, rel_path FROM documents WHERE doc_id IN ({placeholders}) AND deleted=0",
            list(counts.keys()),
        ).fetchall()
        id_to_path = {r["doc_id"]: r["rel_path"] for r in rows}
    else:
        id_to_path = {}

    # Build result: rel_path -> count, sorted by count desc
    backlinks = {
        id_to_path.get(did, did): count
        for did, count in counts.items()
    }
    sorted_backlinks = dict(
        sorted(backlinks.items(), key=lambda x: x[1], reverse=True)[:limit]
    )

    return {"backlinks": sorted_backlinks, "count": len(sorted_backlinks)}
