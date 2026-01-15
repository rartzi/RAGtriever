from __future__ import annotations

from typing import Any
from ..retrieval.retriever import Retriever
from ..models import SourceRef

def tool_search(retriever: Retriever, params: dict[str, Any]) -> dict[str, Any]:
    query = params.get("query", "")
    k = int(params.get("k", retriever.cfg.top_k))
    filters = params.get("filters") or {}
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

def tool_open(retriever: Retriever, params: dict[str, Any]) -> dict[str, Any]:
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

def tool_neighbors(retriever: Retriever, params: dict[str, Any]) -> dict[str, Any]:
    path_or_doc_id = params.get("path_or_doc_id")
    depth = int(params.get("depth", 1))
    vault_id = params.get("vault_id","")
    return retriever.neighbors(path_or_doc_id, vault_id=vault_id, depth=depth)

def tool_status(retriever: Retriever, params: dict[str, Any]) -> dict[str, Any]:
    vault_id = params.get("vault_id","")
    return retriever.status(vault_id=vault_id)
