from __future__ import annotations

import json
import sys
from typing import Any

from ..config import VaultConfig
from ..retrieval.retriever import Retriever
from .tools import tool_search, tool_open, tool_neighbors, tool_status

TOOL_MAP = {
    "vault.search": tool_search,
    "vault.open": tool_open,
    "vault.neighbors": tool_neighbors,
    "vault.status": tool_status,
}

def run_stdio_server(cfg: VaultConfig) -> None:
    """Minimal JSON-RPC-ish server over stdio.

    NOTE: MCP implementations vary; a coding agent should adapt this stub to the
    target MCP SDK/contract for the chosen host (Claude Code, Codex, Gemini, etc.).
    """
    retriever = Retriever(cfg)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            rid = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            if method not in TOOL_MAP:
                resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}
            else:
                result = TOOL_MAP[method](retriever, params)
                resp = {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": req.get("id") if isinstance(req, dict) else None,
                    "error": {"code": -32000, "message": str(e)}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
