from __future__ import annotations

import json
import sys

from ..config import VaultConfig, MultiVaultConfig
from ..retrieval.retriever import Retriever, MultiVaultRetriever
from .tools import tool_search, tool_open, tool_neighbors, tool_status, tool_list_vaults

TOOL_MAP = {
    "vault_search": tool_search,
    "vault_open": tool_open,
    "vault_neighbors": tool_neighbors,
    "vault_status": tool_status,
    "vault_list": tool_list_vaults,
}

def run_stdio_server(cfg: VaultConfig | MultiVaultConfig) -> None:
    """MCP server over stdio with full protocol support.

    Supports both single-vault (VaultConfig) and multi-vault (MultiVaultConfig) configurations.
    """
    if isinstance(cfg, MultiVaultConfig):
        retriever: Retriever | MultiVaultRetriever = MultiVaultRetriever(cfg)
    else:
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

            # Handle notifications (no id = no response expected)
            if rid is None:
                # Notifications like "notifications/initialized" don't get responses
                continue

            # Handle MCP protocol methods
            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "mneme",
                            "version": "3.0.0"
                        }
                    }
                }
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "tools": [
                            {
                                "name": "vault_search",
                                "description": "Search across indexed vaults using hybrid semantic and lexical search",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query"},
                                        "k": {"type": "integer", "description": "Number of results (default: 10)"}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "vault_neighbors",
                                "description": "Find wikilink neighbors and backlinks for a document",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "rel_path": {"type": "string", "description": "Relative path to file"}
                                    },
                                    "required": ["rel_path"]
                                }
                            },
                            {
                                "name": "vault_open",
                                "description": "Open a file in Obsidian",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "rel_path": {"type": "string", "description": "Relative path to file"}
                                    },
                                    "required": ["rel_path"]
                                }
                            },
                            {
                                "name": "vault_status",
                                "description": "Get indexing status and vault configuration",
                                "inputSchema": {"type": "object", "properties": {}}
                            },
                            {
                                "name": "vault_list",
                                "description": "List configured vaults",
                                "inputSchema": {"type": "object", "properties": {}}
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                # MCP tools/call method - extract tool name and arguments
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                if tool_name in TOOL_MAP:
                    try:
                        result = TOOL_MAP[tool_name](retriever, tool_args)
                        # MCP expects content array format for tool results
                        resp = {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {
                                "content": [
                                    {"type": "text", "text": json.dumps(result, indent=2)}
                                ]
                            }
                        }
                    except Exception as tool_error:
                        resp = {
                            "jsonrpc": "2.0",
                            "id": rid,
                            "result": {
                                "content": [
                                    {"type": "text", "text": f"Error: {str(tool_error)}"}
                                ],
                                "isError": True
                            }
                        }
                else:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "result": {
                            "content": [
                                {"type": "text", "text": f"Unknown tool: {tool_name}"}
                            ],
                            "isError": True
                        }
                    }
            elif method in TOOL_MAP:
                # Direct tool call (legacy support)
                result = TOOL_MAP[method](retriever, params)
                resp = {"jsonrpc": "2.0", "id": rid, "result": result}
            else:
                resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": req.get("id") if isinstance(req, dict) else None,
                    "error": {"code": -32000, "message": str(e)}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
