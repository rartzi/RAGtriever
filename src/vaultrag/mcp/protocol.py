from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class JsonRpcRequest:
    jsonrpc: str
    id: Any
    method: str
    params: dict[str, Any] | None = None

@dataclass
class JsonRpcResponse:
    jsonrpc: str = "2.0"
    id: Any = None
    result: Any = None
    error: Any = None
