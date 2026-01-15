from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class Chunked:
    anchor_type: str
    anchor_ref: str
    text: str
    metadata: dict[str, Any]

class Chunker(Protocol):
    def chunk(self, extracted_text: str, extracted_metadata: dict[str, Any]) -> list[Chunked]:
        ...

class ChunkerRegistry:
    def __init__(self) -> None:
        self._by_type: dict[str, Chunker] = {}

    def register(self, file_type: str, chunker: Chunker) -> None:
        self._by_type[file_type] = chunker

    def get(self, file_type: str) -> Chunker | None:
        return self._by_type.get(file_type)
