from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any

@dataclass(frozen=True)
class Extracted:
    text: str
    metadata: dict[str, Any]

class Extractor(Protocol):
    supported_suffixes: tuple[str, ...]

    def extract(self, path: Path) -> Extracted:
        ...

class ExtractorRegistry:
    def __init__(self) -> None:
        self._by_suffix: dict[str, Extractor] = {}

    def register(self, extractor: Extractor) -> None:
        for s in extractor.supported_suffixes:
            self._by_suffix[s.lower()] = extractor

    def get(self, path: Path) -> Extractor | None:
        return self._by_suffix.get(path.suffix.lower())
