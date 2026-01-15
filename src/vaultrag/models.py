from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass(frozen=True)
class SourceRef:
    """Stable reference to a source location for citations and `open()`.

    anchor_type examples:
    - md_heading | md_block | pdf_page | ppt_slide | xls_range | image
    """
    vault_id: str
    rel_path: str
    file_type: str
    anchor_type: str
    anchor_ref: str
    locator: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    snippet: str
    source_ref: SourceRef
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class OpenResult:
    content: str
    source_ref: SourceRef
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Document:
    doc_id: str
    vault_id: str
    rel_path: str
    file_type: str
    mtime: int
    size: int
    content_hash: str
    deleted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    vault_id: str
    anchor_type: str
    anchor_ref: str
    text: str
    text_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
