"""Data classes for parallel indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChunkData:
    """Chunk ready for embedding."""

    chunk_id: str
    doc_id: str
    vault_id: str
    anchor_type: str
    anchor_ref: str
    text: str
    text_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result from parallel extraction worker."""

    abs_path: Path
    rel_path: str
    doc_id: str
    vault_id: str
    file_type: str
    content_hash: str
    mtime: int
    size: int
    chunks: list[ChunkData] = field(default_factory=list)
    embedded_images: list[dict[str, Any]] = field(default_factory=list)
    image_references: list[dict[str, Any]] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)  # (target, link_type)
    error: str | None = None


@dataclass
class ImageTask:
    """Image to be processed in parallel."""

    parent_doc_id: str
    parent_path: str
    vault_id: str
    file_type: str
    image_data: dict[str, Any]
    task_type: str  # "embedded" | "reference"


@dataclass
class ScanStats:
    """Statistics from a scan operation."""

    files_scanned: int = 0
    files_indexed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    embeddings_created: int = 0
    images_processed: int = 0
    elapsed_seconds: float = 0.0
