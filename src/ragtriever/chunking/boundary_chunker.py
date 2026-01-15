from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Chunked

@dataclass
class BoundaryMarkerChunker:
    """Chunk extracted text using explicit boundary markers.

    Markers example:
    - [[[PAGE N]]]
    - [[[SLIDE N]]]
    - [[[SHEET Name]]]

    TODO: implement robust parsing and attach numeric anchors.
    """
    marker_prefix: str  # e.g. "PAGE", "SLIDE", "SHEET"

    def chunk(self, extracted_text: str, extracted_metadata: dict[str, Any]) -> list[Chunked]:
        parts = extracted_text.split(f"[[[{self.marker_prefix} ")
        if len(parts) == 1:
            return [Chunked(anchor_type=self.marker_prefix.lower(), anchor_ref="0", text=extracted_text, metadata=extracted_metadata)]

        chunks: list[Chunked] = []
        for part in parts[1:]:
            # part like: "12]]]\n..."
            head, _, body = part.partition("]]]")
            anchor = head.strip()
            content = body.strip()
            if content:
                chunks.append(Chunked(
                    anchor_type=self.marker_prefix.lower(),
                    anchor_ref=anchor,
                    text=content,
                    metadata=dict(extracted_metadata, anchor=anchor),
                ))
        return chunks
