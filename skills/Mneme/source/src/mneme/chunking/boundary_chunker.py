from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Chunked

@dataclass
class BoundaryMarkerChunker:
    """Chunk extracted text using explicit boundary markers with overlap.

    Markers example:
    - [[[PAGE N]]]
    - [[[SLIDE N]]]
    - [[[SHEET Name]]]

    Supports overlap between pages/slides for context preservation.
    """
    marker_prefix: str  # e.g. "PAGE", "SLIDE", "SHEET"
    overlap_chars: int = 200

    def chunk(self, extracted_text: str, extracted_metadata: dict[str, Any]) -> list[Chunked]:
        parts = extracted_text.split(f"[[[{self.marker_prefix} ")
        if len(parts) == 1:
            return [Chunked(anchor_type=self.marker_prefix.lower(), anchor_ref="0", text=extracted_text, metadata=extracted_metadata)]

        # First pass: extract all sections
        sections = []
        for part in parts[1:]:
            # part like: "12]]]\n..."
            head, _, body = part.partition("]]]")
            anchor = head.strip()
            content = body.strip()
            if content:
                sections.append((anchor, content))

        # Second pass: add overlap and create chunks
        chunks: list[Chunked] = []
        for i, (anchor, content) in enumerate(sections):
            # Add overlap from previous section
            prefix = ""
            if i > 0 and self.overlap_chars > 0:
                prev_content = sections[i-1][1]
                prefix = prev_content[-self.overlap_chars:].strip() + "\n\n"

            full_text = prefix + content

            metadata = dict(extracted_metadata, anchor=anchor)
            if prefix:
                metadata["has_prefix_overlap"] = True

            chunks.append(Chunked(
                anchor_type=self.marker_prefix.lower(),
                anchor_ref=anchor,
                text=full_text.strip(),
                metadata=metadata,
            ))

        return chunks
