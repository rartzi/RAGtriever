from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .base import Chunked

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

@dataclass
class MarkdownChunker:
    """Chunk markdown content by headings.

    TODO:
    - Preserve code blocks atomically
    - Add max size + overlap controls
    - Compute line ranges for citations
    """

    def chunk(self, extracted_text: str, extracted_metadata: dict[str, Any]) -> list[Chunked]:
        text = extracted_text
        matches = list(HEADING_RE.finditer(text))
        if not matches:
            return [Chunked(anchor_type="md_heading", anchor_ref="ROOT", text=text.strip(), metadata=extracted_metadata)]

        chunks: list[Chunked] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = m.group(2).strip()
            body = text[start:end].strip()
            if body:
                chunks.append(Chunked(
                    anchor_type="md_heading",
                    anchor_ref=heading,
                    text=body,
                    metadata=dict(extracted_metadata, heading=heading),
                ))
        return chunks
