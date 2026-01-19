from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .base import Chunked

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

@dataclass
class MarkdownChunker:
    """Chunk markdown content by headings with configurable overlap.

    Supports:
    - Heading-based chunking with overlap between sections
    - Splitting large sections that exceed max_chunk_size
    - Preserving heading hierarchy in metadata even when split
    """

    overlap_chars: int = 200
    max_chunk_size: int = 2000
    preserve_heading_metadata: bool = True

    def chunk(self, extracted_text: str, extracted_metadata: dict[str, Any]) -> list[Chunked]:
        text = extracted_text
        matches = list(HEADING_RE.finditer(text))

        if not matches:
            # No headings - split by max_chunk_size if needed
            if len(text.strip()) > self.max_chunk_size:
                return self._split_large_text(text.strip(), "ROOT", extracted_metadata)
            return [Chunked(anchor_type="md_heading", anchor_ref="ROOT", text=text.strip(), metadata=extracted_metadata)]

        chunks: list[Chunked] = []
        sections = self._extract_sections(text, matches)

        for i, (heading, level, body) in enumerate(sections):
            # Check if section needs splitting
            if len(body) > self.max_chunk_size:
                sub_chunks = self._split_large_text(body, heading, extracted_metadata, level)
                chunks.extend(sub_chunks)
            else:
                # Add overlap from previous section
                prefix = ""
                if i > 0 and self.overlap_chars > 0:
                    prev_body = sections[i-1][2]
                    prefix = prev_body[-self.overlap_chars:].strip() + "\n\n"

                full_text = prefix + body

                metadata = dict(extracted_metadata)
                if self.preserve_heading_metadata:
                    metadata["heading"] = heading
                    metadata["level"] = level
                if prefix:
                    metadata["has_prefix_overlap"] = True

                if full_text.strip():
                    chunks.append(Chunked(
                        anchor_type="md_heading",
                        anchor_ref=heading,
                        text=full_text.strip(),
                        metadata=metadata,
                    ))

        return chunks

    def _extract_sections(self, text: str, matches: list[re.Match]) -> list[tuple[str, int, str]]:
        """Extract heading sections with level and body text."""
        sections = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            level = len(m.group(1))  # Count # characters
            heading = m.group(2).strip()
            body = text[start:end].strip()
            sections.append((heading, level, body))
        return sections

    def _split_large_text(
        self,
        text: str,
        heading: str,
        metadata: dict[str, Any],
        level: int | None = None
    ) -> list[Chunked]:
        """Split large text into chunks with overlap."""
        # First pass: collect chunk data
        chunk_data = []
        start = 0
        split_index = 0

        while start < len(text):
            end = start + self.max_chunk_size
            chunk_text = text[start:end]

            # Try to break at paragraph boundary
            if end < len(text):
                last_para = chunk_text.rfind('\n\n')
                if last_para > self.max_chunk_size * 0.7:  # At least 70% of max_size
                    end = start + last_para
                    chunk_text = text[start:end]

            chunk_data.append((chunk_text.strip(), split_index))
            start = end - self.overlap_chars  # Overlap with next chunk
            split_index += 1

        # Second pass: create chunks with complete metadata
        chunks = []
        total_splits = len(chunk_data)

        for chunk_text, split_idx in chunk_data:
            chunk_metadata = dict(metadata)
            if self.preserve_heading_metadata:
                chunk_metadata["heading"] = heading
                if level is not None:
                    chunk_metadata["level"] = level
                chunk_metadata["split_index"] = split_idx
                chunk_metadata["split_total"] = total_splits
                chunk_metadata["is_split"] = True

            chunks.append(Chunked(
                anchor_type="md_heading_split",
                anchor_ref=f"{heading}:{split_idx}",
                text=chunk_text,
                metadata=chunk_metadata,
            ))

        return chunks
