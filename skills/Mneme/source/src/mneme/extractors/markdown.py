from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import logging

import frontmatter

from .base import Extracted
from ..utils import parse_wikilinks, parse_tags

logger = logging.getLogger(__name__)


@dataclass
class MarkdownExtractor:
    supported_suffixes = (".md",)

    def extract(self, path: Path) -> Extracted:
        raw = path.read_text(encoding="utf-8", errors="replace")
        post = frontmatter.loads(raw)
        text = post.content
        fm = dict(post.metadata or {})

        links = parse_wikilinks(raw)
        tags_inline = parse_tags(raw)
        tags = sorted(set(tags_inline + (fm.get("tags", []) if isinstance(fm.get("tags"), list) else [])))

        # Parse image references
        image_refs = self._parse_image_references(raw, path)

        meta: dict[str, Any] = {
            "frontmatter": fm,
            "wikilinks": links,
            "tags": tags,
            "image_references": image_refs,
        }
        return Extracted(text=text, metadata=meta)

    def _parse_image_references(self, raw: str, md_path: Path) -> list[dict[str, Any]]:
        """Parse markdown and wikilink image references.

        Supports:
        - ![alt](path/to/image.png)
        - ![[image.png]]
        - ![[image.png|caption]]

        Returns list of dicts with:
        - alt_text: alt text or caption
        - rel_path: relative path as written in markdown
        - abs_path: resolved absolute path (if exists)
        """
        image_refs: list[dict[str, Any]] = []
        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

        # Markdown syntax: ![alt](path)
        md_pattern = r"!\[([^\]]*)\]\(([^\)]+)\)"
        for match in re.finditer(md_pattern, raw):
            alt_text = match.group(1)
            rel_path = match.group(2)

            # Resolve path relative to markdown file
            abs_path = (md_path.parent / rel_path).resolve()

            # Only include if file exists and is an image
            if abs_path.exists() and abs_path.suffix.lower() in valid_extensions:
                image_refs.append({
                    "alt_text": alt_text,
                    "rel_path": rel_path,
                    "abs_path": str(abs_path),
                })

        # Wikilink syntax: ![[image.png]] or ![[image.png|caption]]
        wiki_pattern = r"!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]"
        for match in re.finditer(wiki_pattern, raw):
            image_name = match.group(1).strip()
            caption = match.group(2).strip() if match.group(2) else ""

            # Try resolving relative to markdown file first
            abs_path = (md_path.parent / image_name).resolve()

            # Only include if file exists and is an image
            if abs_path.exists() and abs_path.suffix.lower() in valid_extensions:
                image_refs.append({
                    "alt_text": caption,
                    "rel_path": image_name,
                    "abs_path": str(abs_path),
                })
            else:
                logger.debug(f"Image reference not found: {image_name} (from {md_path.name})")

        return image_refs
