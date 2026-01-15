from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter

from .base import Extracted
from ..utils import parse_wikilinks, parse_tags

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

        meta: dict[str, Any] = {
            "frontmatter": fm,
            "wikilinks": links,
            "tags": tags,
        }
        return Extracted(text=text, metadata=meta)
