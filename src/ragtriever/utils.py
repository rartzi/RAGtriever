from __future__ import annotations

import re
from pathlib import Path

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(#[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_/-]+)")

def parse_wikilinks(text: str) -> list[str]:
    return [m.group(1).strip() for m in WIKILINK_RE.finditer(text)]

def parse_tags(text: str) -> list[str]:
    return [m.group(1) for m in TAG_RE.finditer(text)]

def safe_read_text(path: Path, max_bytes: int = 10_000_000) -> str:
    b = path.read_bytes()
    if len(b) > max_bytes:
        raise ValueError(f"File too large for text read: {path} ({len(b)} bytes)")
    # naive decode; agent may improve with chardet, etc.
    return b.decode("utf-8", errors="replace")
