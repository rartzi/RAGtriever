from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import Extracted

@dataclass
class PdfExtractor:
    supported_suffixes = (".pdf",)

    def extract(self, path: Path) -> Extracted:
        """Extract PDF text.

        TODO: Implement page-aware extraction using pdfplumber, returning text with page markers
        and metadata including page count and optionally per-page map.
        """
        # Placeholder minimal extraction; implement properly.
        try:
            import pdfplumber  # type: ignore
        except Exception as e:
            raise RuntimeError("pdfplumber required for PDF extraction") from e

        pages: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                t = page.extract_text() or ""
                pages.append(f"\n\n[[[PAGE {i}]]]\n{t}")

        text = "".join(pages).strip()
        meta: dict[str, Any] = {"page_count": len(pages)}
        return Extracted(text=text, metadata=meta)
