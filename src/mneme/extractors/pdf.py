from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

from .base import Extracted

logger = logging.getLogger(__name__)


@dataclass
class PdfExtractor:
    supported_suffixes = (".pdf",)
    min_image_width: int = 100
    min_image_height: int = 100

    def extract(self, path: Path) -> Extracted:
        """Extract PDF text and embedded images.

        Returns text with page markers and metadata including:
        - page_count: number of pages
        - embedded_images: list of image metadata dicts (page_num, bbox, source_pdf)
        - source_pdf: path to PDF file for later image extraction
        """
        try:
            import pdfplumber  # type: ignore
        except Exception as e:
            raise RuntimeError("pdfplumber required for PDF extraction") from e

        pages: list[str] = []
        embedded_images: list[dict[str, Any]] = []

        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                # Extract text
                t = page.extract_text() or ""
                pages.append(f"\n\n[[[PAGE {i}]]]\n{t}")

                # Extract images from page
                page_images = self._extract_images_from_page(page, page_num=i)
                embedded_images.extend(page_images)

        text = "".join(pages).strip()
        meta: dict[str, Any] = {
            "page_count": len(pages),
            "embedded_images": embedded_images,
            "source_pdf": str(path),  # Store path for later image extraction
        }
        return Extracted(text=text, metadata=meta)

    def _extract_images_from_page(self, page: Any, page_num: int) -> list[dict[str, Any]]:
        """Extract images from a PDF page.

        Returns list of image metadata:
        - page_num: page number
        - bbox: bounding box (x0, y0, x1, y1)
        - width, height: dimensions
        - image_bytes: raw image data
        """
        images: list[dict[str, Any]] = []

        try:
            for img in page.images:
                width = int(img.get("width", 0))
                height = int(img.get("height", 0))

                # Filter by size: skip tiny images (< min_width x min_height)
                if width < self.min_image_width or height < self.min_image_height:
                    continue

                # Extract image data
                # pdfplumber provides x0, y0, x1, y1 coordinates
                bbox = {
                    "x0": float(img.get("x0", 0)),
                    "y0": float(img.get("y0", 0)),
                    "x1": float(img.get("x1", 0)),
                    "y1": float(img.get("y1", 0)),
                }

                images.append({
                    "page_num": page_num,
                    "bbox": bbox,
                    "width": width,
                    "height": height,
                    # Store reference to extract actual image bytes later in indexer
                    # We'll extract the actual image bytes in the indexer when needed
                })

        except Exception as e:
            logger.warning(f"Failed to extract images from page {page_num}: {e}")

        return images
