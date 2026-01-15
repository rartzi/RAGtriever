from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import json
import os

from .base import Extracted

logger = logging.getLogger(__name__)


@dataclass
class TesseractImageExtractor:
    supported_suffixes = (".png", ".jpg", ".jpeg", ".webp")
    ocr_mode: str = "off"  # off|auto|on

    def extract(self, path: Path) -> Extracted:
        """Extract image text via OCR and/or captions.

        OCR is controlled by ocr_mode:
        - "off": no OCR, returns empty text
        - "on": always run OCR
        - "auto": run OCR (currently same as "on")
        """
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError("Pillow required for image extraction") from e

        with Image.open(path) as img:
            w, h = img.size

            # Determine if we should run OCR
            should_ocr = self.ocr_mode in ("on", "auto")
            ocr_text = ""

            if should_ocr:
                ocr_text = self._run_ocr(img)

        meta: dict[str, Any] = {
            "width": w,
            "height": h,
            "ocr_text": ocr_text,
            "ocr_mode": self.ocr_mode,
        }
        return Extracted(text=ocr_text, metadata=meta)

    def _run_ocr(self, img: Any) -> str:
        """Run OCR on image using pytesseract.

        Returns empty string if OCR fails or tesseract is not installed.
        """
        try:
            import pytesseract  # type: ignore
        except ImportError:
            logger.warning(
                "pytesseract not installed. Install with: pip install pytesseract. "
                "Also ensure tesseract-ocr is installed on your system."
            )
            return ""

        try:
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed: {e}. Ensure tesseract-ocr is installed on your system.")
            return ""


# Keep backward compatibility alias
ImageExtractor = TesseractImageExtractor


@dataclass
class GeminiImageExtractor:
    """Image extractor using Google Gemini vision model for rich understanding.

    Uses gemini-2.0-flash model to analyze images and extract:
    - Detailed description of the image content
    - Any text visible in the image (OCR)
    - Key topics/themes
    - Entities mentioned (people, organizations, products)
    - Image type classification (screenshot, diagram, photo, etc.)
    """
    supported_suffixes = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    api_key: str | None = None
    model: str = "gemini-2.0-flash"

    def __post_init__(self) -> None:
        # Try to get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def extract(self, path: Path) -> Extracted:
        """Extract image content using Gemini vision model."""
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            raise RuntimeError("Pillow required for image extraction")

        # Get basic image info
        with Image.open(path) as img:
            w, h = img.size
            img_format = img.format or path.suffix.lstrip(".").upper()

        # Read image bytes for Gemini
        image_bytes = path.read_bytes()

        # Analyze with Gemini
        analysis = self._analyze_with_gemini(image_bytes, path.suffix)

        # Build the text content for indexing
        text_parts = []

        if analysis.get("description"):
            text_parts.append(f"Description: {analysis['description']}")

        if analysis.get("visible_text"):
            text_parts.append(f"Visible text: {analysis['visible_text']}")

        if analysis.get("topics"):
            text_parts.append(f"Topics: {', '.join(analysis['topics'])}")

        if analysis.get("entities"):
            text_parts.append(f"Entities: {', '.join(analysis['entities'])}")

        text = "\n\n".join(text_parts)

        meta: dict[str, Any] = {
            "width": w,
            "height": h,
            "format": img_format,
            "image_type": analysis.get("image_type", "unknown"),
            "description": analysis.get("description", ""),
            "visible_text": analysis.get("visible_text", ""),
            "topics": analysis.get("topics", []),
            "entities": analysis.get("entities", []),
            "analysis_model": self.model,
            "analysis_provider": "gemini",
        }

        return Extracted(text=text, metadata=meta)

    def _analyze_with_gemini(self, image_bytes: bytes, suffix: str) -> dict[str, Any]:
        """Call Gemini API to analyze the image."""
        if not self.api_key:
            logger.warning(
                "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable. "
                "Returning empty analysis."
            )
            return {}

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError:
            logger.warning(
                "google-genai not installed. Install with: pip install google-genai"
            )
            return {}

        try:
            client = genai.Client(api_key=self.api_key)

            # Determine mime type
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }
            mime_type = mime_map.get(suffix.lower(), "image/png")

            # Create the prompt for structured analysis
            prompt = """Analyze this image and provide a structured response in JSON format with these fields:

1. "description": A detailed description of what the image shows (2-3 sentences)
2. "visible_text": Any text visible in the image, transcribed accurately
3. "image_type": One of: screenshot, diagram, flowchart, photo, presentation_slide, document, chart, infographic, logo, ui_mockup, other
4. "topics": List of 3-5 key topics or themes in the image
5. "entities": List of any named entities (people, companies, products, technologies) mentioned or shown

Respond ONLY with valid JSON, no markdown formatting or extra text."""

            response = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
            )

            # Parse the JSON response
            response_text = response.text.strip()
            # Remove markdown code block if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json) and last line (```)
                response_text = "\n".join(lines[1:-1])

            result = json.loads(response_text)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini response as JSON: {e}")
            # Try to extract what we can from non-JSON response
            return {"description": response.text[:500] if 'response' in dir() else ""}
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}")
            return {}


@dataclass
class VertexAIImageExtractor:
    """Image extractor using Google Vertex AI vision model with service account authentication.

    Uses Vertex AI with JSON credential files to analyze images and extract:
    - Detailed description of the image content
    - Any text visible in the image (OCR)
    - Key topics/themes
    - Entities mentioned (people, organizations, products)
    - Image type classification (screenshot, diagram, photo, etc.)
    """
    supported_suffixes = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    project_id: str | None = None
    location: str = "us-central1"
    credentials_file: str | None = None
    model: str = "gemini-2.0-flash-exp"

    def __post_init__(self) -> None:
        # Try to get values from environment if not provided
        if self.project_id is None:
            self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        if self.credentials_file is None:
            self.credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    def extract(self, path: Path) -> Extracted:
        """Extract image content using Vertex AI vision model."""
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            raise RuntimeError("Pillow required for image extraction")

        # Get basic image info
        with Image.open(path) as img:
            w, h = img.size
            img_format = img.format or path.suffix.lstrip(".").upper()

        # Read image bytes for Vertex AI
        image_bytes = path.read_bytes()

        # Analyze with Vertex AI
        analysis = self._analyze_with_vertex_ai(image_bytes, path.suffix)

        # Build the text content for indexing
        text_parts = []

        if analysis.get("description"):
            text_parts.append(f"Description: {analysis['description']}")

        if analysis.get("visible_text"):
            text_parts.append(f"Visible text: {analysis['visible_text']}")

        if analysis.get("topics"):
            text_parts.append(f"Topics: {', '.join(analysis['topics'])}")

        if analysis.get("entities"):
            text_parts.append(f"Entities: {', '.join(analysis['entities'])}")

        text = "\n\n".join(text_parts)

        meta: dict[str, Any] = {
            "width": w,
            "height": h,
            "format": img_format,
            "image_type": analysis.get("image_type", "unknown"),
            "description": analysis.get("description", ""),
            "visible_text": analysis.get("visible_text", ""),
            "topics": analysis.get("topics", []),
            "entities": analysis.get("entities", []),
            "analysis_model": self.model,
            "analysis_provider": "vertex_ai",
        }

        return Extracted(text=text, metadata=meta)

    def _analyze_with_vertex_ai(self, image_bytes: bytes, suffix: str) -> dict[str, Any]:
        """Call Vertex AI API to analyze the image using service account credentials."""
        if not self.project_id:
            logger.warning(
                "No Google Cloud project ID found. Set project_id in config or GOOGLE_CLOUD_PROJECT environment variable. "
                "Returning empty analysis."
            )
            return {}

        if not self.credentials_file:
            logger.warning(
                "No credentials file found. Set credentials_file in config or GOOGLE_APPLICATION_CREDENTIALS environment variable. "
                "Returning empty analysis."
            )
            return {}

        try:
            import vertexai  # type: ignore
            from vertexai.generative_models import GenerativeModel, Part  # type: ignore
            from google.oauth2 import service_account  # type: ignore
        except ImportError:
            logger.warning(
                "vertexai or google-auth not installed. Install with: pip install google-cloud-aiplatform google-auth"
            )
            return {}

        try:
            # Load credentials from JSON file
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Initialize Vertex AI
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials
            )

            # Determine mime type
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }
            mime_type = mime_map.get(suffix.lower(), "image/png")

            # Create the prompt for structured analysis
            prompt = """Analyze this image and provide a structured response in JSON format with these fields:

1. "description": A detailed description of what the image shows (2-3 sentences)
2. "visible_text": Any text visible in the image, transcribed accurately
3. "image_type": One of: screenshot, diagram, flowchart, photo, presentation_slide, document, chart, infographic, logo, ui_mockup, other
4. "topics": List of 3-5 key topics or themes in the image
5. "entities": List of any named entities (people, companies, products, technologies) mentioned or shown

Respond ONLY with valid JSON, no markdown formatting or extra text."""

            # Create model instance
            model = GenerativeModel(self.model)

            # Create image part
            image_part = Part.from_data(data=image_bytes, mime_type=mime_type)

            # Generate content
            response = model.generate_content([image_part, prompt])

            # Parse the JSON response
            response_text = response.text.strip()
            # Remove markdown code block if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json) and last line (```)
                response_text = "\n".join(lines[1:-1])

            result = json.loads(response_text)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Vertex AI response as JSON: {e}")
            # Try to extract what we can from non-JSON response
            return {"description": response.text[:500] if 'response' in dir() else ""}
        except Exception as e:
            logger.warning(f"Vertex AI analysis failed: {e}")
            return {}
