from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import json
import os
import re

from .base import Extracted

logger = logging.getLogger(__name__)


def _fix_json_escapes(text: str) -> str:
    """Fix common invalid escape sequences in JSON returned by LLMs.

    LLMs sometimes return JSON with invalid escape sequences like:
    - \\N, \\: (backslash followed by non-escape character)
    - Unescaped control characters

    This function attempts to fix these before JSON parsing.
    """
    # Fix invalid escape sequences: \X where X is not a valid JSON escape char
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    def fix_escape(match: re.Match) -> str:
        char = match.group(1)
        if char in '"\\bfnrt/':
            return match.group(0)  # Valid escape, keep as-is
        elif char == 'u':
            return match.group(0)  # Unicode escape, keep as-is
        else:
            # Invalid escape - double the backslash to make it literal
            return '\\\\' + char

    # Match backslash followed by any character
    fixed = re.sub(r'\\(.)', fix_escape, text)
    return fixed


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
                ],  # type: ignore[arg-type]
            )

            # Parse the JSON response
            response_text = (response.text or "").strip()
            # Remove markdown code block if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json) and last line (```)
                response_text = "\n".join(lines[1:-1])

            # Try parsing as-is first, then with escape fixes
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fix invalid escape sequences and retry
                fixed_text = _fix_json_escapes(response_text)
                result = json.loads(fixed_text)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini response as JSON: {e}")
            # Try to extract what we can from non-JSON response
            return {"description": (response.text or "")[:500] if 'response' in locals() else ""}
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
        logger.info(f"Starting Vertex AI analysis (image size: {len(image_bytes)} bytes)")

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
            # Security: Log at debug level to avoid exposing credentials path
            logger.debug(f"Loading credentials from: {self.credentials_file}")
            logger.info("Loading Vertex AI credentials from configured file")
            # Load credentials from JSON file
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Security: Log at debug level to avoid exposing project ID
            logger.debug(f"Initializing Vertex AI (project: {self.project_id}, location: {self.location})")
            logger.info(f"Initializing Vertex AI in region: {self.location}")
            # Initialize Vertex AI
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials
            )
            logger.info("Vertex AI initialized successfully")

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
            logger.info(f"Creating model instance: {self.model}")
            model = GenerativeModel(self.model)

            # Create image part
            logger.info(f"Creating image part with mime type: {mime_type}")
            image_part = Part.from_data(data=image_bytes, mime_type=mime_type)

            # Generate content
            logger.info("Sending request to Vertex AI...")
            response = model.generate_content([image_part, prompt])
            logger.info("Received response from Vertex AI")

            # Parse the JSON response
            # Handle both single-part and multi-part responses
            try:
                response_text = response.text.strip()
            except ValueError:
                # Multi-part response (e.g., from gemini-3-pro-image-preview)
                logger.info("Multi-part response detected, concatenating non-thought parts...")
                parts = []
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        # Skip thought parts - only use actual response
                        if hasattr(part, 'thought') and part.thought:
                            continue
                        if hasattr(part, 'text') and part.text:
                            parts.append(part.text)
                response_text = "".join(parts).strip()  # Use empty string join to avoid adding newlines

            logger.info(f"Response text length: {len(response_text)}")
            # Remove markdown code block if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json) and last line (```)
                response_text = "\n".join(lines[1:-1])

            # Try parsing as-is first, then with escape fixes
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fix invalid escape sequences and retry
                fixed_text = _fix_json_escapes(response_text)
                result = json.loads(fixed_text)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Vertex AI response as JSON: {e}")
            # Try to extract what we can from non-JSON response
            return {"description": response_text[:500] if response_text else ""}
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Credentials file error: {type(e).__name__}")
            logger.debug(f"Credentials file error details: {e}")
            return {}
        except ImportError as e:
            logger.error(f"Missing required library: {e}")
            return {}
        except Exception as e:
            # Catch-all for unexpected errors, but log the type for debugging
            logger.error(f"Vertex AI analysis failed: {type(e).__name__}")
            logger.debug(f"Vertex AI analysis error details: {e}")
            return {}


@dataclass
class AIGatewayImageExtractor:
    """Image extractor using Microsoft AI Gateway to access Google Gemini models.

    Uses Microsoft AI Gateway as a proxy to Gemini vision models for image analysis.
    This enables:
    - Enterprise authentication and routing through Microsoft infrastructure
    - Access control and usage tracking via AI Gateway
    - Centralized endpoint management

    Extracts:
    - Detailed description of the image content
    - Any text visible in the image (OCR)
    - Key topics/themes
    - Entities mentioned (people, organizations, products)
    - Image type classification (screenshot, diagram, photo, etc.)
    """
    supported_suffixes = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    gateway_url: str | None = None
    gateway_key: str | None = None
    model: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        # Try to get values from environment if not provided
        if self.gateway_url is None:
            self.gateway_url = os.environ.get("AI_GATEWAY_URL")
        if self.gateway_key is None:
            self.gateway_key = os.environ.get("AI_GATEWAY_KEY")

    def extract(self, path: Path) -> Extracted:
        """Extract image content using Microsoft AI Gateway proxy to Gemini."""
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            raise RuntimeError("Pillow required for image extraction")

        # Get basic image info
        with Image.open(path) as img:
            w, h = img.size
            img_format = img.format or path.suffix.lstrip(".").upper()

        # Read image bytes
        image_bytes = path.read_bytes()

        # Analyze with AI Gateway
        analysis = self._analyze_with_aigateway(image_bytes, path.suffix, path)

        # Build the text content for indexing (same pattern as GeminiImageExtractor)
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
            "analysis_provider": "aigateway",
        }

        return Extracted(text=text, metadata=meta)

    def _analyze_with_aigateway(self, image_bytes: bytes, suffix: str, source_path: Path | None = None) -> dict[str, Any]:
        """Call Microsoft AI Gateway to access Gemini API for image analysis."""
        if not self.gateway_url:
            logger.warning(
                "No AI Gateway URL found. Set aigateway_url in config or AI_GATEWAY_URL environment variable. "
                "Returning empty analysis."
            )
            return {}

        if not self.gateway_key:
            logger.warning(
                "No AI Gateway key found. Set aigateway_key in config or AI_GATEWAY_KEY environment variable. "
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
            # Configure client to use Microsoft AI Gateway endpoint
            # Append /vertex-ai-express to base URL (per ref-docs/GoogleVertexNative.ipynb)
            endpoint = f"{self.gateway_url}/vertex-ai-express"
            client = genai.Client(
                http_options=types.HttpOptions(
                    base_url=endpoint,
                    api_version="v1",
                    timeout=90000,  # 90 second timeout to prevent hanging
                ),
                api_key=self.gateway_key,
                vertexai=True
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

            # Create the prompt for structured analysis (same as GeminiImageExtractor)
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
                    types.Part.from_text(text=prompt),
                ],  # type: ignore[arg-type]
            )

            # Parse the JSON response (same pattern as GeminiImageExtractor)
            response_text = (response.text or "").strip()
            # Remove markdown code block if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            # Try parsing as-is first, then with escape fixes
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                fixed_text = _fix_json_escapes(response_text)
                result = json.loads(fixed_text)
            return result

        except json.JSONDecodeError as e:
            source_info = f" [source: {source_path}]" if source_path else ""
            logger.warning(f"Failed to parse AI Gateway response as JSON: {e}{source_info}")
            return {"description": (response.text or "")[:500] if 'response' in locals() else ""}
        except Exception as e:
            source_info = f" [source: {source_path}]" if source_path else ""
            logger.warning(f"AI Gateway analysis failed: {e}{source_info}")
            return {}
