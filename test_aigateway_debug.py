#!/usr/bin/env python3
"""
Quick debug script to test AI Gateway image analysis endpoint.
Based on ref-docs/GoogleVertexNative.ipynb

Usage:
    python test_aigateway_debug.py

Requires environment variables:
    AI_GATEWAY_URL - Base URL of AI Gateway (e.g., https://ai-gateway.astrazeneca.net)
    AI_GATEWAY_KEY - API key for AI Gateway
"""

import os
import sys
from io import BytesIO
from pathlib import Path

# Try to load from .env if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get credentials
GATEWAY_URL = os.environ.get("AI_GATEWAY_URL", "https://ai-gateway.astrazeneca.net")
API_KEY = os.environ.get("AI_GATEWAY_KEY")

if not API_KEY:
    # Try loading from config.toml (Python 3.11+ has tomllib built-in)
    try:
        import tomllib
        config_path = Path(__file__).parent / "config.toml"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
                API_KEY = config.get("aigateway", {}).get("key")
                GATEWAY_URL = config.get("aigateway", {}).get("url", GATEWAY_URL)
                # Strip any path suffix from URL - we'll add it in tests
                if GATEWAY_URL and "/vertex-ai" in GATEWAY_URL:
                    GATEWAY_URL = GATEWAY_URL.rsplit("/vertex-ai", 1)[0]
    except Exception as e:
        print(f"Could not load from config.toml: {e}")

if not API_KEY:
    print("ERROR: No API key found. Set AI_GATEWAY_KEY env var or configure in config.toml")
    sys.exit(1)


def create_test_image() -> bytes:
    """Create a simple test image with text."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("ERROR: Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Hello AI Gateway", fill='black')

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def test_endpoint(endpoint_path: str, model: str = "gemini-2.5-flash") -> dict:
    """Test a specific endpoint configuration."""
    from google import genai
    from google.genai import types
    from google.genai.types import GenerateContentConfig, HttpOptions

    # Build full endpoint URL
    if endpoint_path:
        endpoint = f"{GATEWAY_URL}/{endpoint_path}"
    else:
        endpoint = GATEWAY_URL

    print(f"\n{'='*60}")
    print(f"Testing endpoint: {endpoint}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    result = {
        "endpoint": endpoint,
        "model": model,
        "success": False,
        "error": None,
        "response": None,
    }

    try:
        # Configure client (matching ref-docs/GoogleVertexNative.ipynb)
        client = genai.Client(
            http_options=HttpOptions(base_url=endpoint, api_version="v1"),
            api_key=API_KEY,
            vertexai=True
        )

        # Create test image
        image_bytes = create_test_image()

        # Test image analysis
        prompt = "What text do you see in this image? Reply briefly."

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ],
            config=GenerateContentConfig(max_output_tokens=100),
        )

        result["success"] = True
        result["response"] = response.text
        print(f"SUCCESS! Response: {response.text[:200]}...")

    except Exception as e:
        result["error"] = str(e)
        error_type = type(e).__name__
        print(f"FAILED: {error_type}: {e}")

    return result


def main():
    print("AI Gateway Image Analysis Debug Script")
    print(f"Base URL: {GATEWAY_URL}")
    print(f"API Key: {API_KEY[:20]}...{API_KEY[-10:]}")

    # Test configurations to try
    test_configs = [
        # (endpoint_path, model)
        ("vertex-ai-express", "gemini-2.5-flash"),
        ("vertex-ai-openai", "gemini-2.5-flash"),
        ("vertex-ai-express", "google/gemini-2.5-flash"),
        ("vertex-ai-openai", "google/gemini-2.5-flash"),
        ("", "gemini-2.5-flash"),  # No path suffix
    ]

    results = []
    for endpoint_path, model in test_configs:
        result = test_endpoint(endpoint_path, model)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    print(f"\nSuccessful configurations ({len(successes)}):")
    for r in successes:
        print(f"  - {r['endpoint']} with model {r['model']}")

    print(f"\nFailed configurations ({len(failures)}):")
    for r in failures:
        print(f"  - {r['endpoint']} with model {r['model']}")
        print(f"    Error: {r['error'][:100]}...")

    if successes:
        print("\n" + "="*60)
        print("RECOMMENDED CONFIGURATION:")
        best = successes[0]
        print(f"  endpoint_path: {best['endpoint'].replace(GATEWAY_URL + '/', '')}")
        print(f"  model: {best['model']}")
        print("="*60)
        return 0
    else:
        print("\nNo working configuration found!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
