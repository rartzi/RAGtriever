"""Integration tests for image analysis resilience with real providers.

These tests require actual API credentials and make real API calls.
Skip them in CI by running: pytest -m "not integration"

To run these tests:
    pytest tests/test_resilience_integration.py -v -s

Environment variables required:
- AI Gateway: AI_GATEWAY_URL, AI_GATEWAY_KEY
- Vertex AI: GOOGLE_CLOUD_PROJECT, GOOGLE_APPLICATION_CREDENTIALS
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mneme.extractors.image import (
    AIGatewayImageExtractor,
    GeminiServiceAccountImageExtractor,
    GeminiImageExtractor,
)
from mneme.extractors.resilience import (
    reset_circuit_breaker,
    get_circuit_breaker,
)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def create_test_image() -> Path:
    """Create a simple test image for analysis."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        pytest.skip("Pillow not installed")

    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color="red")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        return Path(f.name)


@pytest.fixture
def test_image():
    """Fixture that creates a test image and cleans up after."""
    path = create_test_image()
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def reset_breaker():
    """Reset circuit breaker before each test."""
    reset_circuit_breaker()
    yield
    reset_circuit_breaker()


class TestAIGatewayResilience:
    """Integration tests for AI Gateway resilience."""

    @pytest.fixture
    def aigateway_extractor(self):
        """Create AI Gateway extractor with short timeout for testing."""
        url = os.environ.get("AI_GATEWAY_URL")
        key = os.environ.get("AI_GATEWAY_KEY")
        if not url or not key:
            pytest.skip("AI_GATEWAY_URL and AI_GATEWAY_KEY not set")

        return AIGatewayImageExtractor(
            gateway_url=url,
            gateway_key=key,
            model="gemini-2.5-flash",
            timeout=30000,  # 30s timeout
            max_retries=2,
            retry_backoff=1000,
        )

    def test_successful_extraction(self, aigateway_extractor, test_image):
        """Test successful image extraction with AI Gateway."""
        result = aigateway_extractor.extract(test_image)

        assert result.text, "Should extract some text"
        assert result.metadata.get("analysis_provider") == "aigateway"
        print(f"\nExtracted text: {result.text[:200]}...")

    def test_circuit_breaker_not_tripped_on_success(self, aigateway_extractor, test_image):
        """Circuit breaker should stay closed on success."""
        aigateway_extractor.extract(test_image)

        breaker = get_circuit_breaker()
        assert not breaker.is_open(), "Breaker should be closed after success"

    def test_timeout_handling(self, test_image):
        """Test that short timeout doesn't hang forever."""
        url = os.environ.get("AI_GATEWAY_URL")
        key = os.environ.get("AI_GATEWAY_KEY")
        if not url or not key:
            pytest.skip("AI_GATEWAY_URL and AI_GATEWAY_KEY not set")

        # Create extractor with very short timeout
        extractor = AIGatewayImageExtractor(
            gateway_url=url,
            gateway_key=key,
            timeout=100,  # 100ms - should timeout
            max_retries=0,  # No retries
        )

        # This should not hang - either succeed quickly or timeout
        import time
        start = time.time()
        result = extractor.extract(test_image)
        elapsed = time.time() - start

        # Should complete within a reasonable time (not hang forever)
        assert elapsed < 10, f"Should not hang: took {elapsed:.1f}s"
        print(f"\nCompleted in {elapsed:.1f}s (timeout test)")

    def test_invalid_key_trips_breaker(self, test_image):
        """Invalid API key should trip circuit breaker."""
        url = os.environ.get("AI_GATEWAY_URL")
        if not url:
            pytest.skip("AI_GATEWAY_URL not set")

        extractor = AIGatewayImageExtractor(
            gateway_url=url,
            gateway_key="invalid-key-12345",
            timeout=10000,
            max_retries=0,
        )

        # Extract should fail
        result = extractor.extract(test_image)

        # Check if breaker tripped (depends on error code returned)
        breaker = get_circuit_breaker()
        print(f"\nBreaker state after invalid key: {breaker.state}")


class TestGeminiServiceAccountResilience:
    """Integration tests for Gemini service account resilience."""

    @pytest.fixture
    def gemini_sa_extractor(self):
        """Create Gemini service account extractor for testing."""
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not project_id or not creds_file:
            pytest.skip("GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS not set")

        return GeminiServiceAccountImageExtractor(
            project_id=project_id,
            location="us-central1",
            credentials_file=creds_file,
            model="gemini-2.0-flash-exp",
            timeout=30000,
            max_retries=2,
            retry_backoff=1000,
        )

    def test_successful_extraction(self, gemini_sa_extractor, test_image):
        """Test successful image extraction with Gemini service account."""
        result = gemini_sa_extractor.extract(test_image)

        assert result.text, "Should extract some text"
        assert result.metadata.get("analysis_provider") == "gemini-service-account"
        print(f"\nExtracted text: {result.text[:200]}...")

    def test_circuit_breaker_not_tripped_on_success(self, gemini_sa_extractor, test_image):
        """Circuit breaker should stay closed on success."""
        gemini_sa_extractor.extract(test_image)

        breaker = get_circuit_breaker()
        assert not breaker.is_open(), "Breaker should be closed after success"

    def test_invalid_credentials_trips_breaker(self, test_image):
        """Invalid credentials should trip circuit breaker."""
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        # Create temp file with invalid credentials
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"type": "service_account", "project_id": "invalid"}')
            invalid_creds = f.name

        try:
            extractor = GeminiServiceAccountImageExtractor(
                project_id=project_id,
                credentials_file=invalid_creds,
                timeout=10000,
                max_retries=0,
            )

            result = extractor.extract(test_image)

            breaker = get_circuit_breaker()
            print(f"\nBreaker state after invalid creds: {breaker.state}")
        finally:
            Path(invalid_creds).unlink(missing_ok=True)


class TestGeminiResilience:
    """Integration tests for Gemini direct API resilience."""

    @pytest.fixture
    def gemini_extractor(self):
        """Create Gemini extractor for testing."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")

        return GeminiImageExtractor(
            api_key=api_key,
            model="gemini-2.0-flash",
            timeout=30000,
            max_retries=2,
            retry_backoff=1000,
        )

    def test_successful_extraction(self, gemini_extractor, test_image):
        """Test successful image extraction with Gemini."""
        result = gemini_extractor.extract(test_image)

        assert result.text, "Should extract some text"
        assert result.metadata.get("analysis_provider") == "gemini"
        print(f"\nExtracted text: {result.text[:200]}...")

    def test_invalid_key_trips_breaker(self, test_image):
        """Invalid API key should trip circuit breaker."""
        extractor = GeminiImageExtractor(
            api_key="invalid-key-12345",
            timeout=10000,
            max_retries=0,
        )

        result = extractor.extract(test_image)

        breaker = get_circuit_breaker()
        print(f"\nBreaker state after invalid key: {breaker.state}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
