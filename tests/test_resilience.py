"""Unit tests for image analysis resilience module."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from ragtriever.extractors.resilience import (
    CircuitBreaker,
    ErrorCategory,
    ResilientClient,
    classify_error,
    get_circuit_breaker,
    reset_circuit_breaker,
)


class TestErrorClassification:
    """Tests for error classification logic."""

    def test_classify_429_is_rate_limited(self):
        """429 should be RATE_LIMITED, not trip breaker."""
        error = Mock()
        error.code = 429
        assert classify_error(error) == ErrorCategory.RATE_LIMITED

    def test_classify_401_is_auth_failure(self):
        """401 should immediately trip breaker."""
        error = Mock()
        error.code = 401
        assert classify_error(error) == ErrorCategory.AUTH_FAILURE

    def test_classify_403_is_auth_failure(self):
        """403 should immediately trip breaker."""
        error = Mock()
        error.code = 403
        assert classify_error(error) == ErrorCategory.AUTH_FAILURE

    def test_classify_400_is_permanent(self):
        """400 Bad Request should skip image, not trip breaker."""
        error = Mock()
        error.code = 400
        assert classify_error(error) == ErrorCategory.PERMANENT

    def test_classify_404_is_permanent(self):
        """404 Not Found should skip image, not trip breaker."""
        error = Mock()
        error.code = 404
        assert classify_error(error) == ErrorCategory.PERMANENT

    def test_classify_500_is_transient(self):
        """500 should retry and may trip breaker."""
        error = Mock()
        error.code = 500
        assert classify_error(error) == ErrorCategory.TRANSIENT

    def test_classify_503_is_transient(self):
        """503 should retry and may trip breaker."""
        error = Mock()
        error.code = 503
        assert classify_error(error) == ErrorCategory.TRANSIENT

    def test_classify_connection_error_is_transient(self):
        """Connection errors should retry and may trip breaker."""
        error = ConnectionError("Connection refused")
        assert classify_error(error) == ErrorCategory.TRANSIENT

    def test_classify_timeout_error_is_transient(self):
        """Timeout errors should retry and may trip breaker."""
        error = TimeoutError("Request timed out")
        assert classify_error(error) == ErrorCategory.TRANSIENT

    def test_classify_json_decode_error_is_permanent(self):
        """JSON parse errors should skip image, not trip breaker."""
        import json
        error = json.JSONDecodeError("Invalid JSON", "", 0)
        assert classify_error(error) == ErrorCategory.PERMANENT

    def test_classify_error_from_message(self):
        """Error with status code in message should be classified."""
        error = Exception("Server returned 503 Service Unavailable")
        assert classify_error(error) == ErrorCategory.TRANSIENT

    def test_classify_unknown_error_is_transient(self):
        """Unknown errors default to TRANSIENT (safer to retry)."""
        error = Exception("Something unexpected")
        assert classify_error(error) == ErrorCategory.TRANSIENT

    def test_classify_httpx_response_error(self):
        """Errors with response.status_code should be classified."""
        error = Mock()
        del error.code  # Remove code attribute
        del error.status_code  # Remove status_code attribute
        error.response = Mock()
        error.response.status_code = 429
        assert classify_error(error) == ErrorCategory.RATE_LIMITED


class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_breaker_starts_closed(self):
        """Circuit breaker should start in CLOSED state."""
        breaker = CircuitBreaker(threshold=5, reset_seconds=60)
        assert not breaker.is_open()
        assert breaker.state == "CLOSED"

    def test_breaker_trips_after_threshold(self):
        """5 consecutive TRANSIENT failures should open breaker."""
        breaker = CircuitBreaker(threshold=5, reset_seconds=60)

        for _ in range(4):
            breaker.record_failure(ErrorCategory.TRANSIENT)
            assert not breaker.is_open(), "Breaker should not trip before threshold"

        breaker.record_failure(ErrorCategory.TRANSIENT)
        assert breaker.is_open(), "Breaker should trip after 5 failures"
        assert breaker.state == "OPEN"

    def test_breaker_resets_on_success(self):
        """Success should reset failure count."""
        breaker = CircuitBreaker(threshold=5, reset_seconds=60)

        # Accumulate some failures
        for _ in range(4):
            breaker.record_failure(ErrorCategory.TRANSIENT)

        # Success resets
        breaker.record_success()

        # Now need 5 more failures to trip
        for _ in range(4):
            breaker.record_failure(ErrorCategory.TRANSIENT)
        assert not breaker.is_open()

    def test_auth_failure_trips_immediately(self):
        """Single AUTH_FAILURE should open breaker immediately."""
        breaker = CircuitBreaker(threshold=5, reset_seconds=60)
        breaker.record_failure(ErrorCategory.AUTH_FAILURE)
        assert breaker.is_open()

    def test_rate_limited_does_not_trip(self):
        """RATE_LIMITED should never trip breaker."""
        breaker = CircuitBreaker(threshold=5, reset_seconds=60)

        for _ in range(10):
            breaker.record_failure(ErrorCategory.RATE_LIMITED)

        assert not breaker.is_open()

    def test_permanent_does_not_trip(self):
        """PERMANENT should never trip breaker."""
        breaker = CircuitBreaker(threshold=5, reset_seconds=60)

        for _ in range(10):
            breaker.record_failure(ErrorCategory.PERMANENT)

        assert not breaker.is_open()

    def test_breaker_auto_resets_after_timeout(self):
        """Breaker should auto-close after reset_seconds."""
        breaker = CircuitBreaker(threshold=1, reset_seconds=1)

        # Trip the breaker
        breaker.record_failure(ErrorCategory.TRANSIENT)
        assert breaker.is_open()

        # Wait for reset
        time.sleep(1.1)

        # Should be closed now
        assert not breaker.is_open()


class TestResilientClient:
    """Tests for ResilientClient retry logic."""

    def setup_method(self):
        """Reset circuit breaker before each test."""
        reset_circuit_breaker()

    def test_success_on_first_try(self):
        """Successful call should return result without retries."""
        client = ResilientClient(
            timeout_ms=1000,
            max_retries=3,
            provider="test",
        )

        mock_fn = Mock(return_value={"result": "success"})
        result = client.call(mock_fn, "arg1", source_path=Path("test.png"))

        assert result == {"result": "success"}
        assert mock_fn.call_count == 1

    def test_retries_on_transient_error(self):
        """Should retry up to max_retries on TRANSIENT errors."""
        client = ResilientClient(
            timeout_ms=1000,
            max_retries=2,
            backoff_base_ms=10,  # Fast for testing
            provider="test",
        )

        mock_fn = Mock(side_effect=ConnectionError("Connection refused"))
        result = client.call(mock_fn, "arg1", source_path=Path("test.png"))

        assert result is None
        assert mock_fn.call_count == 3  # Initial + 2 retries

    def test_no_retry_on_permanent_error(self):
        """Should not retry PERMANENT errors."""
        import json
        client = ResilientClient(
            timeout_ms=1000,
            max_retries=3,
            provider="test",
        )

        mock_fn = Mock(side_effect=json.JSONDecodeError("Invalid", "", 0))
        result = client.call(mock_fn, "arg1", source_path=Path("test.png"))

        assert result is None
        assert mock_fn.call_count == 1

    def test_no_retry_on_auth_failure(self):
        """Should not retry AUTH_FAILURE errors."""
        client = ResilientClient(
            timeout_ms=1000,
            max_retries=3,
            provider="test",
        )

        # Create an actual exception with code attribute
        class AuthError(Exception):
            code = 401

        mock_fn = Mock(side_effect=AuthError("Unauthorized"))
        result = client.call(mock_fn, "arg1", source_path=Path("test.png"))

        assert result is None
        assert mock_fn.call_count == 1

    def test_skips_when_breaker_open(self):
        """Should return None immediately if breaker is open."""
        breaker = CircuitBreaker(threshold=1, reset_seconds=60)
        breaker.record_failure(ErrorCategory.AUTH_FAILURE)  # Trip immediately

        client = ResilientClient(
            timeout_ms=1000,
            max_retries=3,
            circuit_breaker=breaker,
            provider="test",
        )

        mock_fn = Mock(return_value={"result": "success"})
        result = client.call(mock_fn, "arg1", source_path=Path("test.png"))

        assert result is None
        assert mock_fn.call_count == 0  # Never called

    def test_success_after_retry(self):
        """Should return result if succeeds on retry."""
        client = ResilientClient(
            timeout_ms=1000,
            max_retries=3,
            backoff_base_ms=10,
            provider="test",
        )

        # Fail twice, then succeed
        mock_fn = Mock(side_effect=[
            ConnectionError("Connection refused"),
            ConnectionError("Connection refused"),
            {"result": "success"},
        ])
        result = client.call(mock_fn, "arg1", source_path=Path("test.png"))

        assert result == {"result": "success"}
        assert mock_fn.call_count == 3


class TestCircuitBreakerSingleton:
    """Tests for circuit breaker singleton management."""

    def setup_method(self):
        """Reset circuit breaker before each test."""
        reset_circuit_breaker()

    def test_get_circuit_breaker_returns_same_instance(self):
        """get_circuit_breaker should return same singleton."""
        breaker1 = get_circuit_breaker()
        breaker2 = get_circuit_breaker()
        assert breaker1 is breaker2

    def test_reset_circuit_breaker_clears_singleton(self):
        """reset_circuit_breaker should clear the singleton."""
        breaker1 = get_circuit_breaker()
        breaker1.record_failure(ErrorCategory.AUTH_FAILURE)
        assert breaker1.is_open()

        reset_circuit_breaker()

        breaker2 = get_circuit_breaker()
        assert breaker1 is not breaker2
        assert not breaker2.is_open()
