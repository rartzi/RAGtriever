"""Resilience utilities for image analysis API calls.

Provides retry logic, circuit breaker, and error classification for
graceful handling of API failures during image analysis.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorCategory(Enum):
    """Classification of errors for retry/circuit breaker behavior."""

    TRANSIENT = "transient"  # Retry, may trip breaker after threshold
    RATE_LIMITED = "rate_limited"  # Retry with backoff, never trips breaker
    PERMANENT = "permanent"  # Skip image, never trips breaker
    AUTH_FAILURE = "auth_failure"  # Immediately trip breaker


def _extract_status_code(error: Exception) -> int | None:
    """Extract HTTP status code from various exception types."""
    # google-genai errors
    if hasattr(error, "code"):
        code = getattr(error, "code")
        if isinstance(code, int):
            return code

    if hasattr(error, "status_code"):
        code = getattr(error, "status_code")
        if isinstance(code, int):
            return code

    # httpx errors
    if hasattr(error, "response"):
        response = getattr(error, "response")
        if response is not None and hasattr(response, "status_code"):
            return response.status_code

    # google.api_core errors
    if hasattr(error, "grpc_status_code"):
        # Map gRPC codes to HTTP-like codes for classification
        grpc_code = getattr(error, "grpc_status_code")
        grpc_to_http = {
            1: 499,  # CANCELLED
            2: 500,  # UNKNOWN
            3: 400,  # INVALID_ARGUMENT
            4: 504,  # DEADLINE_EXCEEDED
            5: 404,  # NOT_FOUND
            7: 403,  # PERMISSION_DENIED
            8: 429,  # RESOURCE_EXHAUSTED (rate limit)
            13: 500,  # INTERNAL
            14: 503,  # UNAVAILABLE
            16: 401,  # UNAUTHENTICATED
        }
        if grpc_code in grpc_to_http:
            return grpc_to_http[grpc_code]

    # Try parsing from error message as last resort
    error_str = str(error)
    for code in (400, 401, 403, 404, 429, 500, 502, 503, 504):
        if str(code) in error_str:
            return code

    return None


def _extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After header value from error if present."""
    # httpx response
    if hasattr(error, "response"):
        response = getattr(error, "response")
        if response is not None and hasattr(response, "headers"):
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

    # google-genai may include retry info
    if hasattr(error, "retry_delay"):
        delay = getattr(error, "retry_delay")
        if isinstance(delay, (int, float)):
            return float(delay)

    return None


def classify_error(error: Exception) -> ErrorCategory:
    """Classify exception into retry/breaker behavior category."""
    status = _extract_status_code(error)

    if status:
        if status == 429:
            return ErrorCategory.RATE_LIMITED
        if status in (401, 403):
            return ErrorCategory.AUTH_FAILURE
        if status in (400, 404, 422):
            return ErrorCategory.PERMANENT
        if status in (500, 502, 503, 504):
            return ErrorCategory.TRANSIENT

    # Check exception type for connection issues
    error_type = type(error).__name__
    error_msg = str(error).lower()

    # Connection errors → TRANSIENT
    connection_types = ("Connection", "Timeout", "Socket", "Transport", "Network")
    if any(x in error_type for x in connection_types):
        return ErrorCategory.TRANSIENT

    connection_msgs = ("connection", "timeout", "timed out", "reset", "refused", "unreachable")
    if any(x in error_msg for x in connection_msgs):
        return ErrorCategory.TRANSIENT

    # JSON parse errors → PERMANENT (LLM fluke, skip this image)
    if "json" in error_type.lower() or "decode" in error_msg:
        return ErrorCategory.PERMANENT

    # Unknown errors → TRANSIENT (safer to retry)
    return ErrorCategory.TRANSIENT


class CircuitBreaker:
    """Thread-safe circuit breaker for image analysis APIs.

    Tracks consecutive failures and opens (blocks requests) when threshold
    is reached. Auto-resets after reset_seconds.
    """

    def __init__(self, threshold: int = 5, reset_seconds: int = 60):
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self._failure_count = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        """Check if breaker is open (blocking requests)."""
        with self._lock:
            if self._opened_at is None:
                return False
            # Auto-reset after reset_seconds
            if time.time() - self._opened_at >= self.reset_seconds:
                self._close()
                return False
            return True

    def record_success(self) -> None:
        """Reset failure count on success."""
        with self._lock:
            if self._failure_count > 0:
                logger.debug(f"Circuit breaker: success, resetting failure count from {self._failure_count}")
            self._failure_count = 0

    def record_failure(self, category: ErrorCategory) -> None:
        """Record failure, potentially trip breaker."""
        with self._lock:
            if category == ErrorCategory.AUTH_FAILURE:
                self._open("Authentication failure")
            elif category == ErrorCategory.TRANSIENT:
                self._failure_count += 1
                logger.debug(f"Circuit breaker: failure {self._failure_count}/{self.threshold}")
                if self._failure_count >= self.threshold:
                    self._open(f"{self.threshold} consecutive failures")
            # RATE_LIMITED and PERMANENT don't affect breaker

    def _open(self, reason: str) -> None:
        """Open the circuit breaker."""
        self._opened_at = time.time()
        logger.warning(
            f"Circuit breaker OPEN: {reason}. "
            f"Pausing image analysis for {self.reset_seconds}s"
        )

    def _close(self) -> None:
        """Close the circuit breaker."""
        self._opened_at = None
        self._failure_count = 0
        logger.info("Circuit breaker CLOSED: resuming image analysis")

    @property
    def state(self) -> str:
        """Return current state for debugging."""
        with self._lock:
            if self._opened_at is None:
                return "CLOSED"
            if time.time() - self._opened_at >= self.reset_seconds:
                return "HALF-OPEN"
            return "OPEN"


# Module-level singleton circuit breaker
_default_breaker: CircuitBreaker | None = None
_breaker_lock = threading.Lock()


def get_circuit_breaker(threshold: int = 5, reset_seconds: int = 60) -> CircuitBreaker:
    """Get or create the shared circuit breaker singleton."""
    global _default_breaker
    with _breaker_lock:
        if _default_breaker is None:
            _default_breaker = CircuitBreaker(threshold=threshold, reset_seconds=reset_seconds)
        return _default_breaker


def reset_circuit_breaker() -> None:
    """Reset the circuit breaker singleton (for testing)."""
    global _default_breaker
    with _breaker_lock:
        _default_breaker = None


@dataclass
class ResilientClient:
    """Wraps API calls with retry logic and circuit breaker.

    Usage:
        client = ResilientClient(timeout_ms=30000, provider="aigateway")
        result = client.call(api_function, arg1, arg2, source_path=path)
    """

    timeout_ms: int = 30000
    max_retries: int = 3
    backoff_base_ms: int = 1000
    circuit_breaker: CircuitBreaker | None = None
    provider: str = "unknown"

    _breaker: CircuitBreaker = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._breaker = self.circuit_breaker or get_circuit_breaker()

    @property
    def timeout_s(self) -> float:
        """Timeout in seconds."""
        return self.timeout_ms / 1000.0

    @property
    def backoff_base_s(self) -> float:
        """Base backoff in seconds."""
        return self.backoff_base_ms / 1000.0

    def call(
        self,
        fn: Callable[..., T],
        *args: Any,
        source_path: Path | None = None,
        **kwargs: Any,
    ) -> T | None:
        """Execute fn with retry/circuit breaker. Returns None on failure."""
        source = source_path.name if source_path else "unknown"

        # Check circuit breaker first
        if self._breaker.is_open():
            logger.debug(f"[{self.provider}] Skipping {source}: circuit breaker open")
            return None

        for attempt in range(self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                if attempt > 0:
                    logger.debug(f"[{self.provider}] {source} succeeded on retry {attempt}")
                self._breaker.record_success()
                return result

            except Exception as e:
                category = classify_error(e)

                # Log with context
                self._log_error(e, category, source, attempt)

                # Handle based on category
                if category == ErrorCategory.PERMANENT:
                    return None  # Skip, don't retry

                if category == ErrorCategory.AUTH_FAILURE:
                    self._breaker.record_failure(category)
                    return None  # Don't retry

                # TRANSIENT or RATE_LIMITED: maybe retry
                if attempt < self.max_retries:
                    wait = self._get_backoff(e, category, attempt)
                    logger.info(
                        f"[{self.provider}] Retry {attempt + 1}/{self.max_retries} for {source}: "
                        f"{type(e).__name__} (waiting {wait:.1f}s)"
                    )
                    time.sleep(wait)
                else:
                    # Exhausted retries
                    if category == ErrorCategory.TRANSIENT:
                        self._breaker.record_failure(category)

        return None

    def _get_backoff(self, error: Exception, category: ErrorCategory, attempt: int) -> float:
        """Calculate backoff time, respecting Retry-After header."""
        if category == ErrorCategory.RATE_LIMITED:
            retry_after = _extract_retry_after(error)
            if retry_after:
                logger.info(f"[{self.provider}] Using Retry-After header: {retry_after}s")
                return retry_after
        # Exponential backoff: 1s, 2s, 4s
        return self.backoff_base_s * (2 ** attempt)

    def _log_error(
        self,
        error: Exception,
        category: ErrorCategory,
        source: str,
        attempt: int,
    ) -> None:
        """Log error with appropriate level and context."""
        status = _extract_status_code(error)
        status_str = f" ({status})" if status else ""
        error_type = type(error).__name__

        context = f"[{self.provider}] {source}{status_str}"

        if category == ErrorCategory.AUTH_FAILURE:
            logger.error(
                f"{context}: Auth failed - {error_type}. "
                f"Check credentials/API key. Circuit breaker will open."
            )
        elif category == ErrorCategory.PERMANENT:
            logger.warning(f"{context}: Skipping - {error_type}: {error}")
        elif category == ErrorCategory.RATE_LIMITED:
            logger.info(f"{context}: Rate limited (429)")
        elif category == ErrorCategory.TRANSIENT:
            if attempt == 0:
                logger.warning(f"{context}: {error_type}: {error}")
            else:
                logger.debug(f"{context}: Retry {attempt} failed - {error_type}")
