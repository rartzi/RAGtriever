# Image Analysis Resilience Design

## Overview

Add retry logic, circuit breaker, and timeout handling to image analysis providers (Gemini, Gemini service account, AI Gateway) to prevent scan hangs and gracefully handle API failures.

## Goals

- Prevent scans from hanging on unresponsive APIs
- Retry transient errors with exponential backoff
- Stop hammering broken APIs via circuit breaker
- Log errors with enough context for debugging
- Respect rate limits without tripping circuit breaker

## Module Structure

**New file:** `src/ragtriever/extractors/resilience.py`

```
resilience.py
├── ErrorCategory (enum)
│   ├── TRANSIENT      # retry, may trip breaker
│   ├── RATE_LIMITED   # retry with backoff, never trips breaker
│   ├── PERMANENT      # skip image, never trips breaker
│   └── AUTH_FAILURE   # immediately trip breaker
│
├── classify_error(exception) -> ErrorCategory
├── CircuitBreaker (class) - thread-safe, module-level singleton
└── ResilientClient (class) - wraps API calls with retry logic
```

## Error Classification

| Error Type | Category | Behavior |
|------------|----------|----------|
| Connection errors (reset, refused, DNS) | TRANSIENT | Retry 3x → trip breaker |
| Timeouts | TRANSIENT | Retry 3x → trip breaker |
| 500, 502, 503, 504 | TRANSIENT | Retry 3x → trip breaker |
| 429 Rate limit | RATE_LIMITED | Retry with backoff, respect Retry-After header |
| 400 Bad Request | PERMANENT | Skip image, don't trip breaker |
| 401/403 Auth errors | AUTH_FAILURE | Immediately trip breaker |
| 404 Not Found | PERMANENT | Skip image, don't trip breaker |
| JSON parse errors | PERMANENT | Skip image (LLM fluke) |

## Circuit Breaker

- **Threshold:** 5 consecutive failures to trip
- **Reset:** Auto-closes after 60 seconds
- **Thread-safe:** Uses lock for parallel workers
- **Shared singleton:** All workers share one breaker per provider type
- **Smart tripping:** Only TRANSIENT errors accumulate; AUTH_FAILURE trips immediately

## Configuration

```toml
[image_analysis]
provider = "aigateway"
timeout = 30000              # Shared default (30s)
max_retries = 3
retry_backoff = 1000         # Base backoff in ms (doubles each retry)
circuit_threshold = 5
circuit_reset = 60           # Seconds

# Per-provider timeout overrides (optional)
[aigateway]
timeout = 30000

[gemini_service_account]
timeout = 45000

[gemini]
timeout = 30000
```

## Logging Strategy

| Scenario | Level | Example |
|----------|-------|---------|
| Retry attempt | INFO | `"Retry 2/3 for image.png: 503 Service Unavailable (waiting 2s)"` |
| Circuit breaker tripped | WARNING | `"Circuit breaker OPEN: 5 consecutive failures. Pausing image analysis for 60s"` |
| Circuit breaker reset | INFO | `"Circuit breaker CLOSED: resuming image analysis"` |
| Permanent skip (400/404) | WARNING | `"Skipping image.png: 400 Bad Request - invalid image format"` |
| Auth failure (401/403) | ERROR | `"Auth failed for aigateway: 401 Unauthorized. Check AI_GATEWAY_KEY."` |
| Rate limited | INFO | `"Rate limited (429), backing off 4s for image.png"` |
| Success after retry | DEBUG | `"image.png succeeded on retry 2"` |

All error logs include: source file path, provider name, HTTP status code, retry count.

## Extractor Integration

Each extractor initializes a `ResilientClient` and wraps API calls:

```python
def _analyze_with_aigateway(self, image_bytes, suffix, source_path):
    result = self._client.call(
        self._raw_api_call,
        image_bytes, suffix,
        source_path=source_path,
    )
    return result or {}
```

## Testing

Unit tests in `tests/test_resilience.py`:
- Error classification for each category
- Circuit breaker threshold and reset behavior
- ResilientClient retry logic
- Retry-After header handling

Integration tests:
- `tests/test_aigateway_resilience.py` - test with AI Gateway
- `tests/test_gemini_sa_resilience.py` - test with Gemini service account
