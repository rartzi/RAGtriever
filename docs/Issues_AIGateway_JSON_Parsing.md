# Issue: AI Gateway JSON Parsing Errors

## Status
**Open** - Created 2026-01-23

## Description

When using the AI Gateway (`provider = "aigateway"`) for image analysis, some responses fail to parse as JSON with errors like:

```
Failed to parse AI Gateway response as JSON: Expecting ',' delimiter: line 20 column 3 (char 864)
Failed to parse AI Gateway response as JSON: Expecting ',' delimiter: line 37 column 4 (char 2690)
```

## Impact

- Some images fail to be analyzed and indexed
- The scan continues (graceful error handling) but those images have no AI-generated metadata

## Root Cause (Investigation Needed)

Possible causes:
1. **Model output not properly formatted as JSON** - The prompt asks for JSON but the model sometimes returns malformed JSON
2. **Response truncation** - Long responses may be cut off mid-JSON
3. **Special characters in image content** - OCR'd text or descriptions may contain characters that break JSON
4. **Model hallucination** - The model may add commentary outside the JSON structure

## Reproduction

1. Configure `provider = "aigateway"` in config.toml
2. Run `ragtriever scan --workers 2`
3. Observe stderr for JSON parsing errors

## Code Location

`src/ragtriever/extractors/image.py` - `AIGatewayImageExtractor._analyze_with_aigateway()` method

The JSON parsing happens around line 600+ where we try to extract structured data from the model response.

## Proposed Fixes

### Option A: More Robust JSON Extraction
- Use regex to find JSON block within response
- Strip any text before/after the JSON
- Handle common malformations (trailing commas, etc.)

### Option B: Retry with Stricter Prompt
- If parsing fails, retry with a more explicit prompt
- Add examples of expected JSON format
- Use JSON mode if available via API

### Option C: Fallback to Unstructured
- If JSON parsing fails, extract what we can as plain text
- Store raw description without structured fields

## Related Files

- `src/ragtriever/extractors/image.py` - AIGatewayImageExtractor class
- `test_aigateway_debug.py` - Debug script for testing AI Gateway
- `ref-docs/GoogleVertexNative.ipynb` - Reference implementation

## Testing

The `test_aigateway_debug.py` script can be used to test individual image analysis.
A unit test should be added to `tests/test_image_extractor.py` for AI Gateway.

## Priority

Medium - Images can still be indexed (text content), but miss AI-generated metadata.

---

## Additional Issue: Error Context Needs Parent Document Info

**Problem:** When image analysis fails, error logs show temp file paths (e.g., `/tmp/tmpsg55jt1t.png`) instead of the original source document (e.g., `Innovation Session Cambridge.pptx slide 3`).

**Current behavior:**
```
AI Gateway analysis failed: 400 INVALID_ARGUMENT [source: /var/folders/.../tmpsg55jt1t.png]
```

**Desired behavior:**
```
AI Gateway analysis failed: 400 INVALID_ARGUMENT [source: Innovation Session Cambridge.pptx, slide_3]
```

**Root cause:** The `ImageTask` has `parent_path` with the original document info, but this isn't passed to the image extractor where errors are logged.

**Fix:** Pass parent document context to `_analyze_with_aigateway()` method so error messages include original source location.

**Files to modify:**
- `src/ragtriever/extractors/image.py` - Accept parent context in error logs
- `src/ragtriever/indexer/indexer.py` - Pass parent info to extractor
