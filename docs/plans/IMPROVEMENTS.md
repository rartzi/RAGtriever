# Planned Improvements

This document tracks planned enhancements and technical improvements for RAGtriever.

## Priority: High

### 1. Support for Gemini 3 Models with Complex Responses

**Status:** Planned
**Priority:** High
**Complexity:** Medium

#### Problem
Current Vertex AI integration works with Gemini 2.0 models, but upcoming Gemini 3 models (gemini-3-pro-image, gemini-3-pro-image-preview) provide:
- Higher quality image analysis
- More detailed structured responses
- **Multi-part responses** with "thinking" sections that should be filtered out

#### Current Implementation
```python
# src/ragtriever/extractors/image.py:376-390
try:
    response_text = response.text.strip()
except ValueError:
    # Multi-part response handling (basic)
    parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, 'thought') and part.thought:
                continue  # Skip thought parts
            if hasattr(part, 'text') and part.text:
                parts.append(part.text)
    response_text = "".join(parts).strip()
```

#### Planned Improvements

**Phase 1: Enhanced Response Parsing**
- Add robust multi-part response handler for Gemini 3
- Detect and filter "thinking" parts automatically
- Handle reasoning traces without breaking JSON parsing
- Log thinking content at DEBUG level for diagnostics

**Phase 2: Model Configuration**
```toml
[vertex_ai]
model = "gemini-3-pro-image"  # or gemini-3-pro-image-preview
filter_thinking = true  # Default: true
log_thinking = false    # Log thinking parts for debugging
```

**Phase 3: Response Structure**
```python
class GeminiResponse:
    """Structured response from Gemini models."""
    thinking: Optional[str]  # Reasoning trace (filtered by default)
    content: str            # Actual analysis
    metadata: dict          # Model info, token usage, etc.
```

#### Implementation Tasks
- [ ] Add comprehensive multi-part response tests
- [ ] Implement thinking filter with pattern detection
- [ ] Support both Gemini 2 and Gemini 3 response formats
- [ ] Add model capability detection (auto-select parsing strategy)
- [ ] Update documentation with Gemini 3 examples
- [ ] Benchmark Gemini 3 vs Gemini 2 quality

#### Testing Strategy
```python
# Test cases needed:
# 1. Gemini 2 response (simple)
# 2. Gemini 3 response with thinking
# 3. Gemini 3 response without thinking
# 4. Mixed multi-part responses
# 5. Malformed responses with partial JSON
```

#### Success Criteria
- ✅ Gemini 3 models work without code changes
- ✅ Thinking parts automatically filtered
- ✅ JSON parsing succeeds even with thinking content
- ✅ Backward compatible with Gemini 2
- ✅ Clear logging for debugging response issues

---

### 2. Unified Scan/Watch Pipeline with Parallel Processing

**Status:** Planned
**Priority:** High
**Complexity:** Medium

#### Problem
Scan and watch modes use different code paths for indexing files:
- `scan_parallel()` uses `_extract_and_chunk_one()` with batch embedding
- `watch()` uses `_index_one()` with inline embedding (serial)

This leads to:
- Code duplication
- Watch mode is slower (no parallelism)
- Different behavior between modes

#### Current Architecture
```
Scan Mode:                          Watch Mode:
_extract_and_chunk_one()            _index_one()
  → Extract                           → Extract
  → Chunk                             → Chunk
  → Return result                     → Embed (inline)
  ↓                                   → Store (inline)
Batch embed
Batch store
```

#### Planned Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Pipeline                              │
│  _process_file(path) → ExtractionResult (chunks, metadata)      │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────────┐              ┌───────────────────┐
    │   Scan Mode       │              │   Watch Mode      │
    │   - Batch files   │              │   - Queue events  │
    │   - Parallel      │              │   - Mini-batch    │
    │     extraction    │              │     (N files or   │
    │   - Batch embed   │              │      N seconds)   │
    │   - Batch store   │              │   - Parallel      │
    └───────────────────┘              │     workers       │
                                       └───────────────────┘
```

#### Implementation Tasks
- [ ] Create shared `_process_file()` method (extract + chunk only)
- [ ] Refactor `scan_parallel()` to use shared method
- [ ] Refactor `watch()` to use shared method with mini-batching
- [ ] Add `watch_workers` config option for parallel watch processing
- [ ] Add `watch_batch_size` and `watch_batch_timeout` config options
- [ ] Ensure thread safety for SQLite store

#### Configuration
```toml
[indexing]
# Existing scan settings
extraction_workers = 8
embed_batch_size = 256

# New watch settings
watch_workers = 4           # Parallel workers for watch mode
watch_batch_size = 10       # Batch N files before embedding
watch_batch_timeout = 5     # Or batch after N seconds
```

#### Benefits
- Single code path for file processing
- Watch mode gets parallelism (faster for bulk adds)
- Easier maintenance
- Consistent behavior between modes

---

### 3. Better Error Handling for Office Files

**Status:** Planned
**Priority:** High
**Complexity:** Low

#### Problem
PPTX/DOCX extractors crash on invalid files (temp files, corrupted documents).

#### Current Behavior
```
PackageNotFoundError: Package not found at 'file.pptx'
```

#### Planned Solution
```python
# src/ragtriever/extractors/pptx.py
def extract(self, path: Path) -> Extracted:
    try:
        prs = Presentation(path)
        # ... extraction logic
    except PackageNotFoundError as e:
        logger.warning(f"Skipping invalid PPTX (likely temp/lock file): {path}")
        return Extracted(text="", metadata={"error": "invalid_package", "skipped": True})
    except Exception as e:
        logger.error(f"Failed to extract PPTX {path}: {e}")
        return Extracted(text="", metadata={"error": str(type(e).__name__)})
```

#### Benefits
- Scan continues even with problematic files
- Clear error reporting
- Graceful degradation

---

### 4. Offline Mode Validation

**Status:** Planned
**Priority:** Medium
**Complexity:** Low

#### Problem
Config allows `offline_mode = true` even when model isn't cached, causing confusing errors.

#### Planned Solution
```python
# src/ragtriever/config.py
if offline_mode:
    model_cache_path = Path.home() / ".cache" / "huggingface" / "hub" / \
                       f"models--{embedding_model.replace('/', '--')}"
    if not model_cache_path.exists():
        logger.warning(
            f"Offline mode enabled but model '{embedding_model}' not cached. "
            f"Run once with offline_mode=false to download, or use a cached model. "
            f"Check cached models: ls ~/.cache/huggingface/hub/"
        )
```

---

## Priority: Medium

### 5. Batch Image Processing with Rate Limit Handling

**Status:** Planned
**Priority:** Medium
**Complexity:** Medium

#### Current Behavior
- Images processed sequentially
- Rate limit errors stop processing

#### Planned Improvements
- Exponential backoff on rate limit errors
- Batch processing with configurable concurrency
- Resume from last successful image

```toml
[image_analysis]
batch_size = 5          # Process N images concurrently
retry_on_rate_limit = true
max_retries = 3
backoff_seconds = 5
```

---

### 6. Image Analysis Quality Metrics

**Status:** Planned
**Priority:** Medium
**Complexity:** Medium

#### Goal
Track and compare image analysis quality across providers.

#### Metrics to Track
```python
@dataclass
class AnalysisMetrics:
    provider: str           # tesseract, gemini, vertex_ai
    model: str             # Model name
    success_rate: float    # % of images analyzed
    avg_description_len: int
    avg_entities_count: int
    avg_processing_time: float
    error_rate: float
```

#### Benefits
- Compare Tesseract vs Gemini vs Vertex AI
- Justify cost of API-based analysis
- Identify problematic image types

---

## Priority: Low

### 7. Smart Chunk Size Optimization

**Status:** Planned
**Priority:** Low
**Complexity:** High

#### Current Behavior
- Fixed chunking strategy (by heading for markdown, by boundary for documents)
- No size limits

#### Planned Improvements
- Dynamic chunk sizing based on content type
- Overlap for context preservation
- Max chunk size with smart splitting

```toml
[chunking]
max_chunk_size = 1000    # tokens
overlap = 100            # tokens
strategy = "semantic"    # heading|boundary|semantic
```

---

### 8. Incremental Indexing with Change Detection

**Status:** Partially Implemented
**Priority:** Low
**Complexity:** Medium

#### Current Behavior
- `--full` flag re-indexes everything
- Without `--full`, uses mtime for change detection

#### Planned Improvements
- Content hash-based change detection (✅ partially implemented)
- Track chunk-level changes (not just document-level)
- Skip re-embedding if only metadata changed

---

### 9. MCP Server Enhancements

**Status:** Planned
**Priority:** Low
**Complexity:** Low

#### Planned Tools
```python
# New MCP tools:
- vault.analyze_image: Analyze a single image on-demand
- vault.summarize_document: Summarize a specific document
- vault.find_related: Find documents related to current one
- vault.rebuild_index: Trigger re-indexing
```

---

## Research Needed

### A. Vector Store Alternatives

**Current:** SQLite with brute-force cosine similarity
**Research:**
- FAISS integration for large vaults
- Qdrant for production deployments
- pgvector for PostgreSQL users

### B. Reranking Strategies

**Current:** Basic score fusion (vector + lexical)
**Research:**
- Cross-encoder reranking
- LLM-based relevance scoring
- Learning-to-rank approaches

### C. Multimodal Embeddings

**Current:** Separate text and image processing
**Research:**
- CLIP-based unified embeddings
- Joint text-image search
- Cross-modal retrieval

---

## How to Contribute

1. **Pick an improvement** from Priority: High or Medium
2. **Create a branch**: `feature/improvement-name`
3. **Implement with tests**
4. **Update documentation**
5. **Submit PR** with reference to this document

## Tracking

- 🟢 **Completed**: Feature is implemented and merged
- 🟡 **In Progress**: Work has started
- ⚪ **Planned**: Documented but not started
- 🔴 **Blocked**: Waiting on external dependency

Last Updated: 2026-01-23
