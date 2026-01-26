# CLAUDE.md

Mneme: Local-first vault indexer with hybrid retrieval (semantic + lexical + link-graph). Indexes Obsidian-compatible vaults, serves via CLI/Python/MCP.

**Design:** Local-only, filesystem as source of truth, pluggable Protocol-based adapters

## Migration (v2.0.0)
⚠️ Rename `provider = "vertex_ai"` → `"gemini-service-account"` in config. See [CHANGELOG.md](CHANGELOG.md).

## Commands

```bash
# Setup (requires Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Linting and type checking
ruff check src/ tests/
ruff check --fix src/ tests/    # autofix
ruff format src/ tests/         # formatting
mypy src/

# Testing
pytest                                              # all tests
pytest tests/test_markdown_parsing.py              # single file
pytest tests/test_markdown_parsing.py::test_parse_wikilinks  # single test

# CLI (after creating config.toml)
mneme init --vault "/path/to/vault" --index "~/.mneme/indexes/myvault"
mneme scan --full
mneme query "search term" --k 10
mneme query "search term" --k 10 --rerank  # with cross-encoder reranking
mneme watch              # watch mode for continuous indexing
mneme mcp                # MCP server over stdio
```

## Architecture

**Flow:** Vault → Change Detection → Ingestion → Stores → Retrieval → API/MCP

### Parallel Processing (3.6x speedup)

| Pool | Purpose | Default | Bottleneck |
|------|---------|---------|-----------|
| `extraction_workers` | All file extraction (md/pdf/pptx/xlsx/images) | 8 | CPU |
| `image_workers` | API calls (Gemini/Vertex/Gateway) | 8 | Network I/O |
| `embed_batch_size` | Cross-file embedding batches | 256 | GPU |

**Pipeline:** Parallel Extract → Serial Chunk → Batch Embed → Store. Images processed separately via parallel API workers.

**Config:** `[indexing]` section. **CLI:** `--workers N`, `--no-parallel`, `--batch-size N`. **Perf:** <50 files minimal gain, 50-500 files 2-3x, >500 files 3-4x.

### Unified Processing

Both scan/watch use `_process_file()` → `ProcessResult` (thread-safe, no DB writes). Watch uses `BatchCollector` (size/timeout triggers). **Code:** `src/mneme/indexer/indexer.py:_process_file()`, `parallel_types.py`

### Logging

**Scan:** `[scan] Phase N:`, `Found/Deleted/Failed/Complete`. **Watch:** `[watch] File created/modified/deleted/moved`, `Batch processed`. **Code:** `src/mneme/indexer/change_detector.py`

### Core Design

**Protocols (`**/base.py`):** Extractor, Chunker, Embedder, Store
**IDs (deterministic):** vault_id (12), doc_id (24), chunk_id (32) via blake2b
**Link graph:** `[[wikilinks]]`/`![[embeds]]` → `links` table → `vault_neighbors` MCP tool

**Modules:** extractors/, chunking/, embeddings/, retrieval/, store/, indexer/, mcp/
**Entry points:** cli.py, indexer/indexer.py, retrieval/retriever.py, mcp/tools.py, store/libsql_store.py

### Retrieval Pipeline

**Stages:** Lexical (FTS5) + Semantic (vector) → Fusion (RRF default, k=60) → Boosts → Optional Reranking

**Boosts (✅ enabled, ❌ disabled):**
| Signal | Default | Effect | Bias Risk |
|--------|---------|--------|-----------|
| Backlinks | ✅ | +10%/link (max 2x) | None |
| Recency | ✅ | +10% <14d, +5% <60d, -2% >180d | None |
| Heading | ❌ | H1 +5%, H2 +3%, H3 +2% | ⚠️ Favors markdown |
| Tag | ❌ | +3%/tag (max 9%) | ⚠️ Favors markdown |

**Diversity (MMR):** Max 2 chunks/doc (configurable). **Reranking:** `use_rerank = true` for 20-30% quality gain (cross-encoder).
**Config:** `[retrieval]` section. **Code:** retrieval/hybrid.py, boosts.py, reranker.py, retriever.py


### File Lifecycle

**Scan:** Reconciliation (FS vs DB). **Watch:** Filesystem events (Watchdog). Both handle files/dirs (add/change/delete/move).
**Cleanup:** documents/chunks/embeddings/fts_chunks/links/manifest all deleted. **Code:** indexer/indexer.py, change_detector.py, store/libsql_store.py

### Chunk Metadata & Queries

**Metadata fields:** full_path, vault_root/name/id, file_name/extension/size_bytes, modified_at, obsidian_uri
**Query handling:** FTS5 auto-escaped (phrase search), handles special chars (hyphens/slashes). **Code:** indexer/indexer.py:_extract_and_chunk_one()

## Configuration

TOML sections (see `examples/config.toml.example`): `[vault]`, `[index]`, `[chunking]`, `[embeddings]`, `[image_analysis]`, `[retrieval]`, `[indexing]`, `[mcp]`

**Key options:**
- **Chunking:** v2 chunker (overlap_chars=200), max_chunk_size
- **Embeddings:** sentence_transformers/ollama, offline_mode, use_query_prefix (asymmetric), use_faiss (ANN)
- **Image:** tesseract/gemini/gemini-service-account/aigateway/off
- **Retrieval:** RRF/weighted fusion, boosts, diversity, reranking
- **Indexing:** workers, batch sizes, parallel_scan

### Config Details

**Ignore patterns:** `folder/**`, `**/.DS_Store`, `**/~$*` (Office temp files)
**Offline mode:** `offline_mode = true` (no HF downloads, env: `HF_OFFLINE_MODE`)
**Chunk overlap:** `chunker_version = "v2"`, `overlap_chars = 200`
**Query prefix:** `use_query_prefix = true` (asymmetric BGE retrieval)
**FAISS:** `use_faiss = true` for >10K chunks (5-10x speedup, IVF index)
**Reranking:** `use_rerank = true` (20-30% quality gain, CPU ~100-200ms, GPU ~20-50ms)

### Image Providers

AI providers extract: description, visible text (OCR), type, topics, entities

| Provider | Auth | Latency | Use Case |
|----------|------|---------|----------|
| **tesseract** | None | ~100ms | Local OCR, text only |
| **gemini** | API key | ~500ms | Personal, simple setup |
| **gemini-service-account** | Service acct | ~500ms | Enterprise GCP, IAM |
| **aigateway** | Gateway key | ~1000ms | Enterprise Microsoft |
| **off** | N/A | 0ms | Text-only vaults |

**Config:** `[image_analysis]` provider. **Workers critical** for API providers (8-10 for <100 images). **Code:** extractors/image.py

### Image Resilience

**Features:** Timeouts (30s), retry w/ backoff (3x), circuit breaker (5 failures → 60s reset)
**Errors:** 429/5xx retry, 400 skip, 401/403 trip breaker. **Config:** timeout, max_retries, circuit_threshold. **Code:** extractors/resilience.py

### Image Performance

**Session Reuse (v3.6):** Credentials and API clients initialized once per extractor instance, not per image. Critical for performance.

**Model Comparison (400 images):**

| Model | Provider | Total Time | Median Latency | Throughput |
|-------|----------|------------|----------------|------------|
| **gemini-2.5-flash** | Service Account | **4.9 min** | **4.9s** | 82 img/min |
| gemini-2.5-flash | AI Gateway | 6.0 min | 5.1s | 67 img/min |
| gemini-3-pro-preview | Service Account | 15.1 min | 18.0s | 27 img/min |
| gemini-3-pro-preview | AI Gateway | 7.1 min* | 19.4s | 22 img/min |

*Circuit breaker tripped after ~156 images due to auth timeout

**Recommendations:**
- **Use gemini-2.5-flash** for production (3x faster than 3-pro-preview)
- **Service Account faster** than AI Gateway (23% faster, no routing overhead)
- **Session reuse critical** - eliminates ~2s credential loading per image
- **10 workers optimal** for API-bound workloads with 10-20s latencies

**Monitoring:** Set `level = "DEBUG"` in `[logging]` to capture per-image timing: `[provider] filename: SUCCESS - {ms}ms`

## Security & Side Effects

**Security:** config.toml/.gitignore, sensitive info at DEBUG level only
**Side effects:** `offline_mode=true` sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` env vars globally (load config once at startup)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Office temp files** (`~$*.pptx`) | Close Office apps, use ignore pattern `**/~$*` |
| **Offline mode error** | Set `offline_mode=false` to download model, or `HF_OFFLINE_MODE=0 mneme scan` |
| **Pattern syntax** | `**/~$*` ✓, `**~$*` ✗ |

**Testing:** Create `test_config.toml` (in .gitignore), use `--config test_config.toml`
