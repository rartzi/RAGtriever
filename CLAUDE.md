# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGtriever Local is a local-first vault indexer + hybrid retrieval system with MCP interface. It indexes Obsidian-compatible vaults (Markdown + attachments) into a hybrid search system (semantic + lexical + link-graph) and serves results via CLI, Python API, and MCP tools.

**Key design principles:**
- Local-only: no data leaves the machine
- Filesystem is source of truth; index is derived and can rebuild
- Obsidian-aware parsing (YAML frontmatter, `[[wikilinks]]`, `![[embeds]]`, `#tags`) without Obsidian dependency
- Pluggable adapters via Protocol classes for extractors, chunkers, embedders, and stores

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
ragtriever init --vault "/path/to/vault" --index "~/.ragtriever/indexes/myvault"
ragtriever scan --full
ragtriever query "search term" --k 10
ragtriever query "search term" --k 10 --rerank  # with cross-encoder reranking
ragtriever watch              # watch mode for continuous indexing
ragtriever mcp                # MCP server over stdio
```

## Architecture

### Data Flow
```
Vault (filesystem) → Change Detection → Ingestion Pipeline → Stores → Retrieval Service → Python API / MCP
```

### Parallel Processing Architecture

RAGtriever uses parallel processing to significantly speed up full vault scans (3.6x speedup tested). **All file types** (markdown, PDFs, PowerPoint, Excel, images) are extracted in parallel, not just images. The architecture uses three types of parallelization:

**Worker Pools:**
1. **Extraction Workers** (`extraction_workers`): Parallel file extraction **for ALL file types**
   - Each worker handles file reading and content extraction
   - Processes markdown, PDFs, PowerPoint, Excel, images in parallel
   - **NOT limited to images** - all files are extracted in parallel
   - Default: 8 workers
   - Bottleneck: CPU-bound (parsing, text extraction)

2. **Image Analysis Workers** (`image_workers`): Parallel API calls
   - Dedicated pool for external image analysis APIs
   - Handles Gemini, Vertex AI, AI Gateway requests concurrently
   - Default: 8 workers
   - Bottleneck: Network I/O (API latency)

3. **Cross-File Embedding Batches** (`embed_batch_size`): GPU efficiency
   - Collects chunks from multiple files before embedding
   - Larger batches = better GPU utilization
   - Default: 256 chunks
   - Bottleneck: GPU throughput

**Processing Flow (All File Types):**
```
┌─────────────────────────────────────────────────────────────┐
│         Parallel Extraction (Markdown, PDF, PPTX, etc.)      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │   │
│  │ file.md  │  │ doc.pdf  │  │ pres.pptx│  │ sheet.xlsx│  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │   Chunking (serial)     │
        │   Per-file processing   │
        └─────────┬───────────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │  Cross-File Batching    │
        │  Collect 256 chunks     │
        └─────────┬───────────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │   Embedding (batch)     │
        │   GPU-accelerated       │
        └─────────┬───────────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │    SQLite Storage       │
        │    (transaction batch)  │
        └─────────────────────────┘
```

**Image Analysis (Separate Parallel Pool for API Calls):**
```
┌────────────────────────────────────────────────────────────┐
│       Parallel Image Analysis Workers (API-bound)          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker N │     │
│  │img1.png │  │img2.jpg │  │img3.png │  │img4.jpg │     │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │
└───────┼────────────┼────────────┼────────────┼───────────┘
        │            │            │            │
        └────────────┴────────────┴────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     ▼               ▼               ▼
┌─────────┐    ┌──────────┐    ┌──────────┐
│ Gemini  │    │Vertex AI │    │AI Gateway│
│   API   │    │   API    │    │   API    │
└─────────┘    └──────────┘    └──────────┘
```

**Configuration:**
```toml
[indexing]
extraction_workers = 10   # Parallel file extraction (CPU-bound)
embed_batch_size = 256    # Cross-file embedding batch size
image_workers = 10        # Parallel image API calls (I/O-bound)
parallel_scan = true      # Enable/disable parallelization
```

**CLI Overrides:**
```bash
ragtriever scan --full --workers 10     # Override extraction workers
ragtriever scan --full --no-parallel    # Disable all parallelization
```

**Performance Characteristics:**
- **Small vaults (<50 files)**: Overhead dominates, minimal benefit
- **Medium vaults (50-500 files)**: 2-3x speedup with 8-10 workers
- **Large vaults (>500 files)**: 3-4x speedup, scales with worker count
- **With images**: Image workers critical for API-bound providers (Gemini, Vertex AI, AI Gateway)

**Example Performance (119 files, 16 images):**
- Sequential: ~180s estimated
- Parallel (10 workers): 49s actual
- Speedup: **3.7x**

### Core Protocols (in `**/base.py`)
The codebase uses Python Protocol classes for pluggable adapters:
- `Extractor`: file content extraction (one per file type: `.md`, `.pdf`, `.pptx`, `.xlsx`, images)
- `Chunker`: text segmentation strategy (heading-aware for markdown, boundary markers for documents)
- `Embedder`: vector embedding provider (SentenceTransformers local, Ollama server)
- `Store`: storage layer with lexical (FTS5) and vector search

### ID Generation (deterministic for idempotent updates)
- **vault_id**: `blake2b(vault_root_path)[:12]`
- **doc_id**: `blake2b("{vault_id}:{rel_path}")[:24]`
- **chunk_id**: `blake2b("{doc_id}:{anchor_type}:{anchor_ref}:{text_hash}")[:32]`

### Link Graph
`[[wikilinks]]` and `![[embeds]]` are extracted during indexing and stored in a `links` table for graph traversal (`vault_neighbors` MCP tool).

### Module Organization
```
src/ragtriever/
├── extractors/    # File parsers (markdown, pdf, pptx, xlsx, image)
├── chunking/      # Text segmentation (heading-aware, boundary markers)
├── embeddings/    # Vector generation (SentenceTransformers, Ollama)
├── retrieval/     # Hybrid search + ranking logic
├── store/         # SQLite persistence layer
├── indexer/       # Orchestration (change detection, reconciliation, queue)
└── mcp/           # Model Context Protocol server + tools
```

### Key Entry Points
- `src/ragtriever/cli.py`: Typer CLI (`ragtriever` command) with commands: init, scan, query, watch, open, mcp
- `src/ragtriever/indexer/indexer.py`: Main `Indexer` class orchestrating extract → chunk → embed → store
- `src/ragtriever/retrieval/retriever.py`: `Retriever` class for hybrid search (uses `HybridRanker` to merge vector + lexical results)
- `src/ragtriever/retrieval/reranker.py`: Optional `CrossEncoderReranker` for improving result quality (enabled via `use_rerank = true`)
- `src/ragtriever/mcp/tools.py`: MCP tool implementations (`vault_search`, `vault_open`, `vault_neighbors`, `vault_status`, `vault_list`)
- `src/ragtriever/store/libsql_store.py`: SQLite-based storage with FTS5 + vector BLOBs (`vaultrag.sqlite`)
- `src/ragtriever/store/faiss_index.py`: Optional FAISS index for approximate NN search (enabled via `use_faiss = true`)
- `src/ragtriever/chunking/markdown_chunker.py`: v2 chunker with overlap support for context preservation

### Hybrid Retrieval Strategy
1. Lexical candidates via SQLite FTS5
2. Semantic candidates via vector search
3. Merge + dedupe with configurable weights
4. Optional graph boost (backlinks) and rerank

### File Lifecycle Management

RAGtriever properly handles the complete file lifecycle: **add**, **change**, and **delete** operations are detected and processed by both scan and watch modes.

**Lifecycle Events:**

| Event | Scan Mode | Watch Mode |
|-------|-----------|------------|
| **File Added** | Indexed on next scan | Indexed immediately via filesystem event |
| **File Changed** | Re-indexed on next scan | Re-indexed immediately via filesystem event |
| **File Deleted** | Detected via reconciliation, removed from index | Detected via filesystem event, removed from index |

**How Deletion Detection Works:**

1. **Scan Mode (Reconciliation)**:
   - Compares files on filesystem vs files in database
   - Files in DB but not on filesystem are marked as deleted
   - All related data is cleaned up (chunks, embeddings, FTS, links, manifest)
   - Reports deleted file count in scan output

2. **Watch Mode (Filesystem Events)**:
   - Watchdog library detects `FileDeleted` events
   - Triggers immediate `delete_document()` call
   - Same cleanup logic as scan mode

**What Gets Cleaned Up on Deletion:**
- `documents` table: Document marked as `deleted=1`
- `chunks` table: All chunks for document deleted
- `embeddings` table: All embeddings for those chunks deleted
- `fts_chunks` table: All FTS entries deleted
- `links` table: All outgoing links from deleted file removed
- `manifest` table: Indexing metadata removed

**Code References:**
- `src/ragtriever/indexer/indexer.py:scan()` - Reconciliation logic (lines 119-135)
- `src/ragtriever/indexer/indexer.py:scan_parallel()` - Parallel reconciliation (lines 166-177)
- `src/ragtriever/store/libsql_store.py:delete_document()` - Full cleanup (lines 268-286)
- `src/ragtriever/store/libsql_store.py:get_indexed_files()` - Get indexed files for reconciliation

**CLI Output Example:**
```bash
$ ragtriever scan
Scan complete: 119 files, 792 chunks in 45.2s
  (3 deleted files removed from index)
```

### Enriched Chunk Metadata
Every indexed chunk includes enriched metadata for fast operations without additional database lookups:

| Field | Description | Example |
|-------|-------------|---------|
| `full_path` | Absolute filesystem path | `/Volumes/.../file.md` |
| `vault_root` | Root path of containing vault | `/Volumes/.../my-vault` |
| `vault_name` | Human-readable vault name | `my-thoughts` |
| `vault_id` | Hash identifier for vault | `ba1c6c00379e` |
| `file_name` | Filename only | `notes.md` |
| `file_extension` | File extension | `.md` |
| `file_size_bytes` | File size in bytes | `9488` |
| `modified_at` | ISO 8601 timestamp | `2025-01-19T20:13:34+00:00` |
| `obsidian_uri` | Direct Obsidian app link | `obsidian://open?vault=...` |

**Usage example:**
```bash
# Get full_path directly from query results
ragtriever query "search term" --k 1 | jq -r '.[0].metadata.full_path'

# Open file directly without database lookup
open "$(ragtriever query 'search term' --k 1 | jq -r '.[0].metadata.full_path')"
```

**Code reference:**
- `src/ragtriever/indexer/indexer.py:_extract_and_chunk_one()`: Metadata assembly (lines 272-312)

### Query Handling
Search queries are automatically escaped for FTS5 to handle special characters (hyphens, slashes, etc.) in technical and medical terms. Queries are treated as phrase searches wrapped in double quotes, ensuring terms like "T-DXd", "CDK4/6i", or "HR+/HER2-low" work correctly without FTS5 syntax errors.

## Configuration

TOML-based config (see `examples/config.toml.example`):
- `[vault]`: root path, ignore patterns
- `[index]`: index directory, extractor_version, chunker_version (v2 adds overlap support)
- `[chunking]`: overlap_chars (default: 200), max_chunk_size, preserve_heading_metadata
- `[embeddings]`: provider (sentence_transformers/ollama), model, device (cpu/cuda/mps), batch_size, offline_mode (default: true), use_query_prefix (asymmetric retrieval), use_faiss (approximate NN search)
- `[image_analysis]`: provider (tesseract/gemini/vertex_ai/aigateway/off), gemini_model for Gemini API
- `[vertex_ai]`: project_id, location, credentials_file, model (for Vertex AI with service account auth)
- `[aigateway]`: url, key, model, timeout, endpoint_path (for Microsoft AI Gateway proxy to Gemini)
- `[retrieval]`: k_vec, k_lex, top_k, use_rerank
- `[indexing]`: extraction_workers, embed_batch_size, image_workers, parallel_scan (parallelization settings)
- `[mcp]`: transport (stdio)

### Ignore Patterns
Each vault can specify ignore patterns to exclude files and folders from indexing. Patterns are applied to both full scans and watch mode.

**Supported patterns:**
- `"folder/**"` - ignore entire folder and all contents (e.g., `"00-Input/**"`)
- `"**/.DS_Store"` - ignore file in any directory
- `"**/~$*"` - ignore files matching pattern in any directory

**Example config (multi-vault):**
```toml
[[vaults]]
name = "my-vault"
root = "/path/to/vault"
ignore = [
    ".git/**",
    ".obsidian/cache/**",
    "**/.DS_Store",
    "00-Input/**",      # Staging folder - not indexed
    "99-Archive/**",    # Archive folder - not indexed
    "drafts/**"         # Work in progress - not indexed
]
```

Multiple folders can be ignored per vault. Files moved from ignored folders to non-ignored folders will be indexed; files moved to ignored folders will be removed from the index.

### Offline Mode
Set `offline_mode = true` in `[embeddings]` to use cached models only (no HuggingFace downloads). This is useful in corporate environments with restricted internet access. Can be overridden with the `HF_OFFLINE_MODE` environment variable.

### Chunk Overlap (v2 Chunker)
Set `chunker_version = "v2"` in `[index]` to enable chunk overlap. This preserves context between adjacent chunks by including the last 200 characters (configurable) of the previous chunk in the next chunk. This improves retrieval quality when relevant information spans chunk boundaries.

### Query Instruction Prefix
Set `use_query_prefix = true` in `[embeddings]` to enable asymmetric retrieval for BGE models. This adds a prefix like "Represent this sentence for searching relevant passages: " to queries (but not documents), improving relevance by making the model aware that the query is a search task rather than a passage to be indexed.

### FAISS Approximate Nearest Neighbor Search
For large vaults (>10K chunks), set `use_faiss = true` in `[embeddings]` to enable approximate nearest neighbor search via FAISS. This provides 5-10x speedup (~10-20ms vs 50-100ms+) for vector search at the cost of slight accuracy loss. Requires `faiss-cpu` or `faiss-gpu` installed.

Recommended settings:
- `faiss_index_type = "IVF"` (inverted file index, good for 10K-1M vectors)
- `faiss_nlist = 100` (number of clusters)
- `faiss_nprobe = 10` (clusters to search, higher = more accurate but slower)

### Cross-Encoder Reranking
Set `use_rerank = true` in `[retrieval]` to enable cross-encoder reranking. This improves result quality by 20-30% by reranking top candidates with a model that reads query + document together (cross-encoder) instead of separately (bi-encoder).

**Benefits:**
- Reduces false positives from partial term matches
- Improves result ordering (relevant docs move to top 3)
- Especially helpful for complex queries with multiple concepts

**Performance:**
- CPU: ~100-200ms for 40 candidates
- GPU: ~20-50ms for 40 candidates

**Recommended settings:**
- `rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"` (balanced speed/accuracy)
- `rerank_device = "cpu"` (or "cuda", "mps" for GPU acceleration)
- `rerank_top_k = 10` (number of results to return)

**Example:**
```bash
# Test reranking on a specific query
ragtriever query "kubernetes deployment" --k 10 --rerank
```

**Code reference:**
- `src/ragtriever/retrieval/reranker.py`: CrossEncoderReranker implementation
- `src/ragtriever/retrieval/retriever.py:search()`: Integration point (line 66-67)

### Parallel Scanning (3.6x Speedup)
Parallel scanning is enabled by default for faster full vault scans. Configure in `[indexing]`:
- `extraction_workers = 8`: Parallel file extraction workers (default: 8)
- `embed_batch_size = 256`: Cross-file embedding batch size
- `image_workers = 8`: Parallel image API workers (default: 8)
- `parallel_scan = true`: Enable/disable parallel scanning

**CLI overrides:**
```bash
ragtriever scan --full --workers 8     # Use 8 extraction workers
ragtriever scan --full --no-parallel   # Disable parallelization
```

**Tested performance:** 337s → 93s (3.6x speedup) on 143-file vault with images.

### Image Analysis Providers

RAGtriever supports multiple image analysis providers, each with different capabilities and use cases. All AI-powered providers (gemini, vertex_ai, aigateway) extract structured metadata including:
- **Description**: Detailed 2-3 sentence description of image content
- **Visible Text**: OCR transcription of any text in the image
- **Image Type**: Classification (screenshot, diagram, flowchart, photo, presentation_slide, document, chart, infographic, logo, ui_mockup, other)
- **Topics**: 3-5 key topics or themes
- **Entities**: Named entities (people, companies, products, technologies)

---

#### 1. Tesseract (Local OCR)
**Use case**: Offline OCR, privacy-sensitive documents, no API costs

**Features:**
- Local pytesseract-based OCR
- No external API calls
- Extracts visible text only (no semantic analysis)
- Fast for text-heavy images

**Requirements:**
- `pip install pytesseract`
- System installation: `brew install tesseract-ocr` (macOS) or `apt-get install tesseract-ocr` (Linux)

**Configuration:**
```toml
[image_analysis]
provider = "tesseract"
```

**Limitations:**
- Text extraction only (no image understanding)
- Accuracy depends on image quality
- No structured metadata (description, topics, entities)

---

#### 2. Google Gemini API (Direct)
**Use case**: Personal projects, direct API access, simple auth

**Features:**
- Full vision + language understanding
- Structured metadata extraction
- API key authentication (simple setup)
- Gemini 2.0/2.5 Flash models

**Requirements:**
- `pip install google-genai`
- API key from Google AI Studio

**Configuration:**
```toml
[image_analysis]
provider = "gemini"
gemini_model = "gemini-2.0-flash"  # or "gemini-2.5-flash"

# Option 1: Environment variable (recommended)
# export GEMINI_API_KEY="your-api-key"

# Option 2: Config file (not recommended for security)
# gemini_api_key = "your-api-key"
```

**Performance:**
- Latency: ~500-1000ms per image
- Parallel workers critical for large vaults
- Rate limits: depends on API tier

**Cost:** Pay-per-use (check Google AI pricing)

---

#### 3. Google Vertex AI (Service Account)
**Use case**: Enterprise GCP deployments, service account auth, fine-grained IAM

**Features:**
- Same Gemini models as direct API
- Service account JSON authentication
- GCP IAM integration
- Regional deployment options
- Enterprise SLAs

**Requirements:**
- `pip install google-cloud-aiplatform google-auth`
- GCP project with Vertex AI enabled
- Service account JSON credentials

**Configuration:**
```toml
[image_analysis]
provider = "vertex_ai"

[vertex_ai]
project_id = "your-gcp-project-id"  # or set GOOGLE_CLOUD_PROJECT env var
location = "global"  # or "us-central1", "us-east4", etc.
credentials_file = "/path/to/service-account.json"  # or set GOOGLE_APPLICATION_CREDENTIALS
model = "gemini-2.0-flash-exp"  # regional model availability varies
```

**Performance:**
- Similar to Gemini API (~500-1000ms)
- Benefits from GCP network proximity
- Parallel workers recommended

**Security:**
- Service account credentials (more secure than API keys)
- GCP audit logging
- VPC-SC support for enterprise

**Cost:** GCP Vertex AI pricing (typically higher than direct API)

---

#### 4. Microsoft AI Gateway (Enterprise Proxy)
**Use case**: Enterprise Microsoft shops, centralized governance, routing/monitoring

**Features:**
- Proxies to Google Gemini via Microsoft infrastructure
- Enterprise authentication (API keys managed by gateway)
- Centralized usage tracking and billing
- Access control and rate limiting at gateway level
- Same structured metadata as direct Gemini

**Requirements:**
- `pip install google-genai` (uses same SDK as Gemini)
- AI Gateway URL and API key from Microsoft

**Configuration:**
```toml
[image_analysis]
provider = "aigateway"

[aigateway]
url = "https://your-gateway.azure.com"  # or set AI_GATEWAY_URL env var
key = "your-api-key"  # or set AI_GATEWAY_KEY env var
model = "gemini-2.5-flash"
timeout = 60000  # Request timeout in milliseconds (default: 60s)
endpoint_path = "vertex-ai-express"  # Path suffix appended to URL (default: vertex-ai-express)
```

**Architecture:**
```
RAGtriever → AI Gateway (Microsoft) → /{endpoint_path} → Gemini API
```

**Performance:**
- Latency: ~500-1500ms per image (additional gateway hop)
- Parallel workers critical for throughput
- Example: 16 images in 49s with 10 workers

**Benefits:**
- Centralized governance (audit, compliance)
- Simplified credential management
- Cross-cloud routing (Azure → GCP)
- Enterprise support channels

**Cost:** Gateway pricing + underlying Gemini API costs

---

#### 5. Off (Disabled)
**Use case**: Text-only vaults, cost reduction, privacy

**Configuration:**
```toml
[image_analysis]
provider = "off"
```

**Effect:**
- Images are skipped during indexing
- No image metadata in search results
- Faster scans (no API calls)

---

### Provider Comparison

| Provider | Auth | Location | Latency | Parallel Workers | Use Case |
|----------|------|----------|---------|------------------|----------|
| **tesseract** | None | Local | ~100ms | Not critical | Offline OCR, text extraction only |
| **gemini** | API key | Google Cloud | ~500ms | Critical | Personal projects, simple setup |
| **vertex_ai** | Service account | GCP region | ~500ms | Critical | Enterprise GCP, IAM integration |
| **aigateway** | Gateway key | Microsoft + GCP | ~1000ms | Critical | Enterprise Microsoft, governance |
| **off** | N/A | N/A | 0ms | N/A | Text-only, cost savings |

**Recommendations:**
- **Small vaults (<20 images)**: Any provider works, latency not critical
- **Medium vaults (20-100 images)**: Use 8-10 image workers for any API provider
- **Large vaults (>100 images)**: Maximize image workers (10-20), consider costs
- **Offline/Privacy**: Use tesseract (local only)
- **Enterprise**: Use vertex_ai (GCP) or aigateway (Microsoft) for compliance/governance

### Image Analysis Resilience

All API-based image providers (gemini, vertex_ai, aigateway) include built-in resilience features to handle transient failures gracefully without hanging or crashing scans.

**Features:**
- **Configurable timeouts**: Prevent hanging on unresponsive APIs (default: 30s)
- **Retry with exponential backoff**: Automatically retry transient errors (1s, 2s, 4s)
- **Circuit breaker**: Stop hammering broken APIs after consecutive failures
- **Smart error classification**: Different handling for different error types

**Error Classification:**

| Error Type | Behavior | Trips Circuit Breaker |
|------------|----------|----------------------|
| 429 Rate Limit | Retry with backoff, respect `Retry-After` header | No |
| 500/502/503/504 Server Error | Retry up to 3x | Yes (after threshold) |
| Connection/Timeout | Retry up to 3x | Yes (after threshold) |
| 400 Bad Request | Skip image (invalid) | No |
| 401/403 Auth Error | Skip immediately | Yes (immediately) |
| JSON Parse Error | Skip image (LLM fluke) | No |

**Circuit Breaker:**
- Trips after 5 consecutive transient failures (configurable)
- Auto-resets after 60 seconds (configurable)
- Thread-safe for parallel workers
- Auth failures (401/403) trip immediately

**Configuration:**
```toml
[image_analysis]
provider = "aigateway"  # or gemini, vertex_ai
timeout = 60000              # 60s timeout (ms)
max_retries = 3              # Retry transient errors up to 3x
retry_backoff = 1000         # Base backoff 1s, doubles each retry
circuit_threshold = 5        # Trip breaker after 5 consecutive failures
circuit_reset = 60           # Auto-reset breaker after 60s
```

**Logging:**
- Retry attempts logged at INFO level
- Circuit breaker state changes logged at WARNING level
- All errors include source file path and provider name for debugging

**Code Reference:**
- `src/ragtriever/extractors/resilience.py`: Core resilience module (ErrorCategory, CircuitBreaker, ResilientClient)
- `src/ragtriever/extractors/image.py`: Integration with image extractors

## Security Notes

- `config.toml` and credential files are in `.gitignore` - never commit them
- Config validation checks credential file existence, numeric ranges, and allowed device values
- Sensitive info (credentials paths, project IDs) only logged at DEBUG level

### Side Effects
⚠️ **Important**: Configuration loading has side effects:
- Sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` environment variables globally if `offline_mode=true`
- This affects the entire Python process
- Config should be loaded once at application startup

## Common Issues and Troubleshooting

### Office Temp Files
**Problem:** PowerPoint/Word/Excel creates lock files (e.g., `~$document.pptx`) when documents are open. These are not valid Office files and will cause extraction errors.

**Solution:**
- Close all Office applications before scanning
- The default ignore patterns now exclude these: `**/~$*`, `**/.~lock.*`

**Error message:**
```
PackageNotFoundError: Package not found at '.../~$document.pptx'
```

### Offline Mode with Uncached Models
**Problem:** Config has `offline_mode = true` but the embedding model isn't downloaded locally.

**Solution:**
1. First run with `offline_mode = false` to download the model:
   ```toml
   offline_mode = false
   ```
2. Run scan to download model
3. Switch back to `offline_mode = true`
4. Alternative: Use environment variable: `HF_OFFLINE_MODE=0 ragtriever scan`

**Check cached models:**
```bash
ls ~/.cache/huggingface/hub/
```

**Error message:**
```
OSError: We couldn't connect to 'https://huggingface.co' to load the files
```

### Ignore Pattern Syntax
**Problem:** Glob patterns not matching files correctly.

**Correct patterns:**
- `**/~$*` - Office temp files in any subdirectory (correct)
- `**~$*` - Won't match properly (incorrect)
- `**/.DS_Store` - macOS metadata files (correct)

**Test patterns:**
```bash
find /path/to/vault -name "~$*"  # Check what matches
```

## Testing During Development

To test changes against a real vault, create a `test_config.toml` (already in `.gitignore`):

```bash
ragtriever init --vault "/path/to/test/vault" --index "~/.ragtriever/indexes/test" --out test_config.toml
ragtriever scan --config test_config.toml --full
ragtriever query --config test_config.toml "test query"
```
