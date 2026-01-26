# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mneme Local is a local-first vault indexer + hybrid retrieval system with MCP interface. It indexes Obsidian-compatible vaults (Markdown + attachments) into a hybrid search system (semantic + lexical + link-graph) and serves results via CLI, Python API, and MCP tools.

**Key design principles:**
- Local-only: no data leaves the machine
- Filesystem is source of truth; index is derived and can rebuild
- Obsidian-aware parsing (YAML frontmatter, `[[wikilinks]]`, `![[embeds]]`, `#tags`) without Obsidian dependency
- Pluggable adapters via Protocol classes for extractors, chunkers, embedders, and stores

## Migration Guide (v2.0.0)

⚠️ **Breaking Change**: The `vertex_ai` image analysis provider has been renamed to `gemini-service-account` to better reflect the authentication method.

**Update your config.toml:**
```toml
# Before (v0.1.0)
[image_analysis]
provider = "vertex_ai"
[vertex_ai]
project_id = "your-project"

# After (v2.0.0)
[image_analysis]
provider = "gemini-service-account"
[gemini_service_account]
project_id = "your-project"
```

See [CHANGELOG.md](CHANGELOG.md) for complete migration details.

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

### Data Flow
```
Vault (filesystem) → Change Detection → Ingestion Pipeline → Stores → Retrieval Service → Python API / MCP
```

### Parallel Processing Architecture

Mneme uses parallel processing to significantly speed up full vault scans (3.6x speedup tested). **All file types** (markdown, PDFs, PowerPoint, Excel, images) are extracted in parallel, not just images. The architecture uses three types of parallelization:

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
# Scan mode (full vault indexing)
extraction_workers = 10   # Parallel file extraction (CPU-bound)
embed_batch_size = 256    # Cross-file embedding batch size
image_workers = 10        # Parallel image API calls (I/O-bound)
parallel_scan = true      # Enable/disable parallelization

# Watch mode (continuous indexing)
watch_workers = 4         # Parallel extraction workers for watch
watch_batch_size = 10     # Max files per batch before processing
watch_batch_timeout = 5.0 # Seconds before processing partial batch
watch_image_workers = 4   # Parallel image workers for watch
```

**CLI Overrides:**
```bash
# Scan mode
mneme scan --full --workers 10     # Override extraction workers
mneme scan --full --no-parallel    # Disable all parallelization

# Watch mode (batched by default)
mneme watch                        # Batched mode (default)
mneme watch --batch-size 20        # Override batch size
mneme watch --batch-timeout 10     # Override timeout
mneme watch --no-batch             # Legacy serial mode
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

### Unified Processing Pipeline

Both scan and watch modes use a shared `_process_file()` method for file processing. This ensures consistent behavior and simplifies maintenance.

**Core Method:**
```python
def _process_file(self, abs_path: Path) -> ProcessResult:
    """Process a single file: extract, chunk, and prepare for embedding.

    Thread-safe, no DB writes - returns ProcessResult with chunks and image tasks.
    """
```

**ProcessResult Dataclass:**
```python
@dataclass
class ProcessResult:
    abs_path: Path
    rel_path: str
    doc_id: str
    vault_id: str
    file_type: str
    content_hash: str
    mtime: int
    size: int
    chunks: list[ChunkData]      # Ready for embedding
    image_tasks: list[ImageTask]  # Ready for image processing
    links: list[tuple[str, str]]  # (target, link_type)
    error: str | None = None
    skipped: bool = False
    # Enriched metadata...
```

**Processing Flow:**
```
┌─────────────────────────────────────────────────────────────┐
│              Unified _process_file() Method                  │
│                                                             │
│   ┌──────────┐   ┌──────────┐   ┌───────────────────────┐  │
│   │ Validate │──▶│ Extract  │──▶│ Chunk + Build Metadata│  │
│   │ (file?)  │   │ (by type)│   │ + ImageTasks + Links  │  │
│   └──────────┘   └──────────┘   └───────────────────────┘  │
│                                                             │
│   Returns: ProcessResult (no DB writes, thread-safe)        │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────────┐              ┌───────────────────┐
    │   scan_parallel() │              │  watch_batched()  │
    │   - Parallel      │              │  - BatchCollector │
    │     extraction    │              │  - Parallel       │
    │   - Batch embed   │              │    extraction     │
    │   - Batch store   │              │  - Batch embed    │
    └───────────────────┘              └───────────────────┘
```

**Watch Mode Batching:**

The `watch_batched()` method (default since v1.x) uses a `BatchCollector` to accumulate filesystem events:

```python
# BatchCollector accumulates jobs until:
# - Batch reaches watch_batch_size files (default: 10)
# - watch_batch_timeout seconds elapsed (default: 5.0)

collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=5.0)
batch = collector.add_job(job)  # Returns batch when ready
batch = collector.flush_if_timeout()  # Check for timeout trigger
```

**Batch Processing Flow:**
1. Group jobs by type (delete, move, upsert)
2. Process deletes first (remove from index)
3. Process moves (delete old path, add new to upserts)
4. Parallel extraction via `_process_file()` (watch_workers)
5. Batch embed and store
6. Parallel image processing (watch_image_workers)

**Key Benefits:**
- **Single code path** for file processing logic
- **Thread-safe** design (no shared mutable state, no DB writes)
- **Error isolation** (returns error in result, doesn't raise)
- **Deterministic IDs** (same file = same doc_id/chunk_ids)

**Code Reference:**
- `src/mneme/indexer/parallel_types.py`: `ProcessResult`, `ChunkData`, `ImageTask`
- `src/mneme/indexer/indexer.py:_process_file()`: Unified processing method

### Logging for Auditability

Both scan and watch modes emit structured log messages for auditability and debugging.

**Scan Mode Log Messages (INFO level):**
| Log Message | Description |
|-------------|-------------|
| `[scan] Found N files to process` | Discovery complete |
| `[scan] Deleted: path` | File removed from index |
| `[scan] Phase 0: Removed N deleted file(s)` | Reconciliation complete |
| `[scan] Failed: path - error` | Extraction error |
| `[scan] Worker crashed: path - error` | Unexpected exception |
| `[scan] Phase 1: N files extracted, M failed` | Extraction complete |
| `[scan] Phase 2: N chunks, M embeddings` | Embedding complete |
| `[scan] Phase 3: N images processed` | Image analysis complete |
| `[scan] Complete: N files indexed in Xs` | Scan finished |

**Watch Mode Log Messages (INFO level):**
| Log Message | Description |
|-------------|-------------|
| `[watch] Starting file watcher on: path` | Watcher initialized |
| `[watch] File created: path` | New file detected |
| `[watch] File modified: path` | File changed |
| `[watch] File deleted: path` | File removed |
| `[watch] File moved: old -> new` | File renamed/moved |
| `[watch] Batch processed: N files, M chunks` | Batch complete |

**Debug Level:** Additional details like debounce events, ignored files, individual chunk IDs.

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
src/mneme/
├── extractors/    # File parsers (markdown, pdf, pptx, xlsx, image)
├── chunking/      # Text segmentation (heading-aware, boundary markers)
├── embeddings/    # Vector generation (SentenceTransformers, Ollama)
├── retrieval/     # Hybrid search + ranking logic
├── store/         # SQLite persistence layer
├── indexer/       # Orchestration (change detection, reconciliation, queue)
└── mcp/           # Model Context Protocol server + tools
```

### Key Entry Points
- `src/mneme/cli.py`: Typer CLI (`mneme` command) with commands: init, scan, query, watch, open, mcp
- `src/mneme/indexer/indexer.py`: Main `Indexer` class orchestrating extract → chunk → embed → store
- `src/mneme/retrieval/retriever.py`: `Retriever` class for hybrid search (uses `HybridRanker` to merge vector + lexical results)
- `src/mneme/retrieval/reranker.py`: Optional `CrossEncoderReranker` for improving result quality (enabled via `use_rerank = true`)
- `src/mneme/mcp/tools.py`: MCP tool implementations (`vault_search`, `vault_open`, `vault_neighbors`, `vault_status`, `vault_list`)
- `src/mneme/store/libsql_store.py`: SQLite-based storage with FTS5 + vector BLOBs (`vaultrag.sqlite`)
- `src/mneme/store/faiss_index.py`: Optional FAISS index for approximate NN search (enabled via `use_faiss = true`)
- `src/mneme/chunking/markdown_chunker.py`: v2 chunker with overlap support for context preservation

### Hybrid Retrieval & Ranking

Mneme uses a multi-stage ranking pipeline to produce high-quality search results:

**Stage 1: Candidate Retrieval**
- Lexical candidates via SQLite FTS5 (k_lex results)
- Semantic candidates via vector search (k_vec results)

**Stage 2: Fusion (RRF or Weighted)**

Default: Reciprocal Rank Fusion (RRF) - score-agnostic rank-based fusion that handles different score scales automatically.

```
RRF_score = Σ(1 / (k + rank)) where k=60
```

Why RRF:
- Score-agnostic: handles different scales from lexical vs vector
- Well-researched: standard in information retrieval
- No tuning: k=60 works universally

Alternative: Weighted scoring (legacy) with configurable w_vec/w_lex weights.

**Stage 3: Score Boosting**

After fusion, scores can be boosted by document signals. **Content and semantic relevance are primary** - boosts provide subtle signals, not deterministic ranking.

| Signal | Default Status | Effect |
|--------|---------------|--------|
| **Backlinks** | ✅ Enabled | Hub documents rank higher (+10% per link, max 2x) |
| **Recency** | ✅ Enabled | Fresh docs get subtle boost (+10% for <14 days) |
| **Heading Level** | ❌ Disabled | Title/heading chunks rank higher (when enabled) |
| **Tag Matches** | ❌ Disabled | Tagged chunks rank higher (when enabled) |

**Backlink Boost (Enabled):**
Documents with many incoming links (hub documents) receive higher scores, as they're likely to be more important or central to the vault's knowledge graph. This is file-type agnostic and works for all documents.

**Recency Boost (Enabled):**
Recently modified documents receive a subtle boost:
- Fresh (<14 days): 1.10x boost (10%)
- Recent (<60 days): 1.05x boost (5%)
- Standard (<180 days): 1.00x (no change)
- Old (>180 days): 0.98x (2% penalty)

**Heading Boost (DISABLED by default):**
⚠️ **WARNING: Can create file-type bias favoring markdown over PDFs/PPTX**

When enabled, chunks from section titles and headings receive boosts:
- H1 (document title): +5% boost
- H2 (major sections): +3% boost
- H3 (subsections): +2% boost

**When to enable:**
- You want overviews/summaries to rank higher than details
- Your vault is primarily well-structured markdown
- You're okay with markdown ranking higher than authoritative PDFs

**When to DISABLE (default):**
- Policy/compliance questions where authoritative PDFs matter most
- Mixed content (markdown notes + official docs) where file type shouldn't matter
- You want pure semantic relevance without structural bias

**Tag Boost (DISABLED by default):**
⚠️ **WARNING: Can create file-type bias favoring tagged markdown over PDFs/PPTX**

When enabled, chunks whose tags match query terms receive boosts:
- +3% boost per matching tag (up to 3 tags counted, max 9%)
- Tags are normalized (# prefix removed, hyphens/underscores treated as spaces)
- Example: query "machine learning" matches tag `#machine-learning`

**When to enable:**
- Your vault uses consistent, meaningful tags
- Tags indicate topical relevance
- You want tagged notes to rank slightly higher

**When to DISABLE (default):**
- Mixed content where only some files have tags
- Official docs (PDFs) don't have tags but are authoritative
- You want pure semantic relevance without tag bias

**Stage 4: Optional Reranking**

Cross-encoder reranking with `use_rerank = true` for final quality improvement.

**Configuration:**
```toml
[retrieval]
# Fusion
fusion_algorithm = "rrf"  # "rrf" (default) or "weighted"
rrf_k = 60

# Boosts (ENABLED by default - file-type agnostic)
backlink_boost_enabled = true   # Hub documents with many links
backlink_boost_weight = 0.1
backlink_boost_cap = 10

recency_boost_enabled = true    # Recently modified documents
recency_fresh_days = 14
recency_recent_days = 60
recency_old_days = 180

# Boosts (DISABLED by default - can create file-type bias)
heading_boost_enabled = false   # ⚠️ Can favor markdown over PDFs
# heading_h1_boost = 1.05       # 5% boost for H1 (when enabled)
# heading_h2_boost = 1.03       # 3% boost for H2 (when enabled)
# heading_h3_boost = 1.02       # 2% boost for H3 (when enabled)

tag_boost_enabled = false       # ⚠️ Can favor markdown over PDFs
# tag_boost_weight = 0.03       # 3% boost per tag (when enabled)
# tag_boost_cap = 3             # Max 3 tags (when enabled)

# Diversity (limits chunks per document)
diversity_enabled = true
max_per_document = 2

# Reranking (optional)
use_rerank = false
```

**Code Reference:**
- `src/mneme/retrieval/hybrid.py`: RRF and weighted fusion
- `src/mneme/retrieval/boosts.py`: Backlink, recency, heading, and tag boosting
- `src/mneme/retrieval/reranker.py`: Cross-encoder reranking
- `src/mneme/retrieval/retriever.py`: Pipeline orchestration

### Result Diversity (MMR)

Mneme uses Maximal Marginal Relevance (MMR) to ensure diverse search results by limiting the number of chunks returned from the same document. This prevents a single highly-relevant document from dominating all results.

**How It Works:**

1. **Document Limit**: By default, at most 2 chunks per document are returned (configurable via `max_per_document`)
2. **Relevance Preservation**: Results maintain their original relevance ordering within the diversity constraint
3. **Backfill Behavior**: If limiting chunks per document produces fewer than k results, remaining slots are filled from skipped results

**Why MMR:**
- **Broader coverage**: Users see information from multiple documents rather than many chunks from one file
- **Better exploration**: Exposes diverse perspectives and related concepts across the vault
- **Reduced redundancy**: Avoids showing near-duplicate content from the same document

**Example:**

Without MMR (top 10 results):
- Document A: chunks 1, 2, 3, 4, 5
- Document B: chunks 1, 2, 3
- Document C: chunks 1, 2

With MMR (max_per_document=2):
- Document A: chunks 1, 2
- Document B: chunks 1, 2
- Document C: chunks 1, 2
- Document D: chunks 1, 2
- Document E: chunks 1, 2

**Configuration:**

```toml
[retrieval]
diversity_enabled = true       # Enable MMR diversity
max_per_document = 2           # Max chunks from same document
```

**Disabling:**
Set `diversity_enabled = false` to return all top-k results regardless of source document.

**Code Reference:**
- `src/mneme/retrieval/diversity.py`: MMR diversification implementation

### File Lifecycle Management

Mneme properly handles the complete file lifecycle: **add**, **change**, and **delete** operations are detected and processed by both scan and watch modes. This includes both individual files and entire directories.

**Lifecycle Events:**

| Event | Scan Mode | Watch Mode |
|-------|-----------|------------|
| **File Added** | Indexed on next scan | Indexed immediately via filesystem event |
| **File Changed** | Re-indexed on next scan | Re-indexed immediately via filesystem event |
| **File Deleted** | Detected via reconciliation, removed from index | Detected via filesystem event, removed from index |
| **Directory Added** | All files inside indexed on next scan | All files inside indexed immediately |
| **Directory Deleted** | All files inside detected via reconciliation | Queries DB for files under path, removes all from index |
| **Directory Moved** | Reconciliation detects old paths as deleted, new paths as added | Queries DB for files under path, updates all paths atomically |

**How Deletion Detection Works:**

1. **Scan Mode (Reconciliation)**:
   - Compares files on filesystem vs files in database
   - Files in DB but not on filesystem are marked as deleted
   - All related data is cleaned up (chunks, embeddings, FTS, links, manifest)
   - Reports deleted file count in scan output
   - Works for both individual files and entire directories

2. **Watch Mode (Filesystem Events)**:
   - Watchdog library detects filesystem events (create, modify, delete, move)
   - **New folders with files**: When a folder is created/copied with files inside, the watcher scans the directory recursively and queues all files
   - **Deleted folders**: Queries database for all files under the deleted directory path and removes them from the index
   - **Moved folders**: Queries database for all files under the old path and updates their paths to the new location
   - Triggers immediate indexing or deletion as appropriate
   - Same cleanup logic as scan mode

**What Gets Cleaned Up on Deletion:**
- `documents` table: Document marked as `deleted=1`
- `chunks` table: All chunks for document deleted
- `embeddings` table: All embeddings for those chunks deleted
- `fts_chunks` table: All FTS entries deleted
- `links` table: All outgoing links from deleted file removed
- `manifest` table: Indexing metadata removed

**Code References:**
- `src/mneme/indexer/indexer.py:scan()` - Reconciliation logic (lines 119-135)
- `src/mneme/indexer/indexer.py:scan_parallel()` - Parallel reconciliation (lines 166-177)
- `src/mneme/indexer/change_detector.py:on_deleted()` - Directory deletion handler (lines 135-163)
- `src/mneme/indexer/change_detector.py:on_moved()` - Directory move handler (lines 165-227)
- `src/mneme/store/libsql_store.py:delete_document()` - Full cleanup (lines 268-286)
- `src/mneme/store/libsql_store.py:get_indexed_files()` - Get indexed files for reconciliation
- `src/mneme/store/libsql_store.py:get_files_under_path()` - Query files under directory prefix

**CLI Output Example:**
```bash
$ mneme scan
Scan complete: 119 files, 792 chunks in 45.2s
  (3 deleted files removed from index)
```

### Watch Mode Logging

The file watcher provides comprehensive logging for auditability and debugging. Logs are written to the standard Python logging system.

**INFO Level (Auditability):**
| Event | Log Message |
|-------|-------------|
| Watcher start | `Starting file watcher on: /path/to/vault` |
| Ignore patterns | `Ignore patterns: ['.git/**', '**/.DS_Store']` |
| Watcher ready | `File watcher started - monitoring for changes` |
| Directory created | `Directory created: /path/to/new_folder` |
| Directory scan complete | `Directory scan complete: new_folder - queued 5 files, ignored 1` |
| Directory deleted | `Directory deleted: folder_path - queued 10 files for deletion` |
| Directory moved | `Directory moved: old_path -> new_path - queued 10 files` |
| File created | `File created: documents/notes.md` |
| File modified | `File modified: documents/notes.md` |
| File deleted | `File deleted: old_file.txt` |
| File moved | `File moved: draft.md -> published/final.md` |
| Move from ignored | `File moved from ignored location (treating as create): file.md` |
| Move to ignored | `File moved to ignored location (treating as delete): file.md` |
| Watcher stop | `Stopping file watcher...` / `File watcher stopped` |

**DEBUG Level (Debugging):**
- Debounce interval configuration
- Debounced events (rapid duplicate events filtered)
- Ignored files (matching patterns)
- Individual files queued from new directories
- Empty directory events (no indexed files found)

**Code Reference:**
- `src/mneme/indexer/change_detector.py`: ChangeDetector class with Handler inner class

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
mneme query "search term" --k 1 | jq -r '.[0].metadata.full_path'

# Open file directly without database lookup
open "$(mneme query 'search term' --k 1 | jq -r '.[0].metadata.full_path')"
```

**Code reference:**
- `src/mneme/indexer/indexer.py:_extract_and_chunk_one()`: Metadata assembly (lines 272-312)

### Query Handling
Search queries are automatically escaped for FTS5 to handle special characters (hyphens, slashes, etc.) in technical and medical terms. Queries are treated as phrase searches wrapped in double quotes, ensuring terms like "T-DXd", "CDK4/6i", or "HR+/HER2-low" work correctly without FTS5 syntax errors.

## Configuration

TOML-based config (see `examples/config.toml.example`):
- `[vault]`: root path, ignore patterns
- `[index]`: index directory, extractor_version, chunker_version (v2 adds overlap support)
- `[chunking]`: overlap_chars (default: 200), max_chunk_size, preserve_heading_metadata
- `[embeddings]`: provider (sentence_transformers/ollama), model, device (cpu/cuda/mps), batch_size, offline_mode (default: true), use_query_prefix (asymmetric retrieval), use_faiss (approximate NN search)
- `[image_analysis]`: provider (tesseract/gemini/gemini-service-account/aigateway/off), gemini_model for Gemini API
- `[gemini_service_account]`: project_id, location, credentials_file, model (for Gemini with GCP service account auth)
- `[aigateway]`: url, key, model, timeout, endpoint_path (for Microsoft AI Gateway proxy to Gemini)
- `[retrieval]`: k_vec, k_lex, top_k, use_rerank, fusion_algorithm (rrf/weighted), boost settings (backlinks, recency, heading, tag), diversity_enabled, max_per_document
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
mneme query "kubernetes deployment" --k 10 --rerank
```

**Code reference:**
- `src/mneme/retrieval/reranker.py`: CrossEncoderReranker implementation
- `src/mneme/retrieval/retriever.py:search()`: Integration point (line 66-67)

### Parallel Scanning (3.6x Speedup)
Parallel scanning is enabled by default for faster full vault scans. Configure in `[indexing]`:
- `extraction_workers = 8`: Parallel file extraction workers (default: 8)
- `embed_batch_size = 256`: Cross-file embedding batch size
- `image_workers = 8`: Parallel image API workers (default: 8)
- `parallel_scan = true`: Enable/disable parallel scanning

**CLI overrides:**
```bash
mneme scan --full --workers 8     # Use 8 extraction workers
mneme scan --full --no-parallel   # Disable parallelization
```

**Tested performance:** 337s → 93s (3.6x speedup) on 143-file vault with images.

### Image Analysis Providers

Mneme supports multiple image analysis providers, each with different capabilities and use cases. All AI-powered providers (gemini, gemini-service-account, aigateway) extract structured metadata including:
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

#### 3. Gemini with Service Account
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
provider = "gemini-service-account"

[gemini_service_account]
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
Mneme → AI Gateway (Microsoft) → /{endpoint_path} → Gemini API
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
| **gemini-service-account** | Service account | GCP region | ~500ms | Critical | Enterprise GCP, IAM integration |
| **aigateway** | Gateway key | Microsoft + GCP | ~1000ms | Critical | Enterprise Microsoft, governance |
| **off** | N/A | N/A | 0ms | N/A | Text-only, cost savings |

**Recommendations:**
- **Small vaults (<20 images)**: Any provider works, latency not critical
- **Medium vaults (20-100 images)**: Use 8-10 image workers for any API provider
- **Large vaults (>100 images)**: Maximize image workers (10-20), consider costs
- **Offline/Privacy**: Use tesseract (local only)
- **Enterprise**: Use gemini-service-account (GCP) or aigateway (Microsoft) for compliance/governance

### Image Analysis Resilience

All API-based image providers (gemini, gemini-service-account, aigateway) include built-in resilience features to handle transient failures gracefully without hanging or crashing scans.

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
provider = "aigateway"  # or gemini, gemini-service-account
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
- `src/mneme/extractors/resilience.py`: Core resilience module (ErrorCategory, CircuitBreaker, ResilientClient)
- `src/mneme/extractors/image.py`: Integration with image extractors

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
4. Alternative: Use environment variable: `HF_OFFLINE_MODE=0 mneme scan`

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
mneme init --vault "/path/to/test/vault" --index "~/.mneme/indexes/test" --out test_config.toml
mneme scan --config test_config.toml --full
mneme query --config test_config.toml "test query"
```
