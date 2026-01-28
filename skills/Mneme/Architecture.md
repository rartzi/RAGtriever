# Architecture

## Data Flow (Parallel Pipeline)

```
Vault (filesystem)
  -> File Discovery (Reconciler)
  -> Phase 1: Parallel Extraction + Chunking (ThreadPoolExecutor)
  -> Phase 2: Batched Embedding (cross-file batches of 256)
  -> Phase 3: Parallel Image Analysis (if enabled)
  -> Store (SQLite + FTS5)
  -> Retrieval (hybrid search)
```

## Image Analysis Pipeline

```
Image file
  -> ImageExtractor (Tesseract/Gemini/Gemini service account)
  -> Structured analysis (description, OCR, topics, entities)
  -> Chunking
  -> Indexing
  -> Searchable via query
```

## Execution Modes

- **CLI**: `mneme scan/query/watch/mcp`
- **Watch Mode**: Continuous filesystem monitoring
- **MCP Server**: Integration with Claude Desktop
- **Python API**: Programmatic import and use

## Parallel Processing (3.6x speedup)

| Pool | Purpose | Default | Bottleneck |
|------|---------|---------|-----------|
| `extraction_workers` | All file extraction (md/pdf/pptx/xlsx/images) | 8 | CPU |
| `image_workers` | API calls (Gemini/Vertex/Gateway) | 8 | Network I/O |
| `embed_batch_size` | Cross-file embedding batches | 256 | GPU |

**Pipeline:** Parallel Extract -> Serial Chunk -> Batch Embed -> Store

Images processed separately via parallel API workers.

**Performance:**
- <50 files: minimal gain
- 50-500 files: 2-3x speedup
- >500 files: 3-4x speedup

## Unified Processing

Both scan/watch use `_process_file()` -> `ProcessResult` (thread-safe, no DB writes).

Watch uses `BatchCollector` (size/timeout triggers).

**Code:** `src/mneme/indexer/indexer.py:_process_file()`, `parallel_types.py`

## Watcher Catch-up

On startup, the watcher detects files modified while stopped by comparing filesystem mtimes against manifest timestamps. Stale files are queued for reprocessing alongside new events (no startup delay).

**Logs:**
- `[watch] Checking for files modified since last index...`
- `[watch] Stale file (modified): path`
- `[watch] New file (not in index): path`
- `[watch] Queued N stale files for reindex (X new, Y modified)`

**Code:** `change_detector.py:queue_stale_files()`, `libsql_store.py:get_manifest_mtimes()`

## Core Design

**Protocols (`**/base.py`):** Extractor, Chunker, Embedder, Store

**IDs (deterministic):** vault_id (12), doc_id (24), chunk_id (32) via blake2b

**Link graph:** `[[wikilinks]]`/`![[embeds]]` -> `links` table -> `vault_neighbors` MCP tool

**Modules:** extractors/, chunking/, embeddings/, retrieval/, store/, indexer/, mcp/

**Entry points:** cli.py, indexer/indexer.py, retrieval/retriever.py, mcp/tools.py, store/libsql_store.py

## Retrieval Pipeline

**Stages:** Lexical (FTS5) + Semantic (vector) -> Fusion (RRF default, k=60) -> Boosts -> Optional Reranking

**Boosts:**

| Signal | Default | Effect | Bias Risk |
|--------|---------|--------|-----------|
| Backlinks | Enabled | +10%/link (max 2x) | None |
| Recency | Enabled | +10% <14d, +5% <60d, -2% >180d | None |
| Heading | Disabled | H1 +5%, H2 +3%, H3 +2% | Favors markdown |
| Tag | Disabled | +3%/tag (max 9%) | Favors markdown |

**Diversity (MMR):** Max 2 chunks/doc (configurable)

**Reranking:** `use_rerank = true` for 20-30% quality gain (cross-encoder)

**Code:** retrieval/hybrid.py, boosts.py, reranker.py, retriever.py

## File Lifecycle

**Scan:** Reconciliation (FS vs DB)

**Watch:** Filesystem events (Watchdog)

Both handle files/dirs (add/change/delete/move).

**Cleanup:** documents/chunks/embeddings/fts_chunks/links/manifest all deleted.

**Code:** indexer/indexer.py, change_detector.py, store/libsql_store.py

## Chunk Metadata & Queries

**Metadata fields:** full_path, vault_root/name/id, file_name/extension/size_bytes, modified_at, obsidian_uri

**Query handling:** FTS5 auto-escaped (phrase search), handles special chars (hyphens/slashes)

**Code:** indexer/indexer.py:_extract_and_chunk_one()
