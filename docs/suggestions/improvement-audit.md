# Mneme Improvement Audit

**Date:** 2026-02-06
**Scope:** Full codebase review across scale, performance, quality, and architecture
**Source:** `skills/Mneme/source/src/mneme/` (~7,674 LOC source, ~8,895 LOC tests)

---

## Executive Summary

Mneme is a well-architected local-first RAG indexing system that is **production-ready for personal vaults up to ~50K chunks**. The codebase demonstrates clean Protocol-based abstractions, a solid resilience layer, and good test coverage (1.16:1 test-to-source ratio).

However, four areas present significant improvement opportunities:

| Area | Current Rating | Key Issue |
|------|---------------|-----------|
| **Scale** | 50K chunk ceiling | O(n) brute-force search, in-memory buffering |
| **Performance** | Good for small vaults | Serial DB writes, model reload per query, FAISS save too frequent |
| **Quality** | Good with caveats | Thread safety gaps, no transaction boundaries, edge case gaps |
| **Architecture** | 7/10 | Config duplication, no schema migration, weak DI |

**Total issues identified:** 50+ across all categories
**Estimated improvement effort:** ~6-10 weeks for comprehensive fixes

---

## Table of Contents

1. [Critical Issues (Fix First)](#1-critical-issues-fix-first)
2. [Scale Improvements](#2-scale-improvements)
3. [Performance Optimizations](#3-performance-optimizations)
4. [Quality and Reliability](#4-quality-and-reliability)
5. [Architecture and Extensibility](#5-architecture-and-extensibility)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. Critical Issues (Fix First)

These issues can cause data corruption or loss under real-world usage.

### 1.1 SQLite Connection Not Thread-Safe

**File:** `store/libsql_store.py:145`
**Severity:** CRITICAL

```python
self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
```

Single connection shared across extraction workers, watcher threads, and retriever. `sqlite3.Connection` is NOT thread-safe for concurrent writes. Parallel scan with 8 workers + watcher can corrupt the database.

**Fix:** Use thread-local connections via `threading.local()` or a connection pool with `queue.Queue`.

### 1.2 FAISS Index Not Thread-Safe

**File:** `store/libsql_store.py:358-365`
**Severity:** CRITICAL

FAISS `add()` and `save()` operations have no thread synchronization. Parallel extraction workers calling `add()` concurrently corrupt the vector index.

**Fix:** Wrap all FAISS operations with `threading.Lock()`.

### 1.3 No Transaction Boundaries in Scan Phases

**File:** `indexer/indexer.py:249-348`
**Severity:** CRITICAL

Scan has 4 phases (delete, extract, embed, image). If interrupted between phases, the index is left in an inconsistent state:
- Crash after Phase 0 (deletions) before Phase 1: documents deleted but not re-indexed
- Crash after Phase 1 before Phase 2: chunks in DB but no embeddings

**Fix:** Wrap scan phases in `BEGIN TRANSACTION...COMMIT` with checkpoint tracking.

### 1.4 Manifest Out of Sync with Index

**File:** `store/libsql_store.py:236-247` + `indexer/indexer.py`
**Severity:** CRITICAL

Document upsert and manifest update happen in separate commits. Crash between them leaves manifest stale, causing unnecessary re-processing on next scan.

**Fix:** Upsert document + manifest in a single transaction.

---

## 2. Scale Improvements

### Current Scale Ceiling

| Vault Size | Status | Primary Bottleneck |
|-----------|--------|-------------------|
| <10K chunks | Works well | None |
| 10K-50K | Usable | Brute-force vector search ~200-500ms |
| 50K-100K | Slow | Vector search 500ms-2s, memory 1-2GB |
| 100K-500K | Problematic | Vector search 5-15s, memory 2-4GB |
| 500K+ | Unusable | Vector search 20-60s+, OOM risk |

### 2.1 Brute-Force Vector Search (CRITICAL)

**File:** `store/libsql_store.py:515-578`

Default behavior loads ALL embeddings into memory and computes cosine similarity against every vector. At 500K chunks x 384 dims x 4 bytes = ~750MB per query.

**Recommendations:**
- Enforce FAISS for >10K chunks (currently `use_faiss = False` by default)
- Switch default FAISS index from IVF to HNSW (supports incremental updates, no training)
- Add auto-detection: warn if vault size >10K and FAISS disabled

### 2.2 Full In-Memory Chunk Buffering (CRITICAL)

**File:** `indexer/indexer.py:700-835`

All extracted chunks, texts, and metadata buffered in memory before embedding. For 100K files this can reach 1.3-3.2GB peak.

**Recommendations:**
- Stream from extraction to embedding to storage (process in windows of 100 files)
- Use generator pattern instead of accumulating lists
- Add FAISS vectors incrementally instead of `np.vstack()` all at once

### 2.3 FAISS Index Rebuild From Scratch

**File:** `store/faiss_index.py:64-102`, `libsql_store.py:172-204`

Every scan rebuilds the entire FAISS index. IVF requires retraining. No deletion handling (deleted files still have vectors in FAISS).

**Recommendations:**
- Track deleted chunks with a deletion table, filter during search
- Use HNSW instead of IVF (no training needed, supports incremental updates)
- Reduce save frequency from every 100 vectors to every 5000 or at batch end

### 2.4 Sequential Database Writes

**File:** `indexer/indexer.py:809-815`, `store/libsql_store.py:250-266`

Each chunk upserted individually with 3 SQL operations per chunk (INSERT chunks, DELETE fts, INSERT fts). 100K chunks = 300K SQL operations.

**Recommendations:**
- Use `executemany()` for batch inserts
- Wrap in explicit `BEGIN IMMEDIATE` transactions
- Batch FTS delete-then-insert into single IN clause operations

### 2.5 Unoptimized Deletion Cascade

**File:** `store/libsql_store.py:268-286`

Deleting a document cascades to 4+ separate DELETE queries per chunk. A 1000-chunk PDF takes 2000+ queries.

**Recommendations:**
- Use batch DELETE with `IN` clause
- Add foreign key cascades (`ON DELETE CASCADE`)
- Add missing indexes: `idx_chunks_doc`, `idx_embeddings_chunk`

### 2.6 Watch Mode Scalability

**File:** `indexer/change_detector.py:148-182`

- `rglob("*")` has no depth limit
- Per-file queueing creates massive queue depth for directory copies
- Debounce dictionary (`_last`) leaks memory (never cleaned up)

**Recommendations:**
- Add depth limit to directory scanning
- Clean old entries from debounce dictionary (>10s)
- Apply ignore patterns early before queueing

### 2.7 Hardcoded Configuration Limits

**File:** `config.py` (various lines)

| Parameter | Current | Issue |
|-----------|---------|-------|
| `embedding_batch_size` | 32 | Too small for GPU |
| `embed_batch_size` | 256 | GPU underutilized |
| `extraction_workers` | 8 | Should be `cpu_count() // 2` |
| `faiss_nlist` | 100 | Not tuned for vault size |
| `watch_batch_size` | 10 | Too small for directory copies |

**Recommendation:** Auto-detect based on vault size and hardware (GPU VRAM, CPU count).

---

## 3. Performance Optimizations

### Quick Wins (< 1 hour each)

| # | Optimization | File | Expected Impact |
|---|-------------|------|----------------|
| P1 | Batch SQLite commits with `executemany()` | `indexer.py:809-815` | 10-50x write speedup |
| P2 | Enable FAISS by default for >10K chunks | `config.py` | 100-1000x query speedup |
| P3 | Reduce FAISS save from every 100 to 5000 vectors | `libsql_store.py:361-365` | 20-50% scan speedup |
| P4 | Global embedding model cache | `sentence_transformers.py:44-80` | 100-500ms saved per query |
| P5 | Filter backlink counts by result doc_ids | `retriever.py:106-112` | 5-30% query speedup |
| P6 | FTS5 content deduplication (`content=chunks`) | `libsql_store.py:88-95` | 50% FTS5 size reduction |

### Medium-Effort Optimizations

| # | Optimization | File | Expected Impact |
|---|-------------|------|----------------|
| P7 | Implement manifest-based file skipping (TODO exists) | `indexer.py:717` | 90% incremental scan speedup |
| P8 | Lazy-load reranker model on first use | `reranker.py:43` | 500ms-2s saved at init |
| P9 | Ollama batch/concurrent embedding | `ollama.py:36-42` | 10-100x for Ollama users |
| P10 | Move path_prefix filter to SQL WHERE clause | `libsql_store.py:398-420` | 5-20% filtered query speedup |
| P11 | Store metadata at doc level, not per-chunk | `indexer.py:572-612` | 10-30% DB size reduction |
| P12 | Sort files by size before parallel extraction | `indexer.py:285-289` | Reduced memory spikes |

### Embedding Performance

Current default `embedding_batch_size=32` uses ~2% of a 24GB GPU. Auto-detection could yield 30-40x improvement:

```
GPU VRAM >= 24GB: batch_size = 4096
GPU VRAM >= 12GB: batch_size = 2048
GPU VRAM >=  6GB: batch_size = 512
CPU only:         batch_size = 256
```

---

## 4. Quality and Reliability

### 4.1 Error Handling Gaps

| Issue | File | Severity | Fix |
|-------|------|----------|-----|
| Bare `except Exception` hides real errors | `paths.py:12` | HIGH | Catch only `ValueError` |
| Image extraction returns empty string on failure | `image.py:93-98` | MEDIUM | Return Result type |
| Unvalidated Ollama API responses | `ollama.py:25-34` | MEDIUM | Validate embedding dimensions |
| JSON parsing without try/except | `libsql_store.py:400` | MEDIUM | Wrap with fallback to `{}` |
| CLI `open` command no chunk validation | `cli.py:242-250` | LOW | Check chunk exists |

**Positive:** The resilience layer (`resilience.py`, 350 LOC) is excellent - error classification, circuit breaker, exponential backoff with Retry-After headers.

### 4.2 Test Coverage Gaps

Test ratio is 1.16:1 (excellent), but critical scenarios are untested:

| Gap | Impact | Priority |
|-----|--------|----------|
| Corrupted/malformed files (empty PDF, truncated PPTX) | Scan crashes | HIGH |
| SQLite concurrent access (parallel scan + watch) | Data corruption | HIGH |
| Scan interruption recovery | Inconsistent index | HIGH |
| Multi-vault isolation (cross-vault data leakage) | Incorrect results | HIGH |
| Embedding dimension mismatch (model switch) | Silent corruption | MEDIUM |
| Symlinks (circular, external) | Infinite loops | MEDIUM |
| Large files (>500MB PDFs) | OOM crash | MEDIUM |
| Non-UTF8 encoded files | Data corruption | MEDIUM |

### 4.3 Logging Issues

| Issue | Fix |
|-------|-----|
| 14+ `print()` calls in `libsql_store.py` instead of `logger` | Replace with `logger.info()` |
| No log rotation (FileHandler, not RotatingFileHandler) | Use `RotatingFileHandler(maxBytes=10MB)` |
| `--verbose` enables ALL library debug logs | Selectively enable `mneme` loggers only |
| Skipped files logged silently (`continue` without log) | Add `logger.debug()` with reason |

### 4.4 Data Integrity Issues

| Issue | File | Impact |
|-------|------|--------|
| FAISS out of sync with SQLite (separate operations) | `libsql_store.py:358` | Degraded search |
| Image analysis failures not tracked for retry | `indexer.py:329-333` | Missing content |
| Links extracted but never stored (TODO comment) | `indexer.py:450-454` | Backlinks broken |
| No idempotency for partial embeddings | `indexer.py:320-323` | Wasted compute |

### 4.5 Edge Cases

| Edge Case | Current Handling | Risk |
|-----------|-----------------|------|
| Large files (500MB+) | Loaded entirely into memory | OOM crash |
| Non-UTF8 encoding | `errors="replace"` corrupts text | Data loss |
| Circular symlinks | `rglob()` follows them | Infinite loop |
| Binary files as .txt | Passed to markdown extractor | Crash |
| Windows file locking | No retry | Silent skip |
| Network paths (SMB/NFS) | No timeout on `stat()` | Hangs |
| File modified during scan | No locking | Duplicate chunks |

### 4.6 Dependency Management

| Issue | Risk |
|-------|------|
| Lower bounds only, no upper bounds (e.g., `numpy>=1.26`) | Breaking changes from numpy 2.0 |
| No lock file (`requirements-lock.txt`) | Non-reproducible builds |
| FAISS not listed in optional dependencies | Users can't discover it |
| `google-genai>=1.0.0` is unstable new library | API breakage likely |
| No security scanning (bandit, pip-audit) | Undetected vulnerabilities |

---

## 5. Architecture and Extensibility

### 5.1 Configuration Duplication (Rating: 5/10)

**File:** `config.py` (~750 lines)

`VaultConfig` and `MultiVaultConfig` share ~95% identical fields and ~300 lines of duplicated validation logic.

**Fix:** Extract `SharedVaultSettings` class used by both. Single source of truth for validation.

### 5.2 Module Coupling (Rating: 7/10)

**Files:** `retriever.py:9-14`, `indexer.py:18-28`

Retriever and Indexer import all concrete implementations directly. Cannot swap embedder or store without code changes.

**Fix:** Factory Pattern (`EmbedderFactory`, `StoreFactory`) to isolate instantiation.

### 5.3 No Dependency Injection (Rating: 4/10)

**File:** `retriever.py:21-73`

50+ line `__post_init__` that instantiates all components. Cannot inject mocks for testing. Duplicated in `MultiVaultRetriever`.

**Fix:** `RetrieverDependencies` container + `create_retriever_dependencies()` factory. Allow constructor injection: `Retriever(cfg=cfg, deps=mock_deps)`.

### 5.4 No Schema Migration System (Rating: 3/10)

**File:** `store/libsql_store.py:41-117`

Static schema with no version tracking. Config has `extractor_version`/`chunker_version` fields but they're unused. Schema changes break existing indices with no upgrade path.

**Fix:** Implement `SchemaManager` with version tracking table, sequential migration functions, and automatic upgrade on startup.

### 5.5 Limited Plugin System (Rating: 7/10)

**File:** `extractors/base.py:18-27`, `indexer/indexer.py:144-171`

`ExtractorRegistry` pattern is excellent, but registration is hardcoded in Indexer. No way to add custom extractors without code changes.

**Fix:** Support `custom_extractor_modules` in config. Load via `importlib` and call `register_extractors()` function.

### 5.6 MCP Tool Registration (Rating: 7/10)

**File:** `mcp/server.py:10-16, 59-151`

Tools defined in TOOL_MAP and again in tools/list response (duplication). No input validation against schema.

**Fix:** Self-describing `MCPToolRegistry` where each tool bundles its handler, schema, and metadata. Auto-generates tools/list response.

### 5.7 Observability (Rating: 5/10)

**Files:** `retriever.py`, `indexer/parallel_types.py`

Minimal metrics. ScanStats has basic counts but no timing breakdown. No search performance visibility.

**Fix:** Add `SearchMetrics` class with per-stage timing (embedding, lexical, vector, fusion, boosts, reranking). Expose via `--metrics` CLI flag.

---

## 6. Implementation Roadmap

### Phase 1: Safety (Week 1-2) - Prevents Data Loss

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| SQLite connection pooling | 4h | Prevents concurrent write corruption | `libsql_store.py` |
| FAISS thread locking | 2h | Prevents vector index corruption | `libsql_store.py` |
| Transaction boundaries in scan | 3h | Prevents interrupted scan data loss | `indexer.py` |
| Atomic document + manifest upsert | 2h | Prevents stale manifest state | `libsql_store.py`, `indexer.py` |

**Result:** Safe concurrent operations, crash recovery

### Phase 2: Performance (Week 3-4) - 10-100x Speedup

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Batch SQLite writes (`executemany`) | 3h | 10-50x write speedup | `libsql_store.py` |
| Enable FAISS by default for >10K | 1h | 100-1000x query speedup | `config.py` |
| Reduce FAISS save frequency | 30m | 20-50% scan speedup | `libsql_store.py` |
| Global model cache | 1h | 100-500ms per query saved | `sentence_transformers.py` |
| Manifest-based incremental skip | 4h | 90% incremental scan speedup | `indexer.py` |
| Batch deletion with IN clause | 2h | 10x delete speedup | `libsql_store.py` |

**Result:** Scales from 50K to 250K chunks

### Phase 3: Architecture (Week 5-6) - Maintainability

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Extract SharedVaultSettings | 5h | Eliminate 300 lines duplication | `config.py` |
| Factory Pattern for components | 3h | Reduce coupling, improve testability | New factory files |
| Schema migration system | 4h | Safe upgrades between versions | New `schema_manager.py` |
| Dependency injection container | 4h | Testable components | New `dependencies.py` |

**Result:** Cleaner, more maintainable, testable codebase

### Phase 4: Quality (Week 7-8) - Reliability

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Large file handling (size limits) | 2h | Prevents OOM on large PDFs | `extractors/` |
| Symlink detection and skip | 1h | Prevents infinite loops | `reconciler.py` |
| Replace print() with logger | 1h | Proper log routing | `libsql_store.py` |
| Add log rotation | 30m | Prevents disk fill | `cli.py` |
| Pin dependency upper bounds | 30m | Prevents breaking changes | `pyproject.toml` |
| Encoding detection (chardet) | 1h | Prevents non-UTF8 corruption | `utils.py` |

**Result:** Production-hardened for diverse real-world vaults

### Phase 5: Extensibility (Week 9-10) - Growth

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Plugin system for extractors | 3h | Custom file types without code changes | `config.py`, `indexer.py` |
| MCP tool registry | 3h | Self-documenting, validated MCP tools | `mcp/` |
| Search metrics and observability | 4h | Performance diagnosis capability | New `metrics.py` |
| HNSW index type | 3h | Incremental updates, no training | `faiss_index.py` |
| Streaming indexing pipeline | 6h | 80% memory reduction for large vaults | `indexer.py` |

**Result:** Scales to 1M chunks, extensible for new file types

---

## Scaling Tiers Summary

| Tier | Chunk Range | Requires | Effort |
|------|------------|----------|--------|
| **Current** | 0-50K | Nothing | Done |
| **Tier 2** | 50K-250K | Phase 1+2 (Safety + Performance) | 2-4 weeks |
| **Tier 3** | 250K-1M | + Phase 5 (HNSW, streaming) | 4-6 weeks |
| **Enterprise** | 1M+ | Sharded indexes, read replicas, tiered storage | 8-12 weeks |

---

## Assessment Verdict

**Strengths:**
- Clean Protocol-based design (Extractor, Chunker, Embedder, Store)
- Comprehensive resilience layer (circuit breaker, retry, error classification)
- Good test coverage (1.16:1 ratio, 29 test files)
- Minimal code duplication (<5% excluding config)
- Well-designed multi-vault support
- Frozen dataclasses prevent accidental mutation

**Critical Weaknesses:**
- Thread safety (SQLite, FAISS)
- No transaction boundaries (interrupted scan = corrupted index)
- In-memory buffering limits scale
- Config duplication (300+ lines)
- No schema migration strategy

**Bottom Line:** Mneme is solid for personal use. The fixes in Phases 1-2 would make it reliable for professional use, and Phases 3-5 would make it enterprise-capable.
