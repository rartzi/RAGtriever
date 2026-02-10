# Changelog

All notable changes to Mneme (formerly RAGtriever) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2026-02-09

### Added

- **Agentic Search Tools**: 3 new MCP tools (`vault_list_docs`, `vault_text_search`, `vault_backlinks`) and 3 new CLI commands (`list-docs`, `text-search`, `backlinks`) enabling iterative, multi-step search for complex questions
- **AgenticSearch workflow**: New 6-step iterative search pattern (Orient → Search → Refine → Connect → Read → Synthesize) documented in `Workflows/AgenticSearch.md`
- **Query server dispatch**: Extended unix socket query server with `list_docs`, `text_search`, and `backlinks` actions alongside existing `query` and `ping`
- **Generic socket client**: New `request_via_socket()` function for arbitrary query server requests; `query_via_socket()` preserved as backward-compatible wrapper
- **Multi-vault support**: All new CLI commands support multi-vault configurations with `--vaults` filter

## [3.3.0] - 2026-02-07

### Improved

- **Backlink boost filtering**: Backlink count queries now filter to result doc_ids in SQL instead of scanning all links (5-30% query speedup on large vaults)
- **Path prefix SQL push-down**: `path_prefix` filter applied in FTS5 SQL WHERE clause instead of post-filtering in Python (5-20% filtered query speedup)
- **Lazy reranker loading**: CrossEncoder model loaded on first `rerank()` call, not at Retriever init (saves 500ms-2s startup when reranking enabled but not every query uses it)
- **Chunk deduplication before embedding**: Duplicate `chunk_id`s deduplicated before embedding computation, avoiding wasted GPU/CPU work and ensuring scan output matches actual stored count

### Fixed

- **Duplicate watcher log lines**: Fixed `manage-watcher.sh` redirecting stdout to the same log file that `--log-file` writes to, causing every line to appear 2x
- **Logging handler accumulation**: `_setup_logging()` now clears existing handlers before adding new ones, preventing duplicate log output on repeated calls

### Added

- **FAISS optional dependency**: `pip install mneme[faiss]` for discoverable FAISS installation

## [3.2.0] - 2026-02-07

### Performance

- **Batch SQLite writes**: `upsert_chunks()` and `upsert_embeddings()` now use `executemany()` for batch inserts, replacing per-row execute loops (10-50x write speedup for large scans)
- **Batch deletion**: `delete_document()` uses `DELETE ... WHERE chunk_id IN (...)` instead of per-chunk loops (10x delete speedup)
- **Manifest-based incremental skip**: Scan checks file mtime+size against manifest before extraction, skipping unchanged files entirely (incremental scan completes in <1s for unchanged vaults)
- **FAISS save frequency**: Reduced FAISS index checkpoint frequency from every 1,000 to every 5,000 vectors, with explicit `save_faiss_index()` call at end of scan
- **FAISS auto-warning**: Logs a warning when brute-force vector search is used with >10K chunks and FAISS is disabled

### Fixed

- **Watcher manifest interop**: `_index_one()` (used by watcher) now writes manifest entries, ensuring watcher-indexed files are correctly skipped by subsequent incremental scans
- **Logging consistency**: Replaced all `print()` calls in indexer and retriever with proper `logging.getLogger(__name__)` calls for consistent log routing

## [3.1.1] - 2026-02-06

### Fixed

- **Resilient frontmatter parsing**: Markdown extractor no longer crashes on files with invalid YAML frontmatter (e.g., Obsidian templates using `{{date}}` syntax). Files with unparseable frontmatter are now indexed with their content intact and empty metadata, logged as a warning instead of failing the scan.

## [3.1.0] - 2026-01-26

### Added

- **Watcher catch-up on startup**: The watcher now detects and reindexes files modified while it was stopped. On startup, it compares filesystem mtimes against manifest timestamps and queues stale files for reprocessing. This ensures no changes are missed when restarting the watcher.
  - New `queue_stale_files()` method in `ChangeDetector`
  - New `get_manifest_mtimes()` method in `Store` protocol
  - Logs show catch-up progress: `[watch] Queued N stale files for reindex (X new, Y modified)`

## [3.0.0] - 2026-01-25

### BREAKING CHANGES

**Project Renamed: RAGtriever → Mneme**

The project has been renamed from RAGtriever to Mneme (pronounced NEE-mee), after the Greek Muse of Memory. This name better reflects the project's purpose as a memory layer for your Second Brain.

**What changed:**
- Package name: `ragtriever` → `mneme`
- CLI command: `ragtriever` → `mneme`
- Import paths: `from ragtriever` → `from mneme`
- Default index directory: `~/.ragtriever/` → `~/.mneme/`
- MCP server name: `ragtriever` → `mneme`
- Skill name: `RAGtrieval` → `Mneme`

**Migration Guide:**

1. **Update package:**
   ```bash
   pip uninstall ragtriever
   pip install mneme
   ```

2. **Update CLI commands:**
   ```bash
   # Old
   ragtriever scan --full
   ragtriever query "search term"

   # New
   mneme scan --full
   mneme query "search term"
   ```

3. **Update imports in your code:**
   ```python
   # Old
   from ragtriever.config import VaultConfig
   from ragtriever.retrieval import Retriever

   # New
   from mneme.config import VaultConfig
   from mneme.retrieval import Retriever
   ```

4. **Update config paths (optional):**
   ```toml
   # Old
   [index]
   dir = "~/.ragtriever/indexes/myvault"

   # New
   [index]
   dir = "~/.mneme/indexes/myvault"
   ```
   Note: Existing indexes at `~/.ragtriever/` will continue to work.

5. **Update MCP configuration:**
   ```json
   {
     "mcpServers": {
       "mneme": {
         "command": "mneme",
         "args": ["mcp", "--config", "/path/to/config.toml"]
       }
     }
   }
   ```

6. **Update skill symlink (if using Claude Code):**
   ```bash
   rm ~/.claude/skills/RAGtrieval
   ln -s /path/to/mneme/skills/Mneme ~/.claude/skills/Mneme
   ```

**Why this change?**

- "Mneme" (Greek Muse of Memory) better represents the project's purpose
- Emphasizes the "memory" aspect of a Second Brain system
- Shorter, more memorable name
- Cleaner branding aligned with knowledge management philosophy

## [2.0.0] - 2026-01-25

### BREAKING CHANGES

**Image Analysis Provider Renamed: `vertex_ai` → `gemini-service-account`**

The `vertex_ai` image analysis provider has been renamed to `gemini-service-account` to better reflect that it uses Gemini models via GCP service account authentication, not Vertex AI-specific APIs.

**What changed:**
- Provider string: `"vertex_ai"` → `"gemini-service-account"`
- Config section: `[vertex_ai]` → `[gemini_service_account]`
- Config fields: `vertex_ai_*` → `gemini_sa_*`
  - `vertex_ai_project_id` → `gemini_sa_project_id`
  - `vertex_ai_location` → `gemini_sa_location`
  - `vertex_ai_credentials_file` → `gemini_sa_credentials_file`
  - `vertex_ai_model` → `gemini_sa_model`
  - `vertex_ai_timeout` → `gemini_sa_timeout`
- Class name: `VertexAIImageExtractor` → `GeminiServiceAccountImageExtractor`
- Documentation: `docs/vertex_ai_setup.md` → `docs/gemini_service_account_setup.md`

**Migration Guide:**

Update your `config.toml`:

```toml
# OLD (v0.1.0)
[image_analysis]
provider = "vertex_ai"

[vertex_ai]
project_id = "your-project"
location = "global"
credentials_file = "/path/to/creds.json"
model = "gemini-2.0-flash-exp"

# NEW (v2.0.0)
[image_analysis]
provider = "gemini-service-account"

[gemini_service_account]
project_id = "your-project"
location = "global"
credentials_file = "/path/to/creds.json"
model = "gemini-2.0-flash-exp"
```

**Why this change?**

All three Gemini-based providers (`gemini`, `gemini-service-account`, `aigateway`) use the same Gemini models. The difference is authentication method:
- `gemini`: API key authentication
- `gemini-service-account`: GCP service account authentication
- `aigateway`: Microsoft AI Gateway proxy

The old name `vertex_ai` suggested using Vertex AI APIs, but the provider actually uses Gemini models accessed via Vertex AI with service account credentials.

## [0.1.0] - 2026-01-24

Initial release.

### Added
- Local-first vault indexing with hybrid search (semantic + lexical)
- Support for markdown, PDF, PowerPoint, Excel, and image files
- Multiple embedding providers (SentenceTransformers, Ollama)
- Multiple image analysis providers (Tesseract, Gemini, Vertex AI, AI Gateway)
- MCP (Model Context Protocol) server interface
- Watch mode for continuous indexing
- Parallel scanning with configurable workers
- Cross-encoder reranking support
- FAISS approximate nearest neighbor search
- Result diversity (MMR) and boost algorithms
- Resilient image analysis with circuit breaker pattern
