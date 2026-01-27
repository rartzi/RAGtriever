# Changelog

All notable changes to Mneme (formerly RAGtriever) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
