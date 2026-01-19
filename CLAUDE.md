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
ragtriever watch              # watch mode for continuous indexing
ragtriever mcp                # MCP server over stdio
```

## Architecture

### Data Flow
```
Vault (filesystem) → Change Detection → Ingestion Pipeline → Stores → Retrieval Service → Python API / MCP
```

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
`[[wikilinks]]` and `![[embeds]]` are extracted during indexing and stored in a `links` table for graph traversal (`vault.neighbors` MCP tool).

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
- `src/ragtriever/mcp/tools.py`: MCP tool implementations (`vault.search`, `vault.open`, `vault.neighbors`, `vault.status`)
- `src/ragtriever/store/libsql_store.py`: SQLite-based storage with FTS5 + vector BLOBs (`vaultrag.sqlite`)

### Hybrid Retrieval Strategy
1. Lexical candidates via SQLite FTS5
2. Semantic candidates via vector search
3. Merge + dedupe with configurable weights
4. Optional graph boost (backlinks) and rerank

### Query Handling
Search queries are automatically escaped for FTS5 to handle special characters (hyphens, slashes, etc.) in technical and medical terms. Queries are treated as phrase searches wrapped in double quotes, ensuring terms like "T-DXd", "CDK4/6i", or "HR+/HER2-low" work correctly without FTS5 syntax errors.

## Configuration

TOML-based config (see `examples/config.toml.example`):
- `[vault]`: root path, ignore patterns
- `[index]`: index directory, extractor/chunker versions
- `[embeddings]`: provider (sentence_transformers/ollama), model, device (cpu/cuda/mps), batch_size, offline_mode (default: true)
- `[image_analysis]`: provider (tesseract/gemini/vertex_ai/off), gemini_model for Gemini API
- `[vertex_ai]`: project_id, location, credentials_file, model (for Vertex AI with service account auth)
- `[retrieval]`: k_vec, k_lex, top_k, use_rerank
- `[mcp]`: transport (stdio)

### Offline Mode
Set `offline_mode = true` in `[embeddings]` to use cached models only (no HuggingFace downloads). This is useful in corporate environments with restricted internet access. Can be overridden with the `HF_OFFLINE_MODE` environment variable.

### Image Analysis Options
- **tesseract**: Local OCR using pytesseract (requires tesseract-ocr installed)
- **gemini**: Google Gemini API with API key authentication (set GEMINI_API_KEY)
- **vertex_ai**: Google Vertex AI with service account JSON credentials (requires google-cloud-aiplatform)
- **off**: Disable image analysis

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
