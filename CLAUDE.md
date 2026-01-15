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
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Linting and type checking
ruff check src/ tests/
ruff check --fix src/ tests/    # autofix
mypy src/

# Testing
pytest                                              # all tests
pytest tests/test_markdown_parsing.py              # single file
pytest tests/test_markdown_parsing.py::test_parse_wikilinks  # single test

# CLI (after creating config.toml)
cortex init --vault "/path/to/vault" --index "~/.ragtriever/indexes/myvault"
cortex scan --full
cortex query "search term" --k 10
cortex mcp                    # MCP server over stdio
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

### Key Entry Points
- `src/ragtriever/cli.py`: Typer CLI (`cortex` command)
- `src/ragtriever/indexer/indexer.py`: Main `Indexer` class orchestrating extract → chunk → embed → store
- `src/ragtriever/retrieval/retriever.py`: `Retriever` class for hybrid search
- `src/ragtriever/mcp/tools.py`: MCP tool implementations (`vault.search`, `vault.open`, `vault.neighbors`, `vault.status`)

### Hybrid Retrieval Strategy
1. Lexical candidates via SQLite FTS5
2. Semantic candidates via vector search
3. Merge + dedupe with configurable weights
4. Optional graph boost (backlinks) and rerank

## Configuration

TOML-based config (see `examples/config.toml`):
- `[vault]`: root path, ignore patterns
- `[index]`: index directory, extractor/chunker versions
- `[embeddings]`: provider (sentence_transformers/ollama), model, device (cpu/cuda/mps)
- `[retrieval]`: k_vec, k_lex, top_k, use_rerank
- `[mcp]`: transport (stdio)

## Implementation Priorities

See `PLANNED_TASKS.md` for current priorities and Definition of Done.
