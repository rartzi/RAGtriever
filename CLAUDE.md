# CLAUDE.md

Mneme: Claude Code skill for local-first vault indexing with hybrid retrieval (semantic + lexical + link-graph).

## Project Structure

This repo contains the **Mneme skill** - a self-contained, portable skill for Claude Code.

```
RAGtriever/
├── skills/Mneme/              # The skill (everything is here)
│   ├── SKILL.md               # Routing + quick reference
│   ├── DEPLOYMENT.md          # Installation guide
│   ├── Tools/                 # CLI wrappers
│   │   ├── mneme-wrapper.sh   # Auto-installing CLI
│   │   └── manage-watcher.sh  # Watcher management
│   ├── Workflows/             # Execution procedures
│   ├── source/                # Bundled source code
│   │   ├── pyproject.toml
│   │   ├── src/mneme/         # The mneme package
│   │   └── tests/             # Test suite
│   ├── docs/                  # Documentation
│   └── examples/              # Example configs
├── README.md
├── CHANGELOG.md
└── LICENSE
```

## Development Commands

```bash
# Navigate to source
cd skills/Mneme/source

# Setup (requires Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Linting and type checking
ruff check src/ tests/
ruff check --fix src/ tests/
ruff format src/ tests/
mypy src/

# Testing
pytest
pytest tests/test_markdown_parsing.py
pytest tests/test_markdown_parsing.py::test_parse_wikilinks

# CLI (from repo root with config.toml)
cd ../../..
./skills/Mneme/Tools/mneme-wrapper.sh scan --config config.toml --full
./skills/Mneme/Tools/mneme-wrapper.sh query "search term" --k 10
./skills/Mneme/Tools/mneme-wrapper.sh list-docs --config config.toml
./skills/Mneme/Tools/mneme-wrapper.sh text-search "exact phrase" --config config.toml
./skills/Mneme/Tools/mneme-wrapper.sh backlinks --config config.toml --limit 10
./skills/Mneme/Tools/manage-watcher.sh start
```

## Architecture

**Flow:** Vault → Change Detection → Ingestion → Stores → Retrieval → API/MCP

### Core Components (in `source/src/mneme/`)

| Module | Purpose |
|--------|---------|
| `cli.py` | CLI commands (scan, query, list-docs, text-search, backlinks, watch, mcp) |
| `config.py` | Configuration management |
| `query_server.py` | Unix socket query server (runs inside watcher) |
| `extractors/` | File extractors (markdown, pdf, pptx, xlsx, images) |
| `chunking/` | Text chunking with overlap |
| `embeddings/` | Embedding providers (sentence_transformers, ollama) |
| `retrieval/` | Hybrid search, reranking, boosts |
| `store/` | SQLite/LibSQL storage |
| `indexer/` | Scan and watch orchestration |
| `mcp/` | MCP server for Claude Desktop |

### Retrieval Pipeline

**Stages:** Lexical (FTS5) + Semantic (vector) → Fusion (RRF, k=60) → Boosts → Optional Reranking

**Boosts (enabled by default):**
| Signal | Effect |
|--------|--------|
| Backlinks | +10%/link (max 2x) |
| Recency | +10% <14d, +5% <60d, -2% >180d |

**Diversity:** Max 2 chunks/doc. **Reranking:** `use_rerank = true` for 20-30% quality gain.

### Image Analysis Providers

| Provider | Auth | Use Case |
|----------|------|----------|
| `tesseract` | None | Local OCR |
| `gemini` | API key | Personal |
| `gemini-service-account` | Service acct | Enterprise |
| `off` | N/A | Text-only vaults |

## Configuration

See `skills/Mneme/examples/config.toml.example` for full options.

Key sections: `[vault]`, `[index]`, `[embeddings]`, `[image_analysis]`, `[retrieval]`, `[logging]`

**Key options:**
- `offline_mode = true` - No HuggingFace downloads
- `use_query_prefix = true` - Asymmetric BGE retrieval
- `use_faiss = true` - For >10K chunks
- `use_rerank = true` - Cross-encoder reranking

## Skill Deployment

The skill is self-contained with bundled source (~776KB). Deploy to `~/.claude/skills/Mneme/`:

```bash
# Symlink (development)
ln -s $(pwd)/skills/Mneme ~/.claude/skills/Mneme

# Or copy (distribution)
cp -r skills/Mneme ~/.claude/skills/Mneme
```

First use auto-installs mneme from bundled source (no git required).

See `skills/Mneme/DEPLOYMENT.md` for full deployment guide.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Office temp files** (`~$*.pptx`) | Add ignore pattern `**/~$*` |
| **Offline mode error** | Set `offline_mode=false` to download model first |
| **mneme not found** | Run `./skills/Mneme/Tools/mneme-wrapper.sh --install` |
