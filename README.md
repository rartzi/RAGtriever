<p align="center">
  <img src="assets/hero.jpg" alt="VaultRAG - Local Knowledge Retrieval" width="800"/>
</p>

<h1 align="center">VaultRAG</h1>

<p align="center">
  <strong>Local-first hybrid retrieval system for your second brain</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#mcp-integration">MCP Integration</a> •
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platform"/>
</p>

---

VaultRAG indexes your Obsidian-compatible vault into a powerful hybrid search system combining **semantic search**, **lexical search (FTS5)**, and **link-graph awareness**. All data stays local on your machine.

## Features

- **Hybrid Retrieval** - Combines vector embeddings with full-text search for superior results
- **Obsidian-Aware** - Understands YAML frontmatter, `[[wikilinks]]`, `![[embeds]]`, and `#tags`
- **Multi-Format Support** - Index Markdown, PDF, PPTX, XLSX, and images
- **AI-Powered Image Analysis** - Extract text and metadata from images using Tesseract OCR or Gemini Vision
- **Watch Mode** - Continuously index changes as you edit your vault
- **MCP Server** - Expose your vault to AI agents via the Model Context Protocol
- **100% Local** - Your data never leaves your machine

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vaultrag.git
cd vaultrag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For Tesseract OCR (image text extraction)
brew install tesseract  # macOS
apt-get install tesseract-ocr  # Ubuntu/Debian

# For Gemini Vision (AI-powered image analysis)
# Set GEMINI_API_KEY environment variable
```

## Quick Start

### 1. Create Configuration

```bash
# Copy the example config
cp examples/config.toml.example config.toml

# Edit config.toml with your vault path
```

### 2. Index Your Vault

```bash
# Full scan of your vault
vaultrag scan --full

# Watch for changes (continuous indexing)
vaultrag watch
```

### 3. Search Your Knowledge

```bash
# Search your vault
vaultrag query "machine learning concepts" --k 10

# Filter by path
vaultrag query "meeting notes" --path "Work/Meetings/"
```

## Configuration

Create a `config.toml` file (see `examples/config.toml.example`):

```toml
[vault]
root = "/path/to/your/vault"
ignore = [".git/**", ".obsidian/cache/**", "**/.DS_Store"]

[index]
dir = "~/.vaultrag/indexes/myvault"

[embeddings]
provider = "sentence_transformers"
model = "BAAI/bge-small-en-v1.5"
device = "cpu"  # cpu|cuda|mps

[image_analysis]
provider = "gemini"  # tesseract|gemini|off
gemini_model = "gemini-2.0-flash"

[retrieval]
top_k = 10
```

### Image Analysis Options

| Provider | Description | Requirements |
|----------|-------------|--------------|
| `tesseract` | Local OCR extraction | `tesseract-ocr` installed |
| `gemini` | AI vision analysis with rich metadata | `GEMINI_API_KEY` env var |
| `off` | Skip image indexing | None |

Gemini Vision provides:
- Detailed image descriptions
- Accurate text extraction (OCR)
- Image type classification (screenshot, diagram, photo, etc.)
- Topic and entity extraction

## MCP Integration

VaultRAG exposes your vault to AI agents via the [Model Context Protocol](https://modelcontextprotocol.io/).

### Available Tools

| Tool | Description |
|------|-------------|
| `vault.search` | Hybrid search across your vault |
| `vault.open` | Retrieve full content of a search result |
| `vault.neighbors` | Get linked notes (outlinks/backlinks) |
| `vault.status` | Index statistics |

### Running the MCP Server

```bash
vaultrag mcp
```

### Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "vaultrag": {
      "command": "/path/to/venv/bin/vaultrag",
      "args": ["mcp"],
      "cwd": "/path/to/your/config/directory"
    }
  }
}
```

## Architecture

```
Vault (filesystem)
    │
    ├─► Change Detection (watch/reconcile)
    │
    ├─► Ingestion Pipeline
    │     ├─ Extract (MD, PDF, PPTX, XLSX, images)
    │     ├─ Chunk (heading-aware, boundary markers)
    │     └─ Embed (SentenceTransformers / Ollama)
    │
    ├─► Storage Layer
    │     ├─ SQLite (documents, chunks, metadata)
    │     ├─ FTS5 (lexical search)
    │     └─ Vector store (semantic search)
    │
    └─► Retrieval Service
          ├─ Hybrid ranking (vector + lexical)
          ├─ Python API
          └─ MCP Server
```

## CLI Reference

```bash
vaultrag init     # Create starter config
vaultrag scan     # Index vault (--full for complete reindex)
vaultrag watch    # Watch for changes and index continuously
vaultrag query    # Search your vault
vaultrag open     # Open a specific chunk
vaultrag mcp      # Run MCP server
```

## Documentation

- [Product Requirements (PRD)](docs/PRD.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [MCP Tool Specification](docs/MCP_TOOL_SPEC.json)

## Development

```bash
# Run tests
pytest

# Linting
ruff check src/ tests/

# Type checking
mypy src/
```

## License

[MIT License](LICENSE) - feel free to use this in your own projects.

---

<p align="center">
  Built for knowledge workers who value privacy and local-first software.
</p>
