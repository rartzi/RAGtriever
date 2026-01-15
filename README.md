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
  <a href="#usage">Usage</a> •
  <a href="#mcp-integration">MCP Integration</a> •
  <a href="#documentation">Docs</a>
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
- **AI-Powered Image Analysis** - Extract text and metadata using Tesseract OCR or Gemini Vision
- **Watch Mode** - Continuously index changes as you edit your vault
- **MCP Server** - Expose your vault to AI agents via the Model Context Protocol
- **100% Local** - Your data never leaves your machine

---

## Installation

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/vaultrag.git
cd vaultrag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# For development (includes pytest, ruff, mypy)
pip install -e ".[dev]"
```

### 2. Install Optional Dependencies

```bash
# For Tesseract OCR (local image text extraction)
brew install tesseract          # macOS
sudo apt-get install tesseract-ocr  # Ubuntu/Debian

# For Gemini Vision (AI-powered image analysis)
export GEMINI_API_KEY="your-api-key"
```

### 3. Create Configuration

```bash
# Copy the example config
cp examples/config.toml.example config.toml
```

Edit `config.toml` with your paths:

```toml
[vault]
root = "/path/to/your/obsidian/vault"    # Your vault location
ignore = [".git/**", ".obsidian/cache/**", "**/.DS_Store"]

[index]
dir = "~/.vaultrag/indexes/myvault"       # Where to store the index

[embeddings]
provider = "sentence_transformers"
model = "BAAI/bge-small-en-v1.5"
device = "cpu"  # cpu | cuda | mps (Apple Silicon)

[image_analysis]
provider = "gemini"  # tesseract | gemini | off
gemini_model = "gemini-2.0-flash"

[retrieval]
top_k = 10
```

---

## Quick Start

```bash
# Index your vault
vaultrag scan --full

# Search your knowledge
vaultrag query "machine learning concepts"

# Watch for changes (continuous indexing)
vaultrag watch
```

---

## Usage

VaultRAG can be used in three ways: **CLI**, **Python API**, or **MCP Server**.

### CLI Usage

```bash
# Initialize a new config file
vaultrag init --vault "/path/to/vault" --index "~/.vaultrag/indexes/myvault"

# Full index scan
vaultrag scan --full

# Incremental scan (only changed files)
vaultrag scan

# Search your vault
vaultrag query "project planning" --k 10

# Search with path filter
vaultrag query "meeting notes" --path "Work/Meetings/"

# Watch mode - continuously index changes
vaultrag watch

# Open a specific chunk by ID
vaultrag open <chunk_id>

# Start MCP server
vaultrag mcp
```

### Python API Usage

```python
from vaultrag.config import VaultConfig
from vaultrag.indexer.indexer import Indexer
from vaultrag.retrieval.retriever import Retriever

# Load configuration
cfg = VaultConfig.from_toml("config.toml")

# Index your vault
indexer = Indexer(cfg)
indexer.scan(full=True)

# Search your vault
retriever = Retriever(cfg)
results = retriever.search("machine learning", k=10)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"File: {result.source_ref.rel_path}")
    print(f"Snippet: {result.snippet[:200]}...")
    print("---")

# Open full content of a result
content = retriever.open(results[0].source_ref)
print(content.content)

# Get linked notes (outlinks and backlinks)
neighbors = retriever.neighbors("Projects/MyProject.md")
print(f"Outlinks: {neighbors['outlinks']}")
print(f"Backlinks: {neighbors['backlinks']}")
```

#### Advanced: Custom Filters

```python
# Search with metadata filters
results = retriever.search(
    query="architecture decisions",
    k=10,
    filters={
        "path_prefix": "Projects/",  # Only search in Projects folder
        "file_type": "markdown",     # Only markdown files
    }
)
```

#### Advanced: Direct Store Access

```python
from vaultrag.store.libsql_store import LibSqlStore

# Direct access to the store
store = LibSqlStore(cfg.index_dir / "vaultrag.sqlite")

# Get index status
status = store.status(vault_id="your_vault_id")
print(f"Indexed files: {status['indexed_files']}")
print(f"Indexed chunks: {status['indexed_chunks']}")

# Lexical search only
lexical_results = store.lexical_search("python async", k=20, filters={})

# Vector search only
import numpy as np
query_vector = embedder.embed_texts(["python async"])[0]
vector_results = store.vector_search(query_vector, k=20, filters={})
```

---

## MCP Integration

VaultRAG exposes your vault to AI agents via the [Model Context Protocol](https://modelcontextprotocol.io/).

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `vault.search` | Hybrid search across your vault |
| `vault.open` | Retrieve full content of a search result |
| `vault.neighbors` | Get linked notes (outlinks/backlinks) |
| `vault.status` | Index statistics |

### Running the MCP Server

```bash
# Start the MCP server (stdio transport)
vaultrag mcp
```

### Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vaultrag": {
      "command": "/absolute/path/to/vaultrag/.venv/bin/vaultrag",
      "args": ["mcp"],
      "cwd": "/absolute/path/to/vaultrag"
    }
  }
}
```

Then restart Claude Desktop. You can now ask Claude to search your vault:

> "Search my vault for notes about machine learning"
> "Find all meeting notes from last week"
> "What are the backlinks to my Project X note?"

### MCP Tool Examples

The MCP server accepts JSON-RPC requests. Here are example tool calls:

#### vault.search

```json
{
  "tool": "vault.search",
  "params": {
    "query": "kubernetes deployment",
    "k": 5,
    "filters": {
      "path_prefix": "DevOps/"
    }
  }
}
```

#### vault.open

```json
{
  "tool": "vault.open",
  "params": {
    "source_ref": {
      "rel_path": "Projects/K8s.md",
      "file_type": "markdown",
      "anchor_type": "md_heading",
      "anchor_ref": "Deployment Strategy"
    }
  }
}
```

#### vault.neighbors

```json
{
  "tool": "vault.neighbors",
  "params": {
    "path_or_doc_id": "Projects/MyProject.md",
    "depth": 1
  }
}
```

#### vault.status

```json
{
  "tool": "vault.status",
  "params": {}
}
```

---

## Architecture

```
Vault (filesystem)
    │
    ├─► Change Detection (watch/reconcile)
    │
    ├─► Ingestion Pipeline
    │     ├─ Extract (MD, PDF, PPTX, XLSX, images)
    │     ├─ Chunk (heading-aware for MD, boundary markers for docs)
    │     └─ Embed (SentenceTransformers / Ollama)
    │
    ├─► Storage Layer (SQLite)
    │     ├─ Documents & Chunks tables
    │     ├─ FTS5 virtual table (lexical search)
    │     ├─ Embeddings table (vector search)
    │     └─ Links table (graph)
    │
    └─► Retrieval Service
          ├─ Hybrid ranking (vector + lexical + graph boost)
          ├─ Python API (Retriever class)
          └─ MCP Server (stdio JSON-RPC)
```

---

## Configuration Reference

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| `[vault]` | `root` | Path to your vault | Required |
| `[vault]` | `ignore` | Glob patterns to ignore | `[".git/**"]` |
| `[index]` | `dir` | Path to store the index | Required |
| `[embeddings]` | `provider` | `sentence_transformers` or `ollama` | `sentence_transformers` |
| `[embeddings]` | `model` | Embedding model name | `BAAI/bge-small-en-v1.5` |
| `[embeddings]` | `device` | `cpu`, `cuda`, or `mps` | `cpu` |
| `[image_analysis]` | `provider` | `tesseract`, `gemini`, or `off` | `tesseract` |
| `[image_analysis]` | `gemini_model` | Gemini model for vision | `gemini-2.0-flash` |
| `[retrieval]` | `top_k` | Default number of results | `10` |
| `[retrieval]` | `k_vec` | Vector search candidates | `40` |
| `[retrieval]` | `k_lex` | Lexical search candidates | `40` |

---

## Development

```bash
# Run tests
pytest

# Run single test
pytest tests/test_markdown_parsing.py::test_parse_wikilinks

# Linting
ruff check src/ tests/
ruff check --fix src/ tests/  # Auto-fix

# Type checking
mypy src/
```

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and data flow
- [PRD](docs/PRD.md) - Product requirements
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Development roadmap
- [MCP Tool Spec](docs/MCP_TOOL_SPEC.json) - MCP protocol specification
- [Planned Tasks](docs/PLANNED_TASKS.md) - TODO items

---

## License

[MIT License](LICENSE) - feel free to use this in your own projects.

---

<p align="center">
  Built for knowledge workers who value privacy and local-first software.
</p>
