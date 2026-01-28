<p align="center">
  <img src="assets/hero.jpg" alt="Mneme - Memory for Your Second Brain" width="800"/>
</p>

<h1 align="center">Mneme</h1>

<p align="center">
  <strong>Memory for your Second Brain</strong><br/>
  <em>(pronounced NEE-mee, after the Greek Muse of Memory)</em>
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

Mneme indexes your Obsidian-compatible vault into a powerful hybrid retrieval system combining **semantic search**, **lexical search (FTS5)**, and **link-graph awareness** for retrieval-augmented generation. All data stays local on your machine.

> **⚠️ Breaking Change in v3.0.0**: The project has been renamed from Mneme to Mneme. Update your CLI commands from `mneme` to `mneme`, and update imports from `mneme` to `mneme`. See [CHANGELOG.md](CHANGELOG.md) for migration details.

## Features

- **Hybrid Retrieval** - Combines vector embeddings with full-text search for superior results
- **RRF Fusion** - Reciprocal Rank Fusion for robust score-agnostic result merging
- **Backlink Boost** - Hub documents with more incoming links rank higher (configurable)
- **Recency Boost** - Fresh documents get priority with tiered time-based boosting
- **Chunk Overlap (v1.0+)** - Context preservation with configurable 200-char overlap between chunks
- **Query Instruction Prefix (v1.0+)** - Asymmetric retrieval for better BGE model performance
- **Cross-Encoder Reranking (Optional)** - Refine search results with cross-encoder for 20-30% quality improvement
- **Obsidian-Aware** - Understands YAML frontmatter, `[[wikilinks]]`, `![[embeds]]`, and `#tags`
- **Multi-Format Support** - Index Markdown, PDF, PPTX, XLSX, and images
- **Parallel Scanning (3.6x Faster)** - ThreadPool-based parallel extraction and image analysis
- **Embedded Image Extraction** - Automatically extracts and analyzes images from PDFs, PowerPoints, and Markdown
- **AI-Powered Image Analysis** - Extract text and metadata using Tesseract OCR, Gemini Vision, or Vertex AI
- **Watch Mode** - Continuously index changes as you edit your vault (add, change, delete)
- **File Deletion Detection** - Scan mode detects deleted files and removes them from index
- **MCP Server** - Expose your vault to AI agents via the Model Context Protocol
- **FAISS Scaling** - Optional approximate nearest neighbor search for large vaults (10K+ chunks)
- **100% Local** - Your data never leaves your machine (when using local embeddings + Tesseract)

---

## Installation

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/mneme.git
cd mneme

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
dir = "~/.mneme/indexes/myvault"       # Where to store the index

[embeddings]
provider = "sentence_transformers"
model = "BAAI/bge-small-en-v1.5"
device = "cpu"  # cpu | cuda | mps (Apple Silicon)

[image_analysis]
provider = "gemini-service-account"  # tesseract | gemini | gemini-service-account | off

# For Gemini with service account (recommended for production)
[gemini_service_account]
project_id = "your-gcp-project"
location = "global"
credentials_file = "~/.config/gcloud/service-account-key.json"
model = "gemini-2.0-flash-exp"

[retrieval]
top_k = 10
```

---

## Quick Start

```bash
# Index your vault (uses parallel scanning by default - 3.6x faster)
mneme scan --full

# Search your knowledge
mneme query "machine learning concepts"

# Search with reranking for best quality
mneme query "machine learning concepts" --rerank

# Control parallel scanning
mneme scan --full --workers 8      # Use 8 parallel workers
mneme scan --full --no-parallel    # Disable parallelization

# Watch for changes (continuous indexing)
mneme watch
```

### Enable Reranking (Optional)

For best result quality, enable cross-encoder reranking in your config:

```toml
[retrieval]
use_rerank = true
```

This adds ~100-200ms latency but significantly improves relevance. You can also enable it per-query:

```bash
mneme query "kubernetes deployment" --rerank
```

---

## Usage

Mneme can be used in three ways: **CLI**, **Python API**, or **MCP Server**.

### CLI Usage

```bash
# Initialize a new config file
mneme init --vault "/path/to/vault" --index "~/.mneme/indexes/myvault"

# Full index scan
mneme scan --full

# Incremental scan (only changed files)
mneme scan

# Search your vault
mneme query "project planning" --k 10

# Search with reranking (improves quality by 20-30%)
mneme query "project planning" --k 10 --rerank

# Search with path filter
mneme query "meeting notes" --path "Work/Meetings/"

# Watch mode - continuously index changes
mneme watch

# Open a specific chunk by ID
mneme open <chunk_id>

# Start MCP server
mneme mcp
```

### Python API Usage

```python
from mneme.config import VaultConfig
from mneme.indexer.indexer import Indexer
from mneme.retrieval.retriever import Retriever

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
from mneme.store.libsql_store import LibSqlStore

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

Mneme exposes your vault to AI agents via the [Model Context Protocol](https://modelcontextprotocol.io/).

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `vault_search` | Hybrid search across your vault |
| `vault_open` | Retrieve full content of a search result |
| `vault_neighbors` | Get linked notes (outlinks/backlinks) |
| `vault_status` | Index statistics |
| `vault_list` | List configured vaults |

### Running the MCP Server

```bash
# Start the MCP server (stdio transport)
mneme mcp
```

### Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mneme": {
      "command": "/absolute/path/to/mneme/.venv/bin/mneme",
      "args": ["mcp"],
      "cwd": "/absolute/path/to/mneme"
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

#### vault_search

```json
{
  "method": "tools/call",
  "params": {
    "name": "vault_search",
    "arguments": {
      "query": "kubernetes deployment",
      "k": 5
    }
  }
}
```

#### vault_open

```json
{
  "method": "tools/call",
  "params": {
    "name": "vault_open",
    "arguments": {
      "rel_path": "Projects/K8s.md"
    }
  }
}
```

#### vault_neighbors

```json
{
  "method": "tools/call",
  "params": {
    "name": "vault_neighbors",
    "arguments": {
      "rel_path": "Projects/MyProject.md"
    }
  }
}
```

#### vault_status

```json
{
  "method": "tools/call",
  "params": {
    "name": "vault_status",
    "arguments": {}
  }
}
```

#### vault_list

```json
{
  "method": "tools/call",
  "params": {
    "name": "vault_list",
    "arguments": {}
  }
}
```

---

## Claude Code Skill (Optional)

For users of [Claude Code](https://claude.ai/code), Mneme includes a workflow skill to assist with setup, configuration, and troubleshooting. The skill follows the PAI (Personal AI Infrastructure) standard with workflow-based routing.

### Installation

```bash
# Option 1: Symlink from repository (recommended)
ln -s $(pwd)/skills/Mneme ~/.claude/skills/Mneme

# Option 2: Copy skill
cp -r skills/Mneme ~/.claude/skills/Mneme
```

### Skill Structure

The skill uses a modular PAI-compliant structure:

```
skills/Mneme/
├── SKILL.md                    # Main routing file with workflow table
├── SearchBestPractices.md      # Search strategy and source citation
├── Configuration.md            # Config checklist, image providers
├── Commands.md                 # CLI command reference
├── Troubleshooting.md          # Issue/solution pairs
├── WatcherManagement.md        # Watcher operations
├── DevelopmentWorkflow.md      # Dev workflow, testing
├── Architecture.md             # Data flow, execution modes
└── Workflows/                  # Execution procedures
    ├── SearchVault.md          # Search and cite sources
    ├── SetupVault.md           # Initial vault setup
    ├── ConfigureImageAnalysis.md  # Image provider config
    ├── ManageWatcher.md        # Watcher management
    ├── Scan.md                 # Scanning operations
    └── Troubleshoot.md         # Diagnostic procedures
```

### Available Workflows

| Workflow | Trigger |
|----------|---------|
| **SearchVault** | "what does the vault say about", answering content questions |
| **SetupVault** | "setup mneme", "initialize vault" |
| **ConfigureImageAnalysis** | "configure images", "setup gemini" |
| **ManageWatcher** | "start watcher", "stop watcher", "watcher status" |
| **Scan** | "run scan", "full scan", "incremental scan" |
| **Troubleshoot** | "error", "not working", troubleshooting |

Once installed, Claude Code will automatically use this skill when helping you with Mneme tasks.

**Note:** The skill is optional - Mneme works independently without it. The skill simply helps Claude Code assist you better.

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
| `[embeddings]` | `offline_mode` | Use cached models only | `true` |
| `[image_analysis]` | `provider` | `tesseract`, `gemini`, `gemini-service-account`, or `off` | `tesseract` |
| `[gemini_service_account]` | `project_id` | Google Cloud project ID | - |
| `[gemini_service_account]` | `location` | GCP region | `us-central1` |
| `[gemini_service_account]` | `credentials_file` | Service account JSON path | - |
| `[gemini_service_account]` | `model` | Gemini model name | `gemini-2.0-flash-exp` |
| `[retrieval]` | `top_k` | Default number of results | `10` |
| `[retrieval]` | `k_vec` | Vector search candidates | `40` |
| `[retrieval]` | `k_lex` | Lexical search candidates | `40` |
| `[indexing]` | `extraction_workers` | Parallel file extraction workers | `8` |
| `[indexing]` | `embed_batch_size` | Cross-file embedding batch size | `256` |
| `[indexing]` | `image_workers` | Parallel image API workers | `8` |
| `[indexing]` | `parallel_scan` | Enable parallel scanning | `true` |

**Note**: Mneme automatically extracts and analyzes images embedded in PDFs and PowerPoint presentations, as well as images referenced in Markdown files (`![](image.png)` and `![[image.png]]`). These are indexed as separate chunks linked to their parent documents.

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

- [Architecture](docs/ARCHITECTURE.md) - Complete system architecture and components
- [Testing Guide](docs/testing.md) - Comprehensive test suite documentation
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
- [Gemini Service Account Setup](docs/gemini_service_account_setup.md) - Gemini with GCP service account configuration
- [Improvements](IMPROVEMENTS.md) - Planned enhancements and roadmap
- [Mneme Skill](skills/Mneme/SKILL.md) - Claude Code workflow skill (PAI-compliant)
- [MCP Tool Spec](docs/MCP_TOOL_SPEC.json) - MCP protocol specification

---

## License

[MIT License](LICENSE) - feel free to use this in your own projects.

---

<p align="center">
  Built for knowledge workers who value privacy and local-first software.
</p>
