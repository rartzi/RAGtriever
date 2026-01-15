# RAGtriever Architecture

Complete system architecture explaining all components and how they interact.

## Quick Navigation
- [System Overview](#system-overview) - High-level design
- [Core Components](#core-components) - Vault, Indexer, Store, Retriever
- [Execution Modes](#execution-modes) - CLI, Watch, MCP, Python API
- [Skills vs Non-Skills](#skills-vs-non-skills) - **Important:** RAGtriever is NOT a Claude Code skill
- [Data Flow](#data-flow) - How data moves through the system
- [Extension Points](#extension-points) - How to customize

---

## System Overview

RAGtriever is a **local-first hybrid retrieval system** for Obsidian-compatible vaults.

```
┌─────────────────────────────────────────────────────────────┐
│                        RAGtriever                           │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐   ┌─────────────┐ │
│  │   Vault      │──────▶│   Indexer    │──▶│   Store     │ │
│  │ (Filesystem) │      │  (Pipeline)  │   │  (SQLite)   │ │
│  └──────────────┘      └──────────────┘   └─────────────┘ │
│         │                      │                   │        │
│         │                      ▼                   ▼        │
│         │              ┌──────────────┐   ┌─────────────┐  │
│         │              │  Embeddings  │   │  FTS5 Index │  │
│         │              │   (Vectors)  │   │  (Lexical)  │  │
│         │              └──────────────┘   └─────────────┘  │
│         │                      │                   │        │
│         │                      └───────┬───────────┘        │
│         │                              ▼                    │
│         │                      ┌──────────────┐            │
│         │                      │  Retriever   │            │
│         │                      │  (Hybrid)    │            │
│         │                      └──────────────┘            │
│         │                              │                    │
│         │              ┌───────────────┼───────────────┐   │
│         │              ▼               ▼               ▼   │
│         │         ┌─────────┐   ┌──────────┐   ┌────────┐│
│         └────────▶│   CLI   │   │  Python  │   │  MCP   ││
│                   │         │   │   API    │   │ Server ││
│                   └─────────┘   └──────────┘   └────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Design Principles:**
1. **Filesystem is source of truth** - Index can always be rebuilt
2. **Local-only** - No data leaves machine (unless you choose Gemini/Vertex AI)
3. **Pluggable** - Swap extractors, embedders, chunkers via Protocol classes
4. **Obsidian-aware** - Parses `[[wikilinks]]`, `#tags`, YAML frontmatter

---

## Core Components

### 1. Vault (Input Layer)

**What:** Your Obsidian vault on the filesystem
**Location:** User-specified (e.g., `~/Documents/vault`)
**Format:** Markdown + attachments (PDF, PPTX, images)

```
vault/
├── notes/
│   ├── project.md           # Markdown with [[wikilinks]]
│   └── meeting-notes.md
├── attachments/
│   ├── diagram.png          # Images for vision analysis
│   └── whitepaper.pdf       # PDFs for text extraction
└── .obsidian/              # Ignored by default
```

**Key Point:** RAGtriever reads from vault but never modifies it.

### 2. Indexer (Processing Pipeline)

**What:** Converts files into searchable chunks
**Code:** `src/ragtriever/indexer/indexer.py`
**Entry Point:** `Indexer.scan()`

**Pipeline:**
```python
for file in vault:
    1. Extract  → text + metadata  (MarkdownExtractor, PDFExtractor, ImageExtractor)
    2. Chunk    → segments          (MarkdownChunker, BoundaryMarkerChunker)
    3. Embed    → vectors           (SentenceTransformers, Ollama)
    4. Store    → SQLite + FTS5     (LibSqlStore)
```

**Pluggable via Protocol classes:**
- **Extractors:** One per file type (`.md`, `.pdf`, `.pptx`, `.png`)
- **Chunkers:** Split text semantically (by heading, by page, etc.)
- **Embedders:** Generate vector representations (local or API)

### 3. Store (Persistence Layer)

**What:** SQLite database with hybrid index
**Location:** `~/.ragtriever/indexes/{vault_name}/vaultrag.sqlite`
**Code:** `src/ragtriever/store/libsql_store.py`

**Tables:**
```sql
-- File metadata
documents(doc_id, vault_id, rel_path, file_type, mtime, content_hash, ...)

-- Semantic segments
chunks(chunk_id, doc_id, anchor_type, anchor_ref, text, ...)

-- Full-text search index
fts_chunks USING fts5(chunk_id, vault_id, rel_path, text)

-- Vector embeddings
embeddings(chunk_id, model_id, dims, vector BLOB, ...)
```

**Search capabilities:**
- `lexical_search()` - FTS5 with BM25 ranking
- `vector_search()` - Cosine similarity (brute-force for now)
- Planned: FAISS for approximate nearest neighbors

### 4. Retriever (Query Engine)

**What:** Hybrid search combining lexical + semantic
**Code:** `src/ragtriever/retrieval/retriever.py`
**Entry Point:** `Retriever.hybrid_search()`

**Algorithm:**
```python
# 1. Get candidates from both indexes
lex_results = store.lexical_search(query, k=k_lex)  # FTS5
query_embedding = embedder.embed(query)
vec_results = store.vector_search(query_embedding, k=k_vec)

# 2. Merge with RRF (Reciprocal Rank Fusion)
combined = merge_with_rrf(lex_results, vec_results)

# 3. Optional enhancements
if use_graph_boost:
    boost_by_backlinks(combined)  # Leverage [[wikilinks]]
if use_rerank:
    rerank_with_cross_encoder(combined)

# 4. Return top K
return combined[:top_k]
```

---

## Execution Modes

### Mode 1: CLI (Command Line)

**Usage:** Direct terminal commands
**Code:** `src/ragtriever/cli.py` (Typer application)

**Commands:**
```bash
# Initialize config
ragtriever init --vault /path/to/vault --index ~/.ragtriever/indexes/myvault

# Index vault
ragtriever scan --config config.toml --full

# Query
ragtriever query --config config.toml "search term" --k 10

# Continuous indexing
ragtriever watch --config config.toml

# MCP server
ragtriever mcp --config config.toml
```

**Use Cases:**
- Manual indexing and queries
- Shell scripts and automation
- Testing and debugging

### Mode 2: Watch Mode (Continuous Indexing)

**Usage:** `ragtriever watch --config config.toml`
**Implementation:** Filesystem watcher (watchdog library)

**How It Works:**
```python
from watchdog.observers import Observer

# Monitor vault for changes
observer = Observer()
observer.schedule(handler, vault_root, recursive=True)
observer.start()

# On file change/create/delete:
def on_modified(event):
    if not should_ignore(event.src_path):
        indexer._index_one(event.src_path)  # Re-index immediately
```

**Use Cases:**
- Real-time indexing as you work
- Keep vault always up-to-date
- Background daemon for productivity

**Run as daemon:**
```bash
nohup ragtriever watch --config config.toml &
```

### Mode 3: MCP Server (Claude Integration)

**Usage:** `ragtriever mcp --config config.toml`
**Protocol:** Model Context Protocol (stdio transport)
**Code:** `src/ragtriever/mcp/server.py`

**Tools Exposed to Claude:**
```json
{
  "vault.search": "Hybrid search with semantic + lexical",
  "vault.open": "Retrieve full document/chunk content",
  "vault.neighbors": "Find related chunks via [[wikilinks]]",
  "vault.status": "Get indexing status and statistics"
}
```

**Integration (Claude Desktop config.json):**
```json
{
  "mcpServers": {
    "ragtriever": {
      "command": "ragtriever",
      "args": ["mcp", "--config", "/Users/you/vault/config.toml"]
    }
  }
}
```

**Use Cases:**
- Search your vault during Claude conversations
- AI-powered semantic retrieval
- Knowledge-augmented responses

**Example Interaction:**
```
You: "What did I write about kubernetes deployments?"
Claude: [calls vault.search("kubernetes deployments")]
Claude: "Based on your notes, you wrote about..."
```

### Mode 4: Python API (Programmatic)

**Usage:** Import RAGtriever as a Python library

**Example:**
```python
from ragtriever.config import VaultConfig
from ragtriever.indexer import Indexer
from ragtriever.retrieval import Retriever

# Load config
cfg = VaultConfig.from_toml("config.toml")

# Index
indexer = Indexer(cfg)
indexer.scan(full=True)

# Query
retriever = Retriever(cfg)
results = retriever.hybrid_search("kubernetes", top_k=10)

for result in results:
    print(f"{result.score:.2f} - {result.source_ref.rel_path}")
    print(f"  {result.snippet}")
```

**Use Cases:**
- Custom applications
- Jupyter notebooks
- Integration with other tools
- Build your own UI on top

---

## Skills vs Non-Skills

### ❌ RAGtriever is NOT a Claude Code Skill

**Important clarification:**

**RAGtriever is:**
- ✅ A standalone CLI tool (`ragtriever` command)
- ✅ A Python library (import and use)
- ✅ An MCP server (for Claude integration)
- ✅ Independent application with own lifecycle

**RAGtriever is NOT:**
- ❌ A Claude Code skill (no `/ragtriever` slash command)
- ❌ Dependent on Claude to run
- ❌ Part of Claude Code's skill system

### Relationship with Claude

**Optional Integration via MCP:**
- Claude can use RAGtriever as a tool (like `grep`, `git`, `npm`)
- You run `ragtriever mcp` as a separate process
- Claude Desktop connects to it via stdio
- RAGtriever runs independently - works without Claude

**Comparison:**

| Aspect | Claude Code Skill | RAGtriever |
|--------|-------------------|------------|
| Invocation | `/skill-name` | `ragtriever command` |
| Lifecycle | Managed by Claude | Independent process |
| Distribution | `~/.claude/skills/` | `pip install ragtriever` |
| Runtime | Claude agent subprocess | Standalone Python app |
| MCP | Not applicable | Optional MCP server mode |

---

## Data Flow

### Indexing Flow

```
┌────────────┐
│ Filesystem │ Markdown, PDF, PPTX, images
│   Vault    │
└─────┬──────┘
      │
      ▼
┌──────────────┐
│ Change       │ Compare mtime + content_hash
│ Detection    │ Skip unchanged files
└─────┬────────┘
      │ Files to process
      ▼
┌──────────────┐
│ Extractor    │ MarkdownExtractor (frontmatter, wikilinks, tags)
│ (by type)    │ PDFExtractor (pdfplumber)
│              │ PPTXExtractor (python-pptx)
│              │ ImageExtractor (Tesseract/Gemini/Vertex AI)
└─────┬────────┘
      │ Extracted(text, metadata)
      ▼
┌──────────────┐
│ Chunker      │ MarkdownChunker (by heading)
│ (by content) │ BoundaryMarkerChunker (by PAGE/SLIDE/SHEET)
└─────┬────────┘
      │ List[Chunked(anchor_type, anchor_ref, text)]
      ▼
┌──────────────┐
│ Embedder     │ SentenceTransformers (local)
│              │ Ollama (server)
└─────┬────────┘
      │ numpy arrays (vectors)
      ▼
┌──────────────┐
│ Store        │ upsert_documents()
│ (SQLite)     │ upsert_chunks()
│              │ upsert_embeddings()
│              │ Update FTS5 index
└──────────────┘
```

### Query Flow

```
┌──────────┐
│  Query   │ "kubernetes deployment strategies"
│  String  │
└────┬─────┘
     │
     ├──────────────────┬────────────────────┐
     │                  │                    │
     ▼                  ▼                    ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ Lexical     │  │ Semantic     │  │ (Optional)   │
│ FTS5        │  │ Vector       │  │ Graph        │
│ Search      │  │ Search       │  │ Traversal    │
└─────┬───────┘  └──────┬───────┘  └──────┬───────┘
      │                 │                 │
      │  k_lex=40       │  k_vec=40       │  [[links]]
      └────────┬────────┴─────────────────┘
               ▼
        ┌──────────────┐
        │ Hybrid       │ RRF (Reciprocal Rank Fusion)
        │ Ranker       │ Normalize and merge scores
        └──────┬───────┘
               │
               ▼ (Optional)
        ┌──────────────┐
        │ Reranker     │ Cross-encoder model
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Top K        │ Return best matches
        │ Results      │ with snippets and metadata
        └──────────────┘
```

---

## Extension Points

### 1. Custom Extractors

Add support for new file types:

```python
# my_extractor.py
from ragtriever.extractors.base import Extracted
from pathlib import Path

class CustomExtractor:
    supported_suffixes = (".custom",)

    def extract(self, path: Path) -> Extracted:
        text = parse_custom_format(path)
        metadata = {"format_version": "1.0"}
        return Extracted(text=text, metadata=metadata)
```

Register in `src/ragtriever/indexer/extractors.py`.

### 2. Custom Chunkers

Implement custom chunking strategy:

```python
from ragtriever.chunking.base import Chunked

class SemanticChunker:
    def chunk(self, text: str, metadata: dict) -> list[Chunked]:
        # Use ML model to find semantic boundaries
        chunks = ml_segment(text)
        return [
            Chunked(
                anchor_type="semantic",
                anchor_ref=str(i),
                text=chunk,
                metadata=metadata
            )
            for i, chunk in enumerate(chunks)
        ]
```

### 3. Custom Embedders

Use a different embedding model:

```python
class CustomEmbedder:
    def __init__(self, model_name: str):
        self.model = load_custom_model(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        # Return shape: (len(texts), embedding_dim)
        return self.model.encode(texts)
```

### 4. Custom Stores

Replace SQLite with another backend:

```python
# Implement Store protocol
class PostgresStore:
    def upsert_document(self, doc: Document): ...
    def upsert_chunks(self, chunks: list[Chunk]): ...
    def upsert_embeddings(self, ...): ...
    def lexical_search(self, query: str, k: int): ...
    def vector_search(self, embedding: np.ndarray, k: int): ...
```

---

## Configuration

How config flows through the system:

```
config.toml
    │
    ├── [vault]
    │   ├── root ──────────────▶ Indexer.vault_root
    │   └── ignore ────────────▶ Change Detection
    │
    ├── [embeddings]
    │   ├── provider ──────────▶ SentenceTransformers vs Ollama
    │   ├── model ─────────────▶ "BAAI/bge-small-en-v1.5"
    │   ├── device ────────────▶ cpu/cuda/mps
    │   └── offline_mode ──────▶ HF_HUB_OFFLINE env var
    │
    ├── [image_analysis]
    │   └── provider ──────────▶ Extractor selection:
    │       ├── tesseract ─────▶ TesseractImageExtractor
    │       ├── gemini ────────▶ GeminiImageExtractor
    │       └── vertex_ai ─────▶ VertexAIImageExtractor
    │
    ├── [vertex_ai]           (if provider = vertex_ai)
    │   ├── project_id
    │   ├── credentials_file
    │   ├── location
    │   └── model
    │
    └── [retrieval]
        ├── k_vec ─────────────▶ Retriever.k_vec (40)
        ├── k_lex ─────────────▶ Retriever.k_lex (40)
        └── top_k ─────────────▶ Final result count (10)
```

---

## Performance

### Indexing Speed Factors
- **Embedder device:** `mps` (Mac M1/M2) > `cuda` (NVIDIA) > `cpu`
- **Batch size:** Larger = faster (if you have RAM)
- **Model size:** `all-MiniLM-L6-v2` (384d) faster than `bge-base` (768d)
- **Image analysis:** `tesseract` (local) > `vertex_ai` (API calls)

### Query Speed Factors
- **Vector search:** Brute-force (slow for >10K chunks)
  - Future: FAISS for approximate NN
- **k_vec/k_lex:** Lower = faster
- **FTS5:** Very fast (built-in SQLite)

### Storage
- **~100MB per 1000 documents** (embeddings dominate)
- Embedding size: `num_chunks × embedding_dim × 4 bytes`

---

## Next Steps

- **Setup Guide:** [vertex_ai_setup.md](vertex_ai_setup.md)
- **Troubleshooting:** [troubleshooting.md](troubleshooting.md)
- **Improvements:** [../IMPROVEMENTS.md](../IMPROVEMENTS.md)
- **User Guide:** [../README.md](../README.md)
- **Code Details:** [../CLAUDE.md](../CLAUDE.md)
