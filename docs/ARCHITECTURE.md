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

### High-Level Architecture (Bidirectional)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           RAGtriever                                │
│                                                                     │
│  ┌──────────────┐      ┌──────────────┐      ┌─────────────────┐  │
│  │   Vault      │──────▶│   Indexer    │─────▶│   Store         │  │
│  │ (Filesystem) │      │  (Pipeline)  │      │   (SQLite)      │  │
│  │ Source of    │      │  Extract     │      │  ┌────────────┐ │  │
│  │ Truth        │      │  Chunk       │      │  │ documents  │ │  │
│  └──────────────┘      │  Embed       │      │  │ chunks     │ │  │
│                        │  Store       │      │  │ fts_chunks │ │  │
│                        └──────────────┘      │  │ embeddings │ │  │
│                                              │  │ links      │ │  │
│                                              │  └────────────┘ │  │
│                                              └────────┬────────┘  │
│                                                       │           │
│                                                       │           │
│                                              ┌────────▼────────┐  │
│                                              │   Retriever     │  │
│   User Interfaces                            │   (Hybrid)      │  │
│   ┌─────────┐   ┌──────────┐   ┌────────┐  │  • Lexical FTS5 │  │
│   │   CLI   │   │  Python  │   │  MCP   │  │  • Vector cosine│  │
│   │ ragtriever  │   API    │   │ Server │  │  • RRF merge    │  │
│   └────┬────┘   └────┬─────┘   └───┬────┘  └────────┬────────┘  │
│        │             │             │                  │           │
│        │   Query     │    Query    │     Query        │           │
│        └─────────────┴─────────────┴──────────────────┘           │
│        ┌─────────────┬─────────────┬──────────────────┐           │
│        │   Results   │   Results   │     Results      │           │
│        ▼             ▼             ▼                  ▼           │
└─────────────────────────────────────────────────────────────────────┘

KEY FLOWS:
  ──▶  Indexing: Vault → Indexer → Store (write path)
  ◀── Retrieval: User Interface → Retriever → Store → Results (read path)
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
│   ├── project.md           # Markdown with [[wikilinks]] and ![](images)
│   └── meeting-notes.md
├── attachments/
│   ├── diagram.png          # Standalone images for vision analysis
│   ├── whitepaper.pdf       # PDFs with embedded charts/diagrams
│   └── presentation.pptx    # PowerPoint with slide images
└── .obsidian/              # Ignored by default
```

**Key Points:**
- RAGtriever reads from vault but never modifies it
- Automatically extracts images embedded in PDFs and PowerPoints
- Analyzes images referenced in Markdown (`![](path)` and `![[image]]`)

### 2. Indexer (Processing Pipeline)

**What:** Converts files into searchable chunks
**Code:** `src/ragtriever/indexer/indexer.py`
**Entry Point:** `Indexer.scan()` or `Indexer.watch()`

**Unified Processing (`_process_file()`):**

Both scan and watch modes use a shared `_process_file()` method that:
- Is thread-safe (no DB writes, no shared mutable state)
- Returns a `ProcessResult` with chunks and image tasks
- Handles errors gracefully (returns error in result, doesn't raise)
- Generates deterministic IDs (same file = same doc_id/chunk_ids)

```python
# Single file processing (thread-safe, no DB writes)
result = indexer._process_file(abs_path)  # → ProcessResult

# ProcessResult contains:
# - chunks: list[ChunkData]      # Ready for embedding
# - image_tasks: list[ImageTask] # Ready for image processing
# - links: list[tuple]           # Wikilinks for graph
# - error: str | None            # If processing failed
# - skipped: bool                # If file should be skipped
```

**Pipeline Phases:**
```
Phase 0: Reconciliation (detect deleted files)
Phase 1: Parallel extraction via _process_file()
    - ThreadPoolExecutor(extraction_workers)
    - Each worker: validate → extract → chunk → build metadata
    - Returns ProcessResult with chunks + image_tasks

Phase 2: Batched embedding and storage
    - Collect chunks across files
    - Batch embed (embed_batch_size chunks)
    - Batch write to SQLite

Phase 3: Parallel image analysis
    - ThreadPoolExecutor(image_workers)
    - Process embedded/referenced images
    - Create additional searchable chunks
```

**Pluggable via Protocol classes:**
- **Extractors:** One per file type (`.md`, `.pdf`, `.pptx`, `.png`)
  - Extract text, metadata, and embedded images
  - PDF/PPTX extractors store image metadata for post-processing
- **Chunkers:** Split text semantically (by heading, by page, etc.)
- **Embedders:** Generate vector representations (local or API)
- **Image Analyzers:** Process images via Tesseract OCR, Gemini, or Vertex AI

### 3. Store (Persistence Layer)

**What:** SQLite database with hybrid index
**Location:** `~/.ragtriever/indexes/{vault_name}/vaultrag.sqlite`
**Code:** `src/ragtriever/store/libsql_store.py`

**Tables:**
```sql
-- File metadata
documents(doc_id, vault_id, rel_path, file_type, mtime, content_hash, ...)

-- Semantic segments (text + image analysis)
chunks(chunk_id, doc_id, anchor_type, anchor_ref, text, ...)
-- anchor_type examples: md_heading, page, slide, pdf_image, pptx_image, markdown_image

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

## Advanced Features (v1.0+)

### 1. Chunk Overlap for Context Preservation

**Problem:** Traditional chunking splits text at heading boundaries, which can lose context. When a chunk ends mid-thought and the next begins, semantic relationships are broken.

**Solution:** Configurable overlap between adjacent chunks (default: 200 characters).

**Implementation:**
- **MarkdownChunker:** Adds suffix from previous section as prefix to current section
- **BoundaryMarkerChunker:** Adds overlap between pages/slides/sheets
- **Metadata:** Chunks track `has_prefix_overlap: true` to indicate context preservation

**Configuration:**
```toml
[chunking]
overlap_chars = 200              # Characters of overlap
max_chunk_size = 2000            # Split large sections
preserve_heading_metadata = true  # Keep heading hierarchy
```

**Benefits:**
- 📈 **Better retrieval accuracy** - Context at boundaries preserved
- 🔍 **Reduced information loss** - Concepts spanning chunks remain connected
- 📊 **Measurable impact** - Typical vaults see 80-85% of chunks with overlap

**Code:**
```python
# src/ragtriever/chunking/markdown_chunker.py
def chunk(self, extracted_text, extracted_metadata):
    sections = self._extract_sections(extracted_text)
    chunks = []

    for i, (heading, level, body) in enumerate(sections):
        # Add overlap from previous section
        prefix = ""
        if i > 0 and self.overlap_chars > 0:
            prev_body = sections[i-1][2]
            prefix = prev_body[-self.overlap_chars:].strip() + "\n\n"

        full_text = prefix + body
        chunks.append(Chunked(
            text=full_text,
            metadata={"has_prefix_overlap": len(prefix) > 0, ...}
        ))

    return chunks
```

### 2. Query Instruction Prefix (Asymmetric Retrieval)

**Problem:** BGE-style embedding models perform better with asymmetric retrieval - queries and documents should be embedded differently.

**Solution:** Apply instruction prefix to query embeddings only (not documents).

**Implementation:**
- **Documents:** Embedded without prefix (represents content)
- **Queries:** Prefixed with instruction (represents search intent)
- **Default prefix:** `"Represent this sentence for searching relevant passages: "`

**Configuration:**
```toml
[embeddings]
use_query_prefix = true
query_prefix = "Represent this sentence for searching relevant passages: "
```

**Code:**
```python
# src/ragtriever/embeddings/sentence_transformers.py
class SentenceTransformersEmbedder:
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents (no prefix)."""
        return self._model.encode(list(texts), ...)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with instruction prefix."""
        if self.use_query_prefix and self.query_prefix:
            query = self.query_prefix + query
        return self._model.encode([query], ...)[0]
```

**Benefits:**
- 🎯 **Improved relevance** - Query-document matching optimized for BGE models
- 🔬 **Model-specific tuning** - Follows best practices for asymmetric models
- 🔄 **Backward compatible** - Can be disabled via config

**Retriever Update:**
```python
# src/ragtriever/retrieval/retriever.py
def hybrid_search(self, query: str, top_k: int = 10):
    # Use embed_query instead of embed_texts
    query_embedding = self.embedder.embed_query(query)  # ← New
    vec_results = self.store.vector_search(query_embedding, k=self.k_vec)
    # ... rest of hybrid search
```

### 3. Cross-Encoder Reranking (Optional)

**Problem:** Hybrid search (FTS5 + vectors + RRF) can return noisy candidates in top results. Bi-encoders encode query and documents separately, missing query-document interaction signals.

**Solution:** Rerank top candidates using a cross-encoder model that reads query + document together for accurate relevance scoring.

**Configuration:**
```toml
[retrieval]
use_rerank = true
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_device = "cpu"  # or "cuda", "mps"
rerank_top_k = 10
```

**How It Works:**
```python
# Stage 1: Fast candidate retrieval (hybrid search)
candidates = hybrid_search(query, k_vec=40, k_lex=40)  # FTS5 + vectors + RRF

# Stage 2: Accurate reranking (cross-encoder)
if use_rerank:
    pairs = [(query, candidate.snippet) for candidate in candidates]
    scores = cross_encoder.predict(pairs)
    results = sort_by_scores(candidates, scores)[:top_k]
```

**Performance:**
- Adds ~100-200ms latency (40 candidates on CPU)
- GPU: ~20-50ms
- Quality improvement: +20-30% nDCG@10

**When to Enable:**
- You want best possible result quality
- Latency tolerance: <300ms acceptable
- Getting false positives or poor result ordering
- Complex queries with multiple concepts

**Why It Works:**
- **Bi-encoder (retrieval):** Encodes query/doc separately → fast but approximate
- **Cross-encoder (reranking):** Reads query + doc together → slow but accurate
- **Best of both worlds:** Bi-encoder retrieves 40 candidates, cross-encoder refines to top 10

**Example Results:**

Before (hybrid search only):
```
Query: "kubernetes deployment strategies"
1. ✅ Kubernetes deployment guide (0.89)
2. ❌ AWS Lambda deployment (0.82) ← Noise
3. ✅ K8s production deployment (0.78)
4. ❌ Jenkins CI/CD pipeline (0.75) ← Noise
5. ✅ Helm chart deployment (0.71)
```

After (with reranking):
```
Query: "kubernetes deployment strategies"
1. ✅ Kubernetes deployment guide (0.94)
2. ✅ K8s production deployment (0.91)
3. ✅ Helm chart deployment (0.87)
4. ✅ Blue-green deployment k8s (0.82)
5. ✅ Rolling updates kubernetes (0.79)
```

**Code Reference:**
- `src/ragtriever/retrieval/reranker.py` - Cross-encoder implementation
- `src/ragtriever/retrieval/retriever.py:search()` - Integration point

**CLI Usage:**
```bash
# Enable via config
[retrieval]
use_rerank = true

# Or override per query
ragtriever query "kubernetes deployment" --k 10 --rerank
```

### 4. FAISS Index Support (Planned for Large Vaults)

**Problem:** Brute-force vector search becomes slow at >10K chunks (100ms-1s latency).

**Solution:** FAISS approximate nearest neighbor index for sub-linear search time.

**Status:** Infrastructure complete, disabled by default. Enable when vault grows large.

**Configuration:**
```toml
[embeddings]
use_faiss = false              # Enable for >10K chunks
faiss_index_type = "IVF"       # "Flat" (exact), "IVF" (fast), "HNSW" (fastest)
faiss_nlist = 100              # Clusters for IVF
faiss_nprobe = 10              # Clusters to search (IVF)
```

**Performance Targets:**
- Brute-force: ~100ms-1s for 10K chunks
- FAISS IVF: ~20-50ms for 100K chunks
- FAISS HNSW: ~10-30ms for 100K chunks

**When to Enable:**
- Vault has >10,000 chunks
- Vector search latency becomes noticeable
- Accept 95-99% recall vs 100% (approximate search tradeoff)

**Code:** `src/ragtriever/store/faiss_index.py` (FAISSIndex wrapper class)

### Design Decisions

**Why overlap instead of larger chunks?**
- Larger chunks dilute semantic meaning (less precise retrieval)
- Overlap preserves boundaries while maintaining focused chunks
- Configurable: users can disable (set `overlap_chars = 0`)

**Why query prefix only (not document prefix)?**
- Documents represent content - should be natural text
- Queries represent search intent - benefit from task framing
- Asymmetric approach matches BGE model training methodology

**Why FAISS disabled by default?**
- Small vaults (<10K chunks) see no benefit from approximate search
- Adds complexity (training, index management)
- Users can enable when scaling needs arise

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

def on_deleted(event):
    if not should_ignore(event.src_path):
        store.delete_document(vault_id, rel_path)  # Remove from index
```

**Use Cases:**
- Real-time indexing as you work
- Keep vault always up-to-date
- Background daemon for productivity

**File Lifecycle Handling:**

Both scan and watch modes handle the complete file lifecycle, including directory-level operations:

| Event | Scan Mode | Watch Mode |
|-------|-----------|------------|
| **File Add** | Indexed on scan | Indexed immediately |
| **File Change** | Re-indexed on scan | Re-indexed immediately |
| **File Delete** | Detected via reconciliation | Detected via filesystem event |
| **Directory Add** | All files indexed on scan | All files indexed immediately |
| **Directory Delete** | Files detected via reconciliation | Queries DB for files under path, removes all |
| **Directory Move** | Old paths deleted, new paths added | Queries DB for files, updates all paths atomically |

**Deletion Detection (Scan Mode):**
```python
# Phase 0 of scan: Reconciliation
fs_files = {rel_path for p in scan_files()}  # Files on disk
indexed_files = store.get_indexed_files(vault_id)  # Files in DB

# Find deleted files (works for individual files and entire directories)
deleted = indexed_files - fs_files
for rel_path in deleted:
    store.delete_document(vault_id, rel_path)  # Full cleanup
```

**Directory Operations (Watch Mode):**
```python
# Directory deletion
def on_deleted(event):
    if event.is_directory:
        # Query all indexed files under directory path
        files_under = store.get_files_under_path(vault_id, rel_path)
        for file_rel in files_under:
            queue.put(Job(kind="delete", rel_path=file_rel))

# Directory move
def on_moved(event):
    if event.is_directory:
        # Query all indexed files and update their paths
        files_under = store.get_files_under_path(vault_id, old_rel_path)
        for file_rel in files_under:
            new_file_rel = file_rel.replace(old_rel_path, new_rel_path, 1)
            queue.put(Job(kind="move", rel_path=file_rel, new_rel_path=new_file_rel))
```

**What Gets Cleaned Up on Deletion:**
- Document marked as `deleted=1`
- All chunks removed
- All embeddings removed
- FTS entries removed
- Outgoing links removed
- Manifest entry removed

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
  "vault_search": "Hybrid search with semantic + lexical",
  "vault_open": "Retrieve full document/chunk content",
  "vault_neighbors": "Find related chunks via [[wikilinks]]",
  "vault_status": "Get indexing status and statistics",
  "vault_list": "List configured vaults"
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
Claude: [calls vault_search("kubernetes deployments")]
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
│ Extractor    │ MarkdownExtractor (frontmatter, wikilinks, tags, image refs)
│ (by type)    │ PDFExtractor (pdfplumber + image metadata)
│              │ PPTXExtractor (python-pptx + image bytes)
│              │ ImageExtractor (Tesseract/Gemini/Vertex AI)
└─────┬────────┘
      │ Extracted(text, metadata{embedded_images, image_references})
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
      │
      ▼
┌──────────────┐
│ Process      │ For each embedded image:
│ Embedded     │ 1. Extract bytes (PyMuPDF for PDF, direct for PPTX)
│ Images       │ 2. Save to temp file
│              │ 3. Pass to ImageExtractor (Tesseract/Gemini/Vertex AI)
│              │ 4. Create separate chunk (anchor_type: pdf_image/pptx_image/markdown_image)
│              │ 5. Link to parent document via doc_id
│              │ 6. Embed image analysis text
│              │ 7. Store as searchable chunk
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

## Detailed Sequence Diagrams

### Indexing Flow (Scan → Embed → Store)

This shows the complete sequence when a file is indexed:

```
Filesystem   Indexer      Extractor    Chunker    Embedder      Store (SQLite)
    │            │            │           │           │              │
    │  file.md   │            │           │           │              │
    ├───────────▶│            │           │           │              │
    │            │            │           │           │              │
    │            │ extract()  │           │           │              │
    │            ├───────────▶│           │           │              │
    │            │            │ Parse MD  │           │              │
    │            │            │ wikilinks │           │              │
    │            │            │ tags      │           │              │
    │            │◀───────────┤           │           │              │
    │            │ Extracted  │           │           │              │
    │            │ (text,     │           │           │              │
    │            │  metadata) │           │           │              │
    │            │            │           │           │              │
    │            │ chunk()    │           │           │              │
    │            ├────────────┴──────────▶│           │              │
    │            │            │ Split by  │           │              │
    │            │            │ headings  │           │              │
    │            │◀───────────────────────┤           │              │
    │            │ List[Chunked]          │           │              │
    │            │ (anchor_type, text)    │           │              │
    │            │            │           │           │              │
    │            │ embed()    │           │           │              │
    │            ├────────────┴───────────┴──────────▶│              │
    │            │            │           │ Model     │              │
    │            │            │           │ inference │              │
    │            │◀───────────────────────────────────┤              │
    │            │ np.ndarray (N × embedding_dim)     │              │
    │            │            │           │           │              │
    │            │ upsert_document()      │           │              │
    │            ├────────────┴───────────┴───────────┴─────────────▶│
    │            │            │           │           │  INSERT INTO │
    │            │            │           │           │  documents   │
    │            │            │           │           │              │
    │            │ upsert_chunks()        │           │              │
    │            ├────────────┴───────────┴───────────┴─────────────▶│
    │            │            │           │           │  INSERT INTO │
    │            │            │           │           │  chunks      │
    │            │            │           │           │  INSERT INTO │
    │            │            │           │           │  fts_chunks  │
    │            │            │           │           │  (FTS5)      │
    │            │            │           │           │              │
    │            │ upsert_embeddings()    │           │              │
    │            ├────────────┴───────────┴───────────┴─────────────▶│
    │            │            │           │           │  INSERT INTO │
    │            │            │           │           │  embeddings  │
    │            │            │           │           │  (BLOB)      │
    │            │◀───────────────────────────────────────────────────┤
    │            │            │           │           │  Committed   │
    │◀───────────┤            │           │           │              │
    │  indexed   │            │           │           │              │
```

**Key Steps:**
1. **Extract** - File-type specific parsing (Markdown/PDF/PPTX/Image)
2. **Chunk** - Semantic segmentation (by heading, page, slide)
3. **Embed** - Vector generation using embedding model
4. **Store** - Write to THREE SQLite locations:
   - `documents` table (metadata)
   - `chunks` table + `fts_chunks` FTS5 (text + lexical index)
   - `embeddings` table (vectors as BLOBs)

### Retrieval Flow (Query → Hybrid Search → Results)

This shows the bidirectional query flow from user interface back to results:

```
CLI/API/MCP   Retriever   Embedder    Store (SQLite)
    │            │           │              │
    │  query     │           │              │
    │ "k8s"      │           │              │
    ├───────────▶│           │              │
    │            │           │              │
    │            ├─────── Parallel Search ───────┐
    │            │           │              │    │
    │            │ embed_query()            │    │
    │            ├──────────▶│              │    │
    │            │           │ Model        │    │
    │            │◀──────────┤              │    │
    │            │ query_vec │              │    │
    │            │           │              │    │
    │            │ vector_search(query_vec) │    │
    │            ├──────────┴─────────────▶│    │
    │            │           │  SELECT e.chunk_id, e.vector, c.text
    │            │           │  FROM embeddings e
    │            │           │  JOIN chunks c ON c.chunk_id=e.chunk_id
    │            │           │  WHERE c.vault_id=?
    │            │           │              │    │
    │            │           │  • Load vectors from BLOB
    │            │           │  • Cosine similarity: dot(q,v)/(||q||*||v||)
    │            │           │  • Sort by similarity
    │            │◀──────────────────────────┤    │
    │            │ vec_results (k_vec=40)   │    │
    │            │ [SearchResult]           │    │
    │            │           │              │    │
    │            │ lexical_search(query)    │    │
    │            ├──────────┴─────────────▶│    │
    │            │           │  SELECT chunk_id, bm25(fts_chunks) AS rank
    │            │           │  FROM fts_chunks
    │            │           │  WHERE fts_chunks MATCH "k8s"
    │            │           │  ORDER BY rank LIMIT 40
    │            │           │              │    │
    │            │◀──────────────────────────┤    │
    │            │ lex_results (k_lex=40)   │    │
    │            │ [SearchResult]           │    │
    │            │           │              │    │
    │            └────────── Merge Results ──────┘
    │            │           │              │
    │            │ HybridRanker.merge()    │
    │            │ • Reciprocal Rank Fusion │
    │            │ • Deduplicate            │
    │            │ • Normalize scores       │
    │            │ • Return top_k=10        │
    │            │           │              │
    │◀───────────┤           │              │
    │  results   │           │              │
    │  [10 items]│           │              │
```

**Key Steps:**
1. **Embed Query** - Convert search string to vector
2. **Parallel Search** - Two simultaneous queries:
   - **Vector Search**: Cosine similarity on `embeddings` table (semantic)
   - **Lexical Search**: BM25 on `fts_chunks` FTS5 index (keyword)
3. **Merge** - Hybrid ranking using Reciprocal Rank Fusion (RRF)
4. **Return** - Top K results flow back through interfaces

### SQLite: Vector vs Non-Vector Storage

SQLite serves as a **hybrid database** storing both traditional relational data and vector embeddings:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQLite Database Schema                       │
│                    (vaultrag.sqlite)                            │
│                                                                 │
│  NON-VECTOR STORAGE (Traditional SQL)                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                           │ │
│  │  documents table                                          │ │
│  │  ├─ doc_id, vault_id, rel_path                            │ │
│  │  ├─ file_type, mtime, size                                │ │
│  │  └─ content_hash, metadata_json                           │ │
│  │                                                           │ │
│  │  chunks table                                             │ │
│  │  ├─ chunk_id, doc_id, vault_id                            │ │
│  │  ├─ anchor_type, anchor_ref                               │ │
│  │  └─ text, text_hash, metadata_json                        │ │
│  │                                                           │ │
│  │  links table                                              │ │
│  │  ├─ vault_id, src_rel_path, dst_target                    │ │
│  │  └─ link_type (wikilink, embed)                           │ │
│  │                                                           │ │
│  │  manifest table                                           │ │
│  │  └─ Tracks indexing status, errors                        │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  LEXICAL SEARCH (FTS5 Virtual Table)                           │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                           │ │
│  │  fts_chunks USING fts5                                    │ │
│  │  ├─ chunk_id (UNINDEXED)                                  │ │
│  │  ├─ vault_id (UNINDEXED)                                  │ │
│  │  ├─ rel_path (UNINDEXED)                                  │ │
│  │  └─ text (INDEXED for full-text search)                   │ │
│  │                                                           │ │
│  │  • BM25 ranking algorithm                                 │ │
│  │  • Tokenized with unicode61                               │ │
│  │  • Query: SELECT bm25(fts_chunks) AS rank FROM fts_chunks │ │
│  │           WHERE fts_chunks MATCH "search term"            │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  VECTOR STORAGE (Embeddings as BLOBs)                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                           │ │
│  │  embeddings table                                         │ │
│  │  ├─ chunk_id (FK to chunks)                               │ │
│  │  ├─ model_id (e.g., "BAAI/bge-small-en-v1.5")            │ │
│  │  ├─ dims (embedding dimension, e.g., 384)                 │ │
│  │  └─ vector BLOB (numpy float32 array serialized)          │ │
│  │                                                           │ │
│  │  • Storage: np.asarray(vec, dtype=np.float32).tobytes()  │ │
│  │  • Retrieval: np.frombuffer(blob, dtype=np.float32)      │ │
│  │  • Search: Brute-force cosine similarity (currently)      │ │
│  │  • Future: FAISS index for approximate NN                 │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Storage Breakdown:**

| Data Type | SQLite Storage | Purpose | Query Method |
|-----------|---------------|---------|--------------|
| **Metadata** | `documents`, `chunks`, `links` tables | File info, chunk text, wikilink graph | Standard SQL queries |
| **Lexical Index** | `fts_chunks` FTS5 virtual table | Full-text search with BM25 | `MATCH` operator, BM25 ranking |
| **Vector Embeddings** | `embeddings.vector` BLOB column | Semantic search vectors | Load all, compute cosine similarity |

**Why this hybrid approach?**

1. **Single database file** - Simplifies deployment and backup
2. **Transaction safety** - All writes are atomic within SQLite
3. **No external dependencies** - Works everywhere SQLite works
4. **Future-proof** - Can add vector index extensions (e.g., libSQL vector search)

**Current limitation:** Vector search is brute-force O(N) - loads all embeddings and computes cosine similarity in Python. For >10K chunks, consider:
- FAISS index (approximate nearest neighbors)
- libSQL vector extensions (when available)
- Separate vector store (Qdrant, Milvus)

**Performance characteristics:**
- **FTS5 lexical search**: Very fast (indexed), ~1-10ms for typical queries
- **Vector search**: Slower (brute-force), ~100ms-1s for 10K chunks
- **Hybrid merge**: Fast (RRF on small result sets), ~1-5ms

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
    │       ├── tesseract ─────▶ TesseractImageExtractor (local OCR)
    │       ├── gemini ────────▶ GeminiImageExtractor (API, OAuth2)
    │       └── vertex_ai ─────▶ VertexAIImageExtractor (service account)
    │
    │   Note: Automatically processes:
    │   • Images embedded in PDFs (extracted via PyMuPDF)
    │   • Images on PowerPoint slides (extracted via python-pptx)
    │   • Images referenced in Markdown (![](path), ![[image]])
    │   • Standalone image files (.png, .jpg, .jpeg, .webp, .gif)
    │   Creates separate searchable chunks linked to parent documents.
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
