# PRD — RAGtriever Local

## 1) Product summary
**RAGtriever Local** is a local-first daemon + library that continuously indexes a changing filesystem "vault" (Obsidian-compatible Markdown + attachments) into a hybrid retrieval system (semantic + keyword + link-graph) and serves results to coding agents via MCP and Python.

## 2) Goals
- Local-only (no outbound data by default)
- Decoupled from Obsidian (filesystem-based), but Obsidian-aware:
  - YAML frontmatter
  - `[[wikilinks]]` / `![[embeds]]`
  - `#tags`
  - (optional) block refs `^blockid`
- Continuous sync: adds/updates/deletes/moves
- Hybrid retrieval: vector + lexical + link-graph boost/expansion
- Agent-ready interfaces: MCP tools + Python library + optional CLI
- Extensible extractors for new file types

## 3) Non-goals (v1)
- Obsidian plugin (in-app UI)
- Cloud sync / SaaS hosting
- Perfect OCR/table extraction
- Full GraphRAG reasoning (we store/use links, not advanced graph prompting)

## 4) Core use cases
- Find meeting decisions in Markdown notes, with citations to exact sections.
- Search PDFs for requirements and cite page numbers.
- Search decks/spreadsheets with slide/sheet anchors.
- Feed results back to agents as tool calls: `search → open → answer`.

## 5) Functional requirements

### 5.1 Vault configuration
- Configure one or more vault roots (directories).
- Ignore patterns (e.g., `.git/**`, `.obsidian/cache/**`).
- Treat vault as changing filesystem; never depend on Obsidian APIs.

### 5.2 Sync modes
- Watch mode (automatic): filesystem watcher + debounce + job queue.
- Scan mode (manual): reconcile by scanning all files and comparing state.
- Hybrid: watch + periodic scan reconciliation.

### 5.3 Lifecycle events
- Create: index file
- Modify: re-extract/re-chunk; only re-embed changed chunks
- Delete: remove all chunks
- Move/rename: update paths; keep stable doc IDs if possible

### 5.4 Extraction & chunking
Support:
- Markdown (.md): headings, frontmatter, wikilinks, tags; preserve code blocks
- PDF (.pdf): page-aware extraction; optional OCR fallback
- PowerPoint (.pptx): slide-aware extraction
- Excel (.xlsx): sheet/table-aware extraction
- Images (.png/.jpg/.webp): OCR and/or caption text (feature-flagged)

### 5.5 Embeddings
- Local embeddings by default (SentenceTransformers).
- Optional adapter for local embedding servers (e.g., Ollama) if user chooses.
- Store model_id, dims, and versions.

### 5.6 Storage
- Docstore for canonical chunk text + metadata.
- Vector index for semantic retrieval.
- Lexical index (FTS/BM25) for exact matches.
- Persist link graph for graph boosts/expansion.

### 5.7 Retrieval
- Hybrid retrieval: vector + lexical candidates, merge/dedupe.
- Optional reranking stage (pluggable).
- Return results with citations (path + page/slide/sheet/heading anchors).

### 5.8 Interfaces
- Python library: Indexer + Retriever APIs.
- MCP server over stdio exposing tools: `vault.search`, `vault.open`, `vault.neighbors`, `vault.status`.
- CLI: init/scan/watch/query/open/status/rebuild.

## 6) Non-functional requirements
- Offline operation; no telemetry by default.
- Cross-platform (macOS/Windows/Linux).
- Crash-safe and idempotent indexing.
- Extensible extractor/chunker architecture.

## 7) Acceptance criteria (v1)
- Index mixed vault (md + pdf + pptx + xlsx + images) locally.
- Watch mode updates index on edits/renames/deletes.
- Query returns relevant chunks with stable citations.
- MCP tools work from an agent and can open cited sources.
