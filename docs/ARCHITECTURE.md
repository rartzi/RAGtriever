# Architecture — CortexIndex Local

## High-level diagram

```
Vault roots (filesystem)
   │
   ├─► Change Detection
   │     - Watch (fs events, debounced)
   │     - Scan reconcile (periodic/manual)
   │
   ├─► Ingestion Pipeline (per file)
   │     - Extract (type adapter)
   │     - Chunk (type strategy)
   │     - Embed (local model)
   │     - Upsert stores
   │
   ├─► Stores (local)
   │     - Docstore (SQLite/libSQL): docs, chunks, metadata, manifest
   │     - Lexical index (SQLite FTS5): chunk text search
   │     - Vector index (adapter): SQL vectors / LanceDB / Qdrant / FAISS
   │     - Link graph: outlinks/backlinks
   │
   └─► Retrieval Service
         - Hybrid retrieve (vector + lexical)
         - Merge/dedupe + filters
         - Optional graph boost + rerank
         - Return structured citations
              │
              ├─► Python API (Retriever)
              └─► MCP server (tools over stdio)
```

## Design principles
- **Derived index**: filesystem is source of truth; index can rebuild.
- **Idempotent updates**: deterministic IDs + manifest hashing.
- **Pluggable adapters**: file extractors, chunkers, embedders, stores.
- **Citations first**: every result includes a `SourceRef` that can be opened.
- **Obsidian-aware parsing** (without Obsidian dependency):
  - YAML frontmatter
  - `[[wikilinks]]`, `![[embeds]]`, `#tags`

## Retrieval strategy (default)
1) lexical candidates via FTS
2) semantic candidates via vector store
3) merge + dedupe
4) score with weights + optional recency/tag/graph boosts
5) optional rerank
6) package context bundles per source (avoid dumping 20 fragments from one PDF)
