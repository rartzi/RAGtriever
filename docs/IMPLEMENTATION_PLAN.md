# Implementation plan — CortexIndex Local

## Phase 1: Markdown MVP
- Config + CLI skeleton
- SQLite docstore + FTS5
- Markdown extraction: frontmatter + headings + tags + wikilinks
- Chunking by headings + preserve code blocks
- Embeddings via SentenceTransformers (local)
- Vector store adapter (initial: brute-force in-memory for small sets OR placeholder)
- Retrieval: lexical-only first, then add vector
- MCP tools: `vault.search`, `vault.open`, `vault.status`

## Phase 2: Sync engine
- Watch mode (watchdog) with debounce + job queue
- Manual scan reconciliation
- Correct handling of create/modify/delete/move
- Crash-safe upserts and tombstones

## Phase 3: Hybrid retrieval + filters
- Hybrid candidate generation (FTS + vector)
- Merge/dedupe, metadata filters (path/type/tags/mtime)
- Basic scoring & result packaging

## Phase 4: Attachments
- PDF extractor (page-aware)
- PPTX extractor (slide-aware)
- XLSX extractor (sheet/table-aware)
- Image extractor (OCR optional)

## Phase 5: Link graph
- Persist outlinks and compute backlinks
- `vault.neighbors` tool
- Graph boosts/expansion in ranking

## Phase 6: Hardening
- Deterministic IDs, migrations, rebuild tools
- Optional reranker adapter
- Evaluation harness with query fixtures
