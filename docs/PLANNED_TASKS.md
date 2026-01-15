# Tasks for coding agent (Codex / Claude Code / Gemini)

## Objective
Implement CortexIndex Local as specified in docs/PRD.md with a working local-only indexer + retriever and MCP server.

## Priorities
1) Markdown end-to-end MVP (scan, query, open) with citations
2) Watch mode + reconciliation scan; correct delete/move handling
3) Hybrid retrieval (FTS + vector) + metadata filters
4) Add PDF, PPTX, XLSX, image extractors
5) Link graph + neighbors tool and graph-boosting
6) Hardening: migrations, rebuild, tests

## Definition of Done (v1)
- `cortex scan --full` indexes a test vault
- `cortex query "..."` returns relevant results
- `cortex open` returns the full anchored section/page/slide/sheet
- `cortex watch` updates index on edit/rename/delete
- `cortex mcp` exposes tools: vault.search, vault.open, vault.status, vault.neighbors
- All data stays local; no network calls by default

## Implementation notes
- Keep adapters pluggable: Extractors, Chunkers, Embedders, Stores.
- Use deterministic IDs and a manifest table to ensure idempotency.
- Use SQLite FTS5 for lexical search.
- Provide at least one working vector-store implementation (even if brute force) to enable semantic search locally.
