# VaultRAG Local (spec + skeleton)

docs folder Tcontains:
- Product requirements (PRD) and implementation plan
- A Python package skeleton (`vaultrag`) for a **local-only** vault indexer + retriever
- An MCP server stub (stdio JSON-RPC style) exposing tools to agents (Codex / Claude Code / Gemini)
- Example config + example query fixtures

> This is intentionally a **build-ready skeleton**, not a complete implementation.
> Feed this repo to a coding agent and instruct it to implement the TODOs.

## Quick start (after implementation)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
vaultrag init --vault "/path/to/vault" --index "~/.vaultrag/indexes/myvault"
vaultrag scan --full
vaultrag query "caching decision" --k 10 --path Meetings/
vaultrag mcp   # run MCP server over stdio
```

## Design goals
- Local-only: no data leaves the machine
- Decoupled from Obsidian but Obsidian-aware:
  - YAML frontmatter
  - `[[wikilinks]]`, `![[embeds]]`, `#tags`
- Continuous sync with watch+reconcile
- Hybrid retrieval (vector + lexical) + optional graph boosts
- Expose retrieval via Python API and MCP tools

## Docs
- [PRD](docs/PRD.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [MCP Tool Spec](docs/MCP_TOOL_SPEC.json)

Generated: 2026-01-15
