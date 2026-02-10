# Exploration: Agentic Search vs RAG for Complex Queries

**Date:** 2026-02-09
**Branch:** `explore/rag-exploration-search`
**Status:** Research complete, awaiting implementation decision

## Summary

This document captures research comparing Mneme's current hybrid RAG pipeline against agentic search approaches used by Anthropic (Claude Code), the PromtEngineer/agentic-file-search project, and major industry players. The goal is to determine how Mneme should evolve to handle complex queries that single-pass RAG cannot address.

## Key Finding

**RAG is not dead — it becomes a component inside an agentic architecture.** The industry consensus (Anthropic, OpenAI, Google, Microsoft, Perplexity) is a layered model where simple queries use fast RAG and complex queries trigger an agent loop that calls RAG as one tool among several.

---

## Three Architectures Compared

### 1. Mneme Today — Hybrid RAG (Single-Pass)

```
Query → Embed → [Lexical FTS5 + Vector FAISS] → RRF Fusion → Boosts → Diversity → Rerank → Results
```

**Performance:** 50-200ms warm, 150-300ms cold
**Strengths:** Fast, deterministic, rich signals (backlink boost, recency, reranking)
**Limitations:** No query decomposition, no iterative refinement, no multi-hop, no ambiguity handling

### 2. Claude Code — Pure Agentic Search

```
Query → LLM decides tool → Grep/Glob/Read → Evaluate → Sufficient? → No → Reformulate → Loop
```

**Performance:** 3-30s+ per query
**Key insight:** Anthropic chose regex over vector databases because the model "already understands code structure deeply enough"
**Strengths:** Dynamic strategy, multi-hop, hypothesis-driven, adaptive
**Tradeoffs:** Higher latency, higher cost (3-10x tokens), non-deterministic

### 3. PromtEngineer/agentic-file-search — Zero-Index Document Agent

```
Query → scan_folder → categorize → parse_file (deep read) → backtrack (cross-refs) → Answer
```

**Performance:** 30s-7min per query (Qwen3 32B local)
**Tech:** 6 filesystem tools, Docling parser, LlamaIndex Workflows
**Strengths:** Follows cross-references, preserves document structure, no pre-indexing
**Tradeoffs:** Minimum 30s latency, 2.6K-83K tokens/query, limited to small doc sets

---

## Comparison Matrix

| Dimension | Mneme (Hybrid RAG) | Claude Code (Agentic) | agentic-file-search |
|-----------|--------------------|-----------------------|---------------------|
| **Indexing** | Pre-compute embeddings + FTS5 | None | None |
| **Query latency** | 50-300ms | 3-30s+ | 30s-7min |
| **Token cost/query** | ~1K | 10K-50K | 2.6K-83K |
| **Multi-hop** | No | Yes | Yes |
| **Query decomposition** | No | Yes (LLM decides) | Implicit (agent loop) |
| **Cross-reference following** | No (backlinks = boost only) | Yes | Yes (3-phase) |
| **Scale** | Millions of chunks | Entire codebases | Small doc sets (10-50) |
| **Determinism** | Yes | No | No |

---

## Benchmark Evidence

### PRISM Framework (arxiv 2510.14278)

| Benchmark | Agentic (PRISM) | Single-shot RAG | Relative Improvement |
|-----------|-----------------|-----------------|---------------------|
| HotpotQA Recall | 90.9% | 61.5% | +48% |
| **MuSiQue Recall** | **83.2%** | **44.6%** | **+87%** |
| 2WikiMultiHop Recall | 91.1% | 68.1% | +34% |

**Key pattern:** Performance gap widens with query complexity. Simple lookups show modest improvement; multi-hop queries show 40-87% gains.

### Industry Convergence

- **Azure AI Search:** Agentic retrieval improves relevance by 40% (parallel sub-query decomposition)
- **Anthropic Contextual Retrieval:** 35-67% failure rate reduction (enhanced RAG, not agentic)
- **Google Gemini Deep Research:** RAG used as memory layer *inside* agent, not as primary retrieval
- **Perplexity Deep Research:** 93.9% accuracy on SimpleQA with iterative search

---

## Complex Query Failure Modes in Current Mneme

1. **Multi-hop:** "What topics does the note linked from my daily journal discuss?" — needs link traversal
2. **Comparative:** "How has my thinking about X changed over time?" — needs temporal search + comparison
3. **Synthesizing:** "What are all the open questions across my project notes?" — needs multi-doc pattern extraction
4. **Boolean/filtered:** "Notes about Python that also mention testing" — FTS5 treats as phrase, no AND/OR
5. **Ambiguous:** "Tell me about the bridge" — no disambiguation
6. **Cross-reference:** "What does the note linked from [[Project Alpha]] say?" — needs link traversal + retrieval

---

## Proposed Architecture: Layered Mneme

```
                   ┌─────────────────────┐
                   │   AGENT LAYER       │  ← NEW
                   │  (query planning,   │
                   │   evaluation,       │
                   │   reformulation)    │
                   └────────┬────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
     ┌──────────────┐ ┌──────────┐ ┌──────────────┐
     │  MNEME RAG   │ │  GRAPH   │ │   DIRECT     │
     │  (hybrid     │ │  WALK    │ │   READ       │
     │   search)    │ │  (links) │ │   (full doc) │
     └──────────────┘ └──────────┘ └──────────────┘
```

### Agent Tools

| Tool | Purpose | Implementation Status |
|------|---------|----------------------|
| `mneme_search` | Hybrid retrieval (fast, indexed) | EXISTS: `Retriever.search()` |
| `mneme_graph_walk` | Follow wikilinks from a document | EXISTS: `LibSqlStore.neighbors()` |
| `mneme_read_full` | Read full document content | EXISTS: `LibSqlStore.open()` |
| `mneme_list_docs` | List documents matching criteria | NEW: SQL on `documents` table |
| `mneme_grep` | Regex search across vault | NEW: raw FTS5 or regex on chunks |

### Query Router

| Signal | Route | Rationale |
|--------|-------|-----------|
| Single concept, few keywords | Fast RAG | Embedding similarity sufficient |
| Boolean/comparative | Agent | Needs decomposition + intersection |
| "What does X link to" | Agent (graph walk) | Link traversal, not similarity |
| Multi-hop | Agent | Iterative search + filtering |

### Implementation Phases

1. **Phase 1: Tool Exposure** — Wrap existing functions as callable tools (minimal code)
2. **Phase 2: Query Router** — Classify simple vs complex, route accordingly
3. **Phase 3: Agent Loop** — Iterative search-evaluate-refine with all tools
4. **Phase 4: Contextual Embeddings** — Prepend document context to chunks (orthogonal improvement)

---

## Key Insight: Low-Hanging Fruit

`LibSqlStore.neighbors()` already returns outlinks and backlinks for any document. This graph traversal capability is implemented but only used for boost scoring. Exposing it as an agent tool immediately enables multi-hop "follow the link" queries with zero new code.

---

## Sources

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [PRISM: Agentic Retrieval for Multi-Hop QA](https://arxiv.org/html/2510.14278v1)
- [Azure AI Search Agentic Retrieval](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview)
- [Agentic RAG Survey](https://arxiv.org/html/2501.09136v3)
- [Survey of LLM-based Deep Search Agents](https://arxiv.org/html/2508.05668v3)
- [GitHub: PromtEngineer/agentic-file-search](https://github.com/PromtEngineer/agentic-file-search)
- [Claude Code Master Agent Loop Analysis](https://blog.promptlayer.com/claude-code-behind-the-scenes-of-the-master-agent-loop/)
- [From Classic RAG to Agentic Retrieval: Microsoft Foundry IQ](https://itnext.io/from-classic-rag-to-agentic-retrieval-inside-microsofts-foundry-iq-architecture-7338e1bd4eb4)
