# AgenticSearch Workflow

Iterative, multi-step search strategy for complex questions that require decomposition, graph traversal, or cross-document synthesis.

## Trigger Phrases
- "deep search", "thorough search", "comprehensive answer"
- Multi-hop questions: "how does X relate to Y?"
- Comparative questions: "compare X and Y in the vault"
- Questions requiring synthesis across multiple documents
- When a simple SearchVault query returns incomplete results

## When to Use

Use AgenticSearch instead of SearchVault when:
- Initial search results don't fully answer the question
- The question spans multiple topics or documents
- You need to follow links between documents
- The answer requires comparing or synthesizing information

## Procedure

### 1. Orient — Understand Vault Structure

```bash
mneme list-docs --config config.toml
mneme list-docs --path "projects/" --config config.toml
```

Get a sense of what's indexed, how files are organized, and where relevant content might live.

### 2. Search — Broad Hybrid Search

```bash
mneme query "initial question keywords" --k 15 --config config.toml
```

Start with the user's question. Note:
- **Vocabulary** used in results (may differ from query terms)
- **File paths** where relevant content lives
- **Gaps** — what aspects of the question aren't covered

### 3. Refine — Targeted Text Search

```bash
mneme text-search "exact phrase from step 2" --config config.toml
mneme text-search "discovered vocabulary term" --path "relevant/folder/" --config config.toml
```

Use BM25 text search for:
- Exact phrases discovered in step 2
- Specific names, acronyms, or technical terms
- Scoping to a directory when you know where content lives

### 4. Connect — Follow the Link Graph

```bash
mneme backlinks --limit 10 --config config.toml
```

Or via MCP: `vault_neighbors` with a specific document path.

Follow wikilinks to:
- Find related documents not returned by keyword search
- Discover hub documents (high backlink count = important)
- Trace chains of reasoning across linked notes

### 5. Read — Get Full Document Content

Via MCP: `vault_open` with the `source_ref` from search results.

Read full content when:
- A chunk snippet is too short to understand context
- You need surrounding sections for complete information
- The document is a hub with many backlinks

### 6. Synthesize — Combine Into Answer

Combine information from all steps into a comprehensive answer.

**Rules:**
- Cite ALL sources that contributed
- Note when information is incomplete or contradictory
- Distinguish between direct quotes and synthesis

## Iterative Loop

```
Orient → Search → Refine ←→ Connect → Read → Synthesize
                    ↑                           |
                    └───── if gaps remain ───────┘
```

Repeat the Refine→Connect→Read cycle until:
- The question is fully answered
- No new relevant results are being discovered
- You've exhausted reasonable search variations (3-5 iterations max)

## Stopping Criteria

Stop searching when:
- All aspects of the question are addressed with citations
- Last 2 iterations returned no new relevant information
- You've reached 5 iterations without finding more

## Example

**User asks:** "How do agentic workflows relate to RAG in my notes?"

```bash
# 1. Orient
mneme list-docs --config config.toml
# → See files in projects/, notes/, images/

# 2. Search broad
mneme query "agentic workflows RAG" --k 15 --config config.toml
# → Results mention "retrieval augmented generation", "agent loop", "tool use"

# 3. Refine with discovered terms
mneme text-search "retrieval augmented generation" --config config.toml
mneme text-search "agent loop" --path "projects/" --config config.toml

# 4. Connect — find hub documents
mneme backlinks --limit 10 --config config.toml
# → projects/alpha.md has 5 backlinks — it's a hub

# 5. Read the hub document (via MCP vault_open)

# 6. Synthesize answer with sources from all steps
```

## Available Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `mneme list-docs` | See vault structure | Orient phase |
| `mneme query` | Hybrid semantic+lexical search | Broad search |
| `mneme text-search` | BM25 lexical search only | Exact phrases |
| `mneme backlinks` | Most-linked documents | Find hubs |
| `vault_neighbors` (MCP) | Outlinks + backlinks for one doc | Follow connections |
| `vault_open` (MCP) | Full document content | Deep reading |
