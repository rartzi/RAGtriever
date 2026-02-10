# SearchVault Workflow

Search and answer questions from indexed vault content using the **Agentic Search Process**.

## Trigger Phrases
- "what does the vault say about..."
- "find in vault..."
- "search for..."
- Any question that should be answered from vault content

## 🚨 MANDATORY: Always Follow the Agentic Search Process

**Never skip steps.** Semantic search alone misses critical content (as proven by the OpenClaw security example where semantic search found only positive articles while Orient revealed critical security warnings in a different folder).

```
Orient → Search → Refine → Connect → Synthesize
   ↑                                      |
   └────── if gaps remain ─────────────────┘
```

---

## Procedure

### Step 1: ORIENT — Understand What's Available (MANDATORY)

Before any semantic search, understand the vault structure:

```bash
# Quick orientation - see what folders/content types exist
mneme list-docs --config config.toml | head -50

# If topic-specific, grep for relevant paths
mneme list-docs --config config.toml | grep -i "keyword" | head -30
```

**Why this matters:** Semantic search returns what matches the query embedding. Orient reveals content that exists but uses different vocabulary or lives in unexpected locations (YouTube transcripts, images, subfolders you didn't know existed).

**Output:** Note any surprising folders or file types that might contain relevant content.

---

### Step 2: SEARCH — Hybrid Semantic + Lexical Query

```bash
mneme query --config config.toml "user's question keywords" --k 15
```

**Options:**
- `--k 15` - Default, good for most questions
- `--k 30` - Broader coverage for complex questions
- `--k 30 --rerank` - Best quality for important questions

**Capture from results:**
- Relevant snippets for the answer
- `rel_path` for source citations
- **Vocabulary used** in the content (may differ from query terms)
- **Gaps** — aspects of the question not covered

---

### Step 3: REFINE — Targeted Text Search (If Needed)

Use BM25 text search for:
- Exact phrases discovered in Step 2
- Specific names, acronyms, technical terms
- Content in folders discovered in Step 1

```bash
# Search for exact phrase
mneme text-search "exact phrase from results" --config config.toml

# Search within specific folder discovered in Orient
mneme text-search "keyword" --path "105-external-thinkers/" --config config.toml
```

---

### Step 4: CONNECT — Check Hub Documents (For Important Questions)

Find authoritative documents via backlink analysis:

```bash
# Find most-linked documents (hubs are often authoritative)
mneme backlinks --config config.toml --limit 10
```

**When to use:**
- Question involves people, projects, or concepts that might be hubs
- Initial results seem to miss central documents
- Topic likely has a "main" document others reference

---

### Step 5: SYNTHESIZE — Formulate Response with Sources

Combine information from all steps into a clear answer.

**Rules:**
- Cite ALL sources that contributed (not just semantic search results)
- Note when information came from different search methods
- Flag if Orient revealed content you haven't fully explored
- Distinguish between direct quotes and synthesis

---

## MANDATORY: Cite Sources

**Every answer MUST end with a Sources section:**

```markdown
## Sources
- `path/to/file.md` (Section: Heading Name)
- `folder/document.pdf` (Page: 5)
- `presentations/deck.pptx` (Slide: 12)
- `images/diagram.png` (Image analysis)
- `youtube/channel/video.md` (Transcript) ← Don't miss these!
```

**Rules:**
- List ALL files that contributed to the answer
- Include specific location (heading, page, slide number)
- Note content type (transcript, image analysis, etc.)
- Use the `rel_path` from search results

---

## Complete Example

**User asks:** "What is OpenClaw's impact on the AI landscape?"

### Step 1: Orient
```bash
mneme list-docs --config config.toml | grep -i "claw\|agent" | head -20
```
**Discovers:** Files in `obsidian/`, `substack/`, AND `youtube/` folders!

### Step 2: Search
```bash
mneme query --config config.toml "OpenClaw impact AI landscape" --k 15
```
**Returns:** Positive articles about 150K agents, autonomous economies

### Step 3: Refine
```bash
mneme text-search "OpenClaw" --path "105-external-thinkers/youtube/" --config config.toml
```
**Discovers:** Security breach article that semantic search missed!

### Step 4: Connect
```bash
mneme backlinks --config config.toml --limit 10
```
**Finds:** Hub documents with many references

### Step 5: Synthesize
Combine positive AND negative perspectives, cite all sources.

---

## When to Iterate

Run additional cycles when:
- Orient revealed folders you haven't searched yet
- Results use vocabulary different from your query
- Important aspects of the question remain unanswered
- You found only one perspective (positive OR negative)

**Stop when:**
- All aspects of the question are addressed with citations
- You've searched content from all relevant folders
- Last iteration returned no new relevant information

---

## Quick Reference: The 5-Step Checklist

Before finalizing any vault answer, verify:

- [ ] **Oriented** — Ran `list-docs` to see what's available
- [ ] **Searched** — Ran semantic `query` for main results
- [ ] **Refined** — Used `text-search` for exact terms/names if needed
- [ ] **Connected** — Checked `backlinks` for hub documents if relevant
- [ ] **Cited** — Listed ALL sources in Sources section
