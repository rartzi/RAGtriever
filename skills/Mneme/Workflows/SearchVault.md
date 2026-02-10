# SearchVault Workflow

Search and answer questions from indexed vault content.

## Trigger Phrases
- "what does the vault say about..."
- "find in vault..."
- "search for..."
- Any question that should be answered from vault content

## Choose Your Approach

**Match search depth to question complexity:**

| Question Type | Approach | Steps |
|---------------|----------|-------|
| Simple lookup ("find X", "what is Y's role") | Quick semantic | Search → Cite |
| Comprehensive ("insights about", "impact of") | Full agentic | Orient → Search → Refine → Connect → Cite |
| Initially incomplete results | Expand | Add Orient + Refine to find what's missing |

## Understanding the Tradeoffs

### Quick Semantic Search
- ✅ **Fast** (~2 seconds)
- ✅ Good for known topics, specific lookups
- ❌ Misses content with different vocabulary
- ❌ Misses content in unexpected folders
- ❌ May return one-sided perspective

### Full Agentic Search
- ✅ Discovers content in unexpected places
- ✅ Finds counter-narratives and multiple perspectives
- ✅ Uses document vocabulary, not just query terms
- ❌ Takes longer (~30 seconds)
- ❌ Overkill for simple lookups

### The OpenClaw Lesson

Semantic search for "OpenClaw impact AI landscape" returned only positive articles. But `list-docs | grep -i claw` revealed a YouTube folder containing **critical security warnings** the semantic search completely missed.

**Takeaway:** For comprehensive questions, Orient first. For simple lookups, semantic is fine.

---

## Procedure

### Step 1: ORIENT — Understand What's Available

**When to use:** Unfamiliar topics, comprehensive questions, or when you need full coverage.
**Skip when:** Simple lookups for known content.

```bash
# Quick orientation - see what folders/content types exist
mneme list-docs --config config.toml | head -50

# If topic-specific, grep for relevant paths
mneme list-docs --config config.toml | grep -i "keyword" | head -30
```

**Why it helps:** Semantic search only returns what matches the query embedding. Orient reveals content that exists but uses different vocabulary or lives in unexpected locations (YouTube transcripts, images, subfolders).

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

## Quick Reference: Decision Checklist

Before finalizing any vault answer:

**For simple lookups:**
- [ ] Ran semantic `query` for results
- [ ] Results directly answer the question
- [ ] Cited sources

**For comprehensive questions (expand if initial results seem incomplete):**
- [ ] **Oriented** — Ran `list-docs | grep keyword` to see what content types exist
- [ ] **Searched** — Ran semantic `query` for main results
- [ ] **Refined** — Used `text-search` in folders discovered during Orient
- [ ] **Connected** — Checked `backlinks` for hub documents if relevant
- [ ] **Balanced** — Results include multiple perspectives (if applicable)
- [ ] **Cited** — Listed ALL sources in Sources section

**Signs you need to expand to agentic:**
- Results seem one-sided (all positive or all negative)
- Results all come from one folder/content type
- Important aspects of the question aren't addressed
- You know content exists that wasn't returned
