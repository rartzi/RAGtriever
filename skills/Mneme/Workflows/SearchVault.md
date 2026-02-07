# SearchVault Workflow

Search and answer questions from indexed vault content.

## Trigger Phrases
- "what does the vault say about..."
- "find in vault..."
- "search for..."
- Any question that should be answered from vault content

## Procedure

### 1. Run the Query

```bash
mneme query --config config.toml "user's question keywords" --k 15
```

**Performance:** If the watcher is running, queries route through its built-in query server automatically (~0.1s vs ~5s cold-start). No extra flags needed.

**Options:**
- `--k 15` - Return 15 results (increase for broad questions)
- `--k 30 --rerank` - Best coverage for complex questions
- `--no-socket` - Force cold-start (skip watcher query server)

### 2. Analyze Results

- Review returned chunks for relevant information
- Note the `rel_path` field for source citation
- Look for vocabulary that might inform follow-up queries

### 3. Formulate Response

Synthesize information from the search results into a clear answer.

### 4. MANDATORY: Cite Sources

**Every answer MUST end with a Sources section:**

```
## Sources
- `path/to/file.md` (Section: Heading Name)
- `folder/document.pdf` (Page: 5)
- `presentations/deck.pptx` (Slide: 12)
- `images/diagram.png` (Image analysis)
```

**Rules:**
- List ALL files that contributed to the answer
- Include specific location (heading, page, slide number)
- For images, note it came from image analysis
- Use the `rel_path` from search results

## Example

**User asks:** "What types of agentic workflows exist?"

**Run:**
```bash
mneme query --config config.toml "agentic workflows" --k 15
```

**Response:**
```
Based on the vault content, there are two main types of agentic workflows with LLMs:

1. **Reflexion - Learning from Feedback**
   - Uses self-reflection and evaluation
   - Components: Actor, Evaluator, Experience memory

2. **Multi-Agent Systems**
   - Specialized agents working together
   - Examples: Search agent, Reasoning agent, Hypothesis agent

## Sources
- `99-Images/Pasted image 20240917165150.png` (Image: Agentic workflows diagram)
- `99-Images/Pasted image 20240917165410.png` (Image: Amazon Bedrock Agents)
```

## Iterating on Search

If initial results don't fully answer the question:

1. **Note vocabulary** from results that differs from query
2. **Run follow-up query** using discovered terms
3. **Search specific names** if people/entities seem missing

```bash
# First attempt (broad)
mneme query "project team" --k 15

# Results use "sponsor", "ownership" - not "team"
# Second attempt (using document vocabulary)
mneme query "sponsor ownership leads" --k 15
```

## When to Ask for Clarification

- Query is broad/ambiguous
- Initial results seem incomplete
- Domain terms are unclear
- Results don't match apparent intent

Keep it brief - one clarifying question, then search.
