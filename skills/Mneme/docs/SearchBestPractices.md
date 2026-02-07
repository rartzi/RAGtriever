# Search Best Practices

## CRITICAL: Answering Questions from Vault Content

**When the user asks ANY question that could be answered from vault content:**

1. **ALWAYS use `mneme query` to search the indexed vault**
2. **NEVER answer from memory or assumptions about the codebase**
3. **The vault content is the source of truth** - not the Mneme repository code itself
4. **ALWAYS cite sources in your response** - Include the file paths and locations

## MANDATORY: Citing Sources in Responses

**Every answer based on vault content MUST include a "Sources" section at the end:**

```
## Sources
- `path/to/file.md` (Section: Heading Name)
- `folder/document.pdf` (Page: 5)
- `presentations/deck.pptx` (Slide: 12)
- `images/diagram.png` (Image analysis)
```

**Source citation rules:**
- List ALL files that contributed to the answer
- Include the specific location within the file (heading, page, slide number)
- For images, note that it came from image analysis
- Use the `rel_path` from search results
- If information comes from multiple chunks in the same file, list it once with all relevant sections

## Vocabulary Mismatch is the #1 Issue

Your query words often differ from document words:
- "team" in query -> "ownership" in document
- "responsibilities" -> "scope" or "role"
- "people" -> "sponsor", "lead", "manager"

**The embedding model doesn't always connect synonyms.**

## Effective Search Strategy

1. **Iterate, don't rely on one query**
   - First results inform your next query
   - Note the vocabulary used in results, then search using those terms

2. **Increase k for broad questions**
   ```bash
   mneme query "topic" --k 20           # More results than default 10
   mneme query "topic" --k 30 --rerank  # Best coverage
   ```

3. **Try synonyms and domain terms**
   - Generic terms often miss specific content
   - Use role-specific terms: "sponsor", "owner", "lead"
   - Use domain vocabulary from initial results

4. **Narrow after broad**
   - Start: broad concept query
   - Then: specific names/terms discovered in initial results

5. **When something seems missing, search directly**
   - Search for specific names, acronyms, or exact phrases
   - Example: If "team" misses someone, search their name directly

## Example: Finding Project Team Members

```bash
# First attempt (broad)
mneme query "Navari project team" --k 15

# Results use "sponsor", "ownership" - not "team"
# Second attempt (using document vocabulary)
mneme query "Navari sponsor ownership comms" --k 15

# Still missing someone? Search directly
mneme query "Anne-Claire Navari" --k 5
```

## When to Ask for Clarification

**Don't guess with endless query variations. Ask the user.**

### Ask When:

- **Query is broad/ambiguous**
  - User: "Navari team"
  - Ask: "Are you looking for specific roles (sponsors, leads) or the full org structure?"

- **Initial results seem incomplete**
  - "I found Justin and Jorge, but results may not have everyone. Are there specific people or roles you expected to see?"

- **Domain terms are unclear**
  - "What do you mean by 'responsibilities' - reporting structure, project scope, or job duties?"

- **Results don't match apparent intent**
  - "These results focus on project timeline. Were you looking for something else?"

### How to Ask

Keep it brief - one clarifying question, then search:
- "Are you looking for [specific thing] or [other thing]?"
- "I found X and Y. Anyone else you expected?"
- "Should I search for specific names or roles?"

**Don't over-ask** - gather context, then execute.

## Question Types to Search For

- "What ideas do we have?" -> Search vault for ideas
- "List all projects" -> Search vault for projects
- "Who is working on X?" -> Search vault for people/assignments
- "What's the status of Y?" -> Search vault for status updates
- "Find information about Z" -> Search vault content
