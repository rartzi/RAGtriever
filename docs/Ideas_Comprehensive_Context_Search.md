# Comprehensive Context Search Enhancement

## Problem Statement

When users ask "who/what/where context" questions (e.g., "Who is Marta Milo?", "In what context is B7H4 mentioned?"), the AI agent tends to:
- Report only the highest-scoring result
- Miss other relevant contexts
- Prioritize detail over completeness

**Example:** When asked about Marta Milo, initially reported 1 project but she actually appears in 2 projects.

## Proposed Solutions

### Option 1: Create a Skill (Immediate)

Create `comprehensive-context-search` skill that triggers on:
- "Who is [person]?"
- "What projects is [person] involved in?"
- "In what context is [topic] mentioned?"
- "Where does [term] appear?"

**Required workflow:**
1. Search broadly (k=15-20)
2. Group by source document
3. Present ALL contexts first
4. Then provide details

**Response structure:**
```
[Entity] appears in X projects across Y presentations:

Presentations:
1. [Document A]
2. [Document B]

Projects:
1. [Project 1] - [brief description]
2. [Project 2] - [brief description]

[Ask if user wants details on any specific context]
```

### Option 2: Code Enhancement - Result Grouping

Add CLI functionality:

```bash
ragtriever query "Marta Milo" --group-by document --k 20

# Output:
# Found in 2 documents:
# 1. Innovation Session Cambridge 5 Aug.pptx (3 mentions)
#    - Project: AI-Powered Virtual Lab
# 2. Innovation Session Cambridge 9 June.pptx (2 mentions)
#    - Project: Single-cell PBMC AI estimator
```

Or add specialized commands:
```bash
ragtriever query-person "Marta Milo"    # Auto-groups by project
ragtriever query-context "B7H4"         # Shows all contexts
```

### Option 3: Enhanced Indexing Metadata

Extract structured metadata during indexing:

```python
chunk_metadata = {
    "project_name": "AI-Powered Virtual Lab",  # From slide title
    "people": ["Ricardo Miragaia", "Marta Milo", "Pablo Moreno"],
    "presentation_context": "Innovation Session Cambridge Aug 5"
}

# Enable queries like:
ragtriever query --filter "person:Marta Milo" --group-by project
```

## Implementation Priority

1. **Immediate:** Create comprehensive-context-search skill
2. **Short-term:** Add `--group-by document` CLI flag
3. **Medium-term:** Enhanced metadata extraction during indexing

## Related Issues

- AI agent behavior: tendency to cherry-pick "best" result over comprehensiveness
- RAG result presentation: no built-in document grouping
- Metadata extraction: currently minimal structured data from presentations

## Session Context

- Date: 2026-01-23
- Discovered during: Marta Milo and B7H4 context queries
- Files indexed: 15 PowerPoint presentations in `101-ppt/`
