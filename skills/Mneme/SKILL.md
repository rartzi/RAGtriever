---
name: Mneme
description: Answer questions from indexed vault content using Mneme search. USE WHEN user asks questions that should be answered from vault content, OR needs to setup/configure Mneme, OR troubleshoot scanning/indexing issues, OR manage the watcher service.
---

# Mneme

Claude Code skill for working with Mneme (pronounced NEE-mee) — a local-first vault indexer with hybrid retrieval (semantic + lexical + link-graph).

**Primary Purpose:** Search and answer questions from the indexed vault content, not the Mneme codebase.

## Skill Structure

This skill is **self-contained and portable**. It can be deployed to any project.

```
skills/Mneme/
├── SKILL.md              # This file (routing + quick reference)
├── DEPLOYMENT.md         # Deployment guide
├── Tools/                # Portable scripts
│   ├── mneme-wrapper.sh  # Auto-installing CLI wrapper
│   └── manage-watcher.sh # Watcher management
├── Workflows/            # Execution procedures (7 workflows)
├── docs/                 # Reference documentation
│   ├── ARCHITECTURE.md   # System architecture + Sprint 1/2 features
│   ├── Commands.md       # CLI command reference
│   ├── Configuration.md  # Config options + image providers
│   ├── SearchBestPractices.md  # Search strategy tips
│   ├── DevelopmentWorkflow.md  # Dev setup + testing
│   ├── USERGUIDE.md      # End-user guide
│   └── PRD.md            # Product requirements
├── examples/             # Example config files
└── source/               # Bundled source code
```

## Search Strategy: Know the Tradeoffs

**Two approaches, each with tradeoffs:**

| Approach | Speed | Coverage | Best For |
|----------|-------|----------|----------|
| **Quick Semantic** | Fast (~2s) | Matches query embedding only | Simple lookups, known topics |
| **Agentic Search** | Slower (~30s) | Full vault exploration | Complex questions, unfamiliar topics |

### Quick Semantic Search
```bash
mneme query "specific known term" --k 10
```
**Pros:** Fast, efficient for targeted lookups
**Cons:** Misses content with different vocabulary, unexpected locations, counter-narratives

**Use when:** Looking up a specific person, project, or well-defined term you know exists

### Agentic Search (Orient → Search → Refine → Connect)
```bash
mneme list-docs | grep -i "topic"     # Orient: what exists?
mneme query "topic" --k 15            # Search: semantic matches
mneme text-search "exact term"        # Refine: discovered vocabulary
mneme backlinks --limit 10            # Connect: hub documents
```
**Pros:** Discovers content in unexpected places, multiple perspectives, counter-narratives
**Cons:** Takes longer, may be overkill for simple questions

**Use when:**
- Exploring unfamiliar topics
- Questions asking for "insights" or "impact"
- Need comprehensive/balanced view
- Initial results seem incomplete or one-sided

### The OpenClaw Lesson

Semantic search for "OpenClaw impact AI landscape" returned only positive articles about 150K agents and autonomous economies. But `list-docs | grep -i claw` revealed a YouTube folder containing critical security warnings the semantic search completely missed.

**Takeaway:** For questions requiring comprehensive coverage, Orient first. For simple lookups, semantic search is fine.

## Workflow Routing

| Workflow | Trigger | File |
|----------|---------|------|
| **SearchVault** | ANY vault content question — always uses full agentic process | `Workflows/SearchVault.md` |
| **AgenticSearch** | Reference doc for the 6-step deep search pattern | `Workflows/AgenticSearch.md` |
| **SetupVault** | "setup mneme", "initialize vault", "create index" | `Workflows/SetupVault.md` |
| **ConfigureImageAnalysis** | "configure images", "setup gemini", "image analysis" | `Workflows/ConfigureImageAnalysis.md` |
| **ManageWatcher** | "start watcher", "stop watcher", "watcher status" | `Workflows/ManageWatcher.md` |
| **Scan** | "run scan", "full scan", "incremental scan" | `Workflows/Scan.md` |
| **Troubleshoot** | "error", "not working", "fix", troubleshooting | `Workflows/Troubleshoot.md` |

## Examples

**Example 1: Answer a vault question (CORRECT - uses agentic process)**
```
User: "What is OpenClaw's impact on AI?"
-> Invokes SearchVault workflow with FULL agentic process:

Step 1 - ORIENT:
   mneme list-docs --config config.toml | grep -i "claw\|openclaw"
   -> Discovers: files in obsidian/, substack/, AND youtube/ folders!

Step 2 - SEARCH:
   mneme query "OpenClaw impact AI" --k 15
   -> Returns positive articles about agent economies

Step 3 - REFINE:
   mneme text-search "OpenClaw" --path "youtube/" --config config.toml
   -> Discovers CRITICAL security article semantic search missed!

Step 4 - CONNECT:
   mneme backlinks --config config.toml --limit 10
   -> Finds hub documents

Step 5 - SYNTHESIZE:
   -> Combines BOTH positive and negative perspectives
   -> Cites ALL sources including the YouTube transcript

-> Returns balanced answer with complete Sources section
```

**Example 2: Setup new vault**
```
User: "Setup Mneme for my vault at ~/Documents/notes"
-> Invokes SetupVault workflow
-> Runs init, configures settings, runs initial scan
-> Returns confirmation with status
```

**Example 3: Manage watcher**
```
User: "Is the watcher running?"
-> Invokes ManageWatcher workflow
-> Uses Tools/manage-watcher.sh to check status
-> Returns status and recent activity
```

**Example 4: WRONG approach (don't do this)**
```
User: "What insights exist about topic X?"
-> WRONG: Just runs mneme query "topic X" --k 15
-> WRONG: Returns answer from only semantic matches
-> WRONG: Misses content in unexpected folders
-> WRONG: Provides incomplete or biased view

ALWAYS use the 5-step process even for "simple" questions!
```

## Quick Reference

### Running mneme Commands

The skill provides a portable wrapper that auto-installs mneme:

```bash
# Via skill wrapper (portable, auto-installs)
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh <command>

# Via project-local wrapper (in RAGtriever directory)
./bin/mneme <command>

# Via global install (if pip installed)
mneme <command>
```

### Common Commands

```bash
# Search (auto-routes through watcher if running for ~0.1s response)
mneme query "search term" --k 10

# Search (force cold-start, skip watcher socket)
mneme query "search term" --k 10 --no-socket

# Full scan with logging (REQUIRED)
mkdir -p logs
mneme scan --config config.toml --full --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log

# Status
mneme status

# Watcher management (also starts query server for fast CLI queries)
~/.claude/skills/Mneme/Tools/manage-watcher.sh status|start|stop|health
```

### Agentic Search Commands

```bash
# List indexed documents (orient phase)
mneme list-docs --config config.toml
mneme list-docs --path "projects/" --config config.toml

# BM25 text search (exact phrase matching, bypasses semantic search)
mneme text-search "exact phrase" --config config.toml
mneme text-search "term" --path "notes/" --k 20 --config config.toml

# Backlink analysis (find hub documents)
mneme backlinks --config config.toml --limit 10
mneme backlinks --paths "projects/alpha.md" --config config.toml
```

**IMPORTANT:** All scan and watch operations MUST include logging for audit purposes.

### Query Performance

When the watcher is running, `mneme query` automatically routes through the watcher's built-in query server via unix socket, skipping Python and model startup (~5s -> ~0.1s). If the watcher is not running, queries fall back to cold-start automatically.

### Installation

```bash
# Install mneme (auto-installs to ~/.mneme/)
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --install

# Update mneme
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --update

# Check where mneme is installed
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --where
```

## Detailed Documentation

| Topic | File |
|-------|------|
| **Deployment guide** | `DEPLOYMENT.md` |
| Architecture & all sprint features | `docs/ARCHITECTURE.md` |
| Search strategy & vocabulary mismatch | `docs/SearchBestPractices.md` |
| Config checklist & image providers | `docs/Configuration.md` |
| Common commands reference | `docs/Commands.md` |
| Dev workflow & testing | `docs/DevelopmentWorkflow.md` |
| Issue/solution pairs | `Workflows/Troubleshoot.md` |
| Watcher operations | `Workflows/ManageWatcher.md` |

## Tips for Claude Code Users

1. **Match search depth to question complexity** - Simple lookups → semantic; comprehensive questions → agentic
2. **Orient when exploring unfamiliar topics** - `list-docs | grep keyword` reveals content semantic search misses
3. **ALWAYS cite sources** - Every response MUST end with a "Sources" section
4. **Vault content != Mneme code** - Don't confuse vault search with codebase search
5. **Use --help to discover options** - Run `mneme <command> --help` instead of guessing
6. **Use the portable wrapper** - `Tools/mneme-wrapper.sh` handles installation automatically
7. **Check multiple perspectives** - If results seem one-sided, Orient to find other content types
8. **Use backlinks for important topics** - Hub documents often contain authoritative information

## Decision Guide: When to Use What

| Question Type | Approach | Example |
|---------------|----------|---------|
| Specific lookup | Quick semantic | "What's Alex Rivera's role?" |
| Known document | Quick semantic | "Find the AWS deck" |
| Comprehensive insights | Full agentic | "What are the impacts of X?" |
| Unfamiliar topic | Full agentic | "What does the vault say about Y?" |
| Comparing/contrasting | Full agentic | "How does A relate to B?" |
| One-sided initial results | Expand to agentic | Add Orient + Refine steps |

**Rule of thumb:** If unsure, start with semantic search. If results seem incomplete, one-sided, or miss obvious content → expand to agentic approach.
