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

## Workflow Routing

| Workflow | Trigger | File |
|----------|---------|------|
| **SearchVault** | "what does the vault say about", "find in vault", answering content questions | `Workflows/SearchVault.md` |
| **AgenticSearch** | "deep search", "thorough search", multi-hop or comparative questions, incomplete initial results | `Workflows/AgenticSearch.md` |
| **SetupVault** | "setup mneme", "initialize vault", "create index" | `Workflows/SetupVault.md` |
| **ConfigureImageAnalysis** | "configure images", "setup gemini", "image analysis" | `Workflows/ConfigureImageAnalysis.md` |
| **ManageWatcher** | "start watcher", "stop watcher", "watcher status" | `Workflows/ManageWatcher.md` |
| **Scan** | "run scan", "full scan", "incremental scan" | `Workflows/Scan.md` |
| **Troubleshoot** | "error", "not working", "fix", troubleshooting | `Workflows/Troubleshoot.md` |

## Examples

**Example 1: Answer a vault question**
```
User: "What types of agentic workflows exist?"
-> Invokes SearchVault workflow
-> Runs: mneme query "agentic workflows" --k 15
-> Returns answer with Sources section citing file paths
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

**Example 4: Deep multi-hop search**
```
User: "How do agentic workflows relate to RAG in my notes?"
-> Invokes AgenticSearch workflow
-> Runs: list-docs, query, text-search, backlinks iteratively
-> Follows wikilink graph to discover connected documents
-> Returns synthesized answer with sources from multiple documents
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
| Architecture & Sprint 1/2 features | `docs/ARCHITECTURE.md` |
| Search strategy & vocabulary mismatch | `docs/SearchBestPractices.md` |
| Config checklist & image providers | `docs/Configuration.md` |
| Common commands reference | `docs/Commands.md` |
| Dev workflow & testing | `docs/DevelopmentWorkflow.md` |
| Issue/solution pairs | `Workflows/Troubleshoot.md` |
| Watcher operations | `Workflows/ManageWatcher.md` |

## Tips for Claude Code Users

1. **ALWAYS search vault content first** - Use `mneme query` when user asks questions
2. **ALWAYS cite sources** - Every response MUST end with a "Sources" section
3. **Vault content != Mneme code** - Don't confuse vault search with codebase search
4. **Use --help to discover options** - Run `mneme <command> --help` instead of guessing
5. **Use the portable wrapper** - `Tools/mneme-wrapper.sh` handles installation automatically
6. **Use AgenticSearch for complex questions** - When simple query doesn't fully answer, iterate with list-docs, text-search, and backlinks
