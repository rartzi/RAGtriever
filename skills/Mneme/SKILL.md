---
name: Mneme
description: Answer questions from indexed vault content using Mneme search. USE WHEN user asks questions that should be answered from vault content, OR needs to setup/configure Mneme, OR troubleshoot scanning/indexing issues, OR manage the watcher service.
---

# Mneme

Claude Code skill for working with Mneme (pronounced NEE-mee) — a local-first vault indexer with hybrid retrieval (semantic + lexical + link-graph).

**Primary Purpose:** Search and answer questions from the indexed vault content, not the Mneme codebase.

## Workflow Routing

| Workflow | Trigger | File |
|----------|---------|------|
| **SearchVault** | "what does the vault say about", "find in vault", answering content questions | `Workflows/SearchVault.md` |
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
-> Runs: ./bin/mneme query "agentic workflows" --k 15
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
-> Checks process status, reports health
-> Returns status and recent activity
```

## Quick Reference

- **Wrapper:** Always use `./bin/mneme` (auto-handles venv)
- **Config:** `config.toml` in project root
- **Search:** `./bin/mneme query "term" --k 10`
- **Scan:** `./bin/mneme scan --config config.toml --full --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log`
- **Watch:** `./scripts/manage_watcher.sh status|start|stop` (logs to `logs/watch_YYYYMMDD.log`)

**IMPORTANT:** All scan and watch operations MUST include logging for audit purposes.

## Detailed Documentation

| Topic | File |
|-------|------|
| Search strategy & vocabulary mismatch | `SearchBestPractices.md` |
| Config checklist & image providers | `Configuration.md` |
| Common commands reference | `Commands.md` |
| Issue/solution pairs | `Troubleshooting.md` |
| Watcher operations | `WatcherManagement.md` |
| Dev workflow & testing | `DevelopmentWorkflow.md` |
| Data flow & execution modes | `Architecture.md` |

## Tips for Claude Code Users

1. **ALWAYS search vault content first** - Use `mneme query` when user asks questions
2. **ALWAYS cite sources** - Every response MUST end with a "Sources" section
3. **Vault content != Mneme code** - Don't confuse vault search with codebase search
4. **Use --help to discover options** - Run `mneme <command> --help` instead of guessing
5. **Use `./bin/mneme` wrapper** - Auto-handles venv setup
