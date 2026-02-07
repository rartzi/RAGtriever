# Common Commands

## How to Run mneme

The skill provides multiple ways to run mneme:

```bash
# 1. Skill wrapper (portable, auto-installs to ~/.mneme/)
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh <command>

# 2. Project-local wrapper (in RAGtriever directory)
./bin/mneme <command>

# 3. Global (if pip installed)
mneme <command>
```

**The examples below use `mneme` - replace with the appropriate path for your setup.**

## Pre-flight Checks

### Install mneme (if not installed)

```bash
# Auto-install to ~/.mneme/
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --install

# Check where mneme is installed
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --where

# Update existing installation
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --update
```

### Manual setup (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
mneme scan --config config.toml
```

## Discovering CLI Options

**IMPORTANT: Always check CLI help for current options instead of guessing:**

```bash
# Main help - shows all commands
mneme --help

# Command-specific help
mneme scan --help
mneme watch --help
mneme query --help
mneme init --help
mneme mcp --help
mneme status --help
```

**Example workflow:**
```bash
# User asks: "How do I enable logging for scan?"
# Instead of guessing, check:
mneme scan --help | grep -A 2 log

# Output shows:
#   --log-file TEXT    Log file path for audit trail
#   --log-level TEXT   Log level: DEBUG, INFO, WARNING, ERROR
#   --verbose, -v      Enable verbose (DEBUG) logging
```

## Initial Setup

```bash
# Generate starter config
mneme init --vault "/path/to/vault" --index "~/.mneme/indexes/myvault"

# Edit config.toml to configure image analysis and embeddings
```

## Scanning and Indexing

**IMPORTANT: Always include logging for audit purposes.**

```bash
# Full scan with logging (REQUIRED)
mkdir -p logs
mneme scan --config config.toml --full --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log

# Full scan with explicit parallel settings
mneme scan --config config.toml --full --workers 8 --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log

# Sequential scan (disable parallelization)
mneme scan --config config.toml --full --no-parallel --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log

# Incremental scan (only changed files)
mneme scan --config config.toml --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log
```

## Scanning with Verbose Logging and Profiling

```bash
# Scan with verbose (DEBUG) logging
mneme scan --config config.toml --full --log-file logs/scan.log --verbose

# Scan with profiling (performance analysis)
mneme scan --config config.toml --full --profile logs/profile.txt --log-file logs/scan.log

# Scan with both logging and profiling
mneme scan --config config.toml --full \
    --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log \
    --profile logs/profile_$(date +%Y%m%d_%H%M%S).txt \
    --verbose
```

## Querying

When the watcher is running, queries automatically route through its built-in query server (~0.1s vs ~5s cold-start).

```bash
# Basic search (hybrid: semantic + lexical, auto-routes via watcher socket)
mneme query "search term" --k 10

# More results
mneme query "kubernetes deployment" --k 20

# With cross-encoder reranking (+20-30% quality, +100-200ms)
mneme query "machine learning" --rerank

# Force cold-start (bypass watcher query server)
mneme query "search term" --no-socket

# With config file
mneme query --config config.toml "search term" --k 10

# JSON output
mneme query "project status" --json
```

## Watcher Management

The watcher monitors vault changes and includes a **built-in query server** that provides ~0.1s query latency (vs ~5s cold-start). Use the portable management script:

```bash
# Check status
~/.claude/skills/Mneme/Tools/manage-watcher.sh status

# Start watcher + query server (auto-installs mneme, creates logs)
~/.claude/skills/Mneme/Tools/manage-watcher.sh start

# Stop watcher
~/.claude/skills/Mneme/Tools/manage-watcher.sh stop

# Restart watcher
~/.claude/skills/Mneme/Tools/manage-watcher.sh restart

# Health check (verifies watcher + query server socket)
~/.claude/skills/Mneme/Tools/manage-watcher.sh health
```

**Manual watcher management:**

```bash
# Start in background with logging
mkdir -p logs
nohup mneme watch --config config.toml --log-file logs/watch_$(date +%Y%m%d).log &
echo $! > logs/watcher.pid

# Check if running
pgrep -f "mneme watch" && echo "Watcher running"

# Stop watcher
pkill -f "mneme watch"
```

## MCP Server (Claude Desktop Integration)

```bash
# Start MCP server
mneme mcp --config config.toml

# Add to Claude Desktop config.json:
{
  "mcpServers": {
    "mneme": {
      "command": "~/.mneme/venv/bin/mneme",
      "args": ["mcp", "--config", "/full/path/to/config.toml"]
    }
  }
}
```

## Quick Workflows

### Setup New Vault (Tesseract)
```bash
mneme init --vault ~/vault --index ~/.mneme/indexes/myvault
# Edit config.toml: set provider = "tesseract"
mkdir -p logs
mneme scan --config config.toml --full --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log
mneme query "test query" --k 5
```

### Setup with Gemini Service Account
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"
unset GEMINI_API_KEY
mneme init --vault ~/vault --index ~/.mneme/indexes/myvault
# Edit config.toml: configure [gemini_service_account] section
mkdir -p logs
mneme scan --config config.toml --full --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log
mneme query "image content" --k 5
```

### Incremental Updates
```bash
mneme scan --config config.toml --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log
```

## Verifying Successful Indexing

### Scan Mode
**Console output:**
```
Scan complete: 149 files, 3106 chunks in 182.3s
  (7 images analyzed, 149 files skipped unchanged)
```

**Log file:**
```bash
# View summary
grep '\[scan\] Complete:' logs/scan_*.log

# Check phases
grep '\[scan\] Phase' logs/scan_*.log
```

### Watch Mode
**Individual file indexing (INFO level):**
```bash
# List all indexed files
grep '\[watch\] Indexed:' logs/watch_*.log
```

**Real-time monitoring:**
```bash
tail -f logs/watch_$(date +%Y%m%d).log | grep '\[watch\]'
```

### Check for Failures
```bash
# Scan or watch failures
grep -E 'Failed:|ERROR' logs/*.log

# Count indexed files
grep -c '\[watch\] Indexed:' logs/watch_*.log
```

### Check Status
```bash
mneme status
# Output: Indexed files: 149, Indexed chunks: 3106
```

## Database Queries

```bash
# Count documents
sqlite3 ~/.mneme/indexes/myvault/vaultrag.sqlite \
    "SELECT COUNT(*) FROM documents WHERE deleted=0;"

# Count chunks
sqlite3 ~/.mneme/indexes/myvault/vaultrag.sqlite \
    "SELECT COUNT(*) FROM chunks;"

# Show file types
sqlite3 ~/.mneme/indexes/myvault/vaultrag.sqlite \
    "SELECT file_type, COUNT(*) FROM documents WHERE deleted=0 GROUP BY file_type;"
```
