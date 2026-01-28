# Common Commands

**Note:** All commands use `./bin/mneme` which auto-handles venv setup.

## Pre-flight Checks

**IMPORTANT: Always use `./bin/mneme` instead of `mneme` directly.**

The `./bin/mneme` wrapper automatically:
- Creates virtual environment if missing (Python 3.11+)
- Installs dependencies if needed
- Activates venv and runs the command

```bash
# These just work - no manual venv setup needed
./bin/mneme scan --config config.toml --full
./bin/mneme watch --config config.toml
./bin/mneme query "search term" --k 10
```

**Manual setup (only if wrapper fails):**
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
# Generate starter config (auto-creates venv if needed)
./bin/mneme init --vault "/path/to/vault" --index "~/.mneme/indexes/myvault"

# Edit config.toml to configure image analysis and embeddings
```

## Scanning and Indexing

```bash
# Full scan (re-index everything) - uses parallel processing by default
./bin/mneme scan --config config.toml --full

# Full scan with explicit parallel settings
./bin/mneme scan --config config.toml --full --workers 8

# Sequential scan (disable parallelization)
./bin/mneme scan --config config.toml --full --no-parallel

# Incremental scan (only changed files)
./bin/mneme scan --config config.toml

# Watch mode (continuous indexing)
./bin/mneme watch --config config.toml
```

## Scanning with Logging and Profiling

```bash
# Scan with logging to file
./bin/mneme scan --config config.toml --full --log-file logs/scan.log

# Scan with verbose (DEBUG) logging
./bin/mneme scan --config config.toml --full --log-file logs/scan.log --verbose

# Scan with profiling (performance analysis)
./bin/mneme scan --config config.toml --full --profile logs/profile.txt

# Scan with both logging and profiling
./bin/mneme scan --config config.toml --full \
    --log-file logs/scan.log \
    --profile logs/profile.txt \
    --verbose

# Watch mode with logging
./bin/mneme watch --config config.toml --log-file logs/watch.log --verbose
```

## Querying

```bash
# Basic search (hybrid: semantic + lexical)
./bin/mneme query --config config.toml "search term" --k 10

# More results
./bin/mneme query --config config.toml "kubernetes deployment" --k 20
```

## MCP Server (Claude Desktop Integration)

```bash
# Start MCP server
./bin/mneme mcp --config config.toml

# Add to Claude Desktop config.json (use absolute path to wrapper):
{
  "mcpServers": {
    "mneme": {
      "command": "/full/path/to/RAGtriever/bin/mneme",
      "args": ["mcp", "--config", "/full/path/to/config.toml"]
    }
  }
}
```

## Quick Workflows

### Setup New Vault (Tesseract)
```bash
./bin/mneme init --vault ~/vault --index ~/.mneme/indexes/myvault
# Edit config.toml: set provider = "tesseract"
./bin/mneme scan --config config.toml --full
./bin/mneme query --config config.toml "test query" --k 5
```

### Setup with Gemini service account
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"
unset GEMINI_API_KEY
./bin/mneme init --vault ~/vault --index ~/.mneme/indexes/myvault
# Edit config.toml: configure [gemini_service_account] section
./bin/mneme scan --config config.toml --full
./bin/mneme query --config config.toml "image content" --k 5
```

### Incremental Updates
```bash
./bin/mneme scan --config config.toml  # Only changed files
```

### Watch Mode (Background)
```bash
# Preferred: Use management script
./scripts/manage_watcher.sh start

# Or manually:
nohup ./bin/mneme watch --config config.toml &
echo $! > logs/watcher.pid

# Check if running
pgrep -f "mneme watch" && echo "Watcher running"

# Stop watcher later
pkill -f "mneme watch"
```

## Verifying Successful Indexing

### Scan Mode
**Console output:**
```
Scan complete: 135 files, 963 chunks in 133.0s
```

**Log file (if logging enabled):**
```bash
# View summary
grep '\[scan\] Complete:' logs/scan_20260123.log

# Check phases
grep '\[scan\] Phase' logs/scan_20260123.log
```

### Watch Mode
**Individual file indexing (INFO level):**
```bash
# List all indexed files
grep '\[watch\] Indexed:' logs/watch_20260123.log
```

**Real-time monitoring:**
```bash
# Terminal 1: Run watcher
mneme watch

# Terminal 2: Monitor indexing
tail -f logs/watch_20260123.log | grep '\[watch\] Indexed:'
```

### Check for Failures
```bash
# Scan or watch failures
grep -E 'Failed:|ERROR' logs/*.log

# Count indexed files
grep -c '\[watch\] Indexed:' logs/watch_20260123.log
```

### Verify in Database
```bash
mneme status
# Output: Indexed files: 135, Indexed chunks: 963
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
