# Mneme User Guide

Comprehensive guide for using Mneme — Memory for your Second Brain.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Basic Setup](#basic-setup)
  - [Embedding Configuration](#embedding-configuration)
  - [Image Analysis Providers](#image-analysis-providers)
  - [Logging Configuration](#logging-configuration)
- [CLI Reference](#cli-reference)
  - [scan](#scan-command)
  - [watch](#watch-command)
  - [query](#query-command)
  - [mcp](#mcp-command)
- [Watcher Management](#watcher-management)
- [MCP Server Integration](#mcp-server-integration)
- [Claude Code Skill](#claude-code-skill)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Create config
mneme init --vault "/path/to/vault" --index "~/.mneme/indexes/myvault"

# 3. Index your vault
mneme scan --full

# 4. Search
mneme query "your search term"

# 5. Watch for changes (optional)
mneme watch
```

---

## Configuration

### Basic Setup

Create a `config.toml` file:

```toml
[vault]
root = "/path/to/your/obsidian/vault"
ignore = [
    ".git/**",
    ".obsidian/cache/**",
    "**/.DS_Store",
    "**/~$*",           # Office temp files
    "**/.~lock.*"       # LibreOffice lock files
]

[index]
dir = "~/.mneme/indexes/myvault"
extractor_version = "v1"
chunker_version = "v2"  # v2: adds chunk overlap

[chunking]
overlap_chars = 200      # Context preservation between chunks
max_chunk_size = 2000

[retrieval]
top_k = 10
use_rerank = false       # Enable for better quality (+100-200ms)
```

### Embedding Configuration

```toml
[embeddings]
provider = "sentence_transformers"
model = "BAAI/bge-small-en-v1.5"
device = "cpu"           # cpu | cuda | mps (Apple Silicon)
batch_size = 32
offline_mode = true      # Use cached models only

# Asymmetric retrieval for BGE models
use_query_prefix = true
query_prefix = "Represent this sentence for searching relevant passages: "

# FAISS for large vaults (>10K chunks)
use_faiss = false
```

**Device options:**
- `cpu` - Works everywhere, slowest
- `cuda` - NVIDIA GPU, fastest
- `mps` - Apple Silicon (M1/M2/M3)

**Offline mode:** Set `offline_mode = true` for corporate environments. Download models first with `offline_mode = false`, then enable.

### Image Analysis Providers

#### Tesseract (Local OCR)
```toml
[image_analysis]
provider = "tesseract"
# Requires: brew install tesseract (macOS) or apt install tesseract-ocr
```

#### Gemini API (Simple)
```toml
[image_analysis]
provider = "gemini"
gemini_model = "gemini-2.0-flash"
# Set GEMINI_API_KEY environment variable
```

#### Gemini with Service Account (Enterprise)
```toml
[image_analysis]
provider = "gemini-service-account"

[gemini_service_account]
project_id = "your-gcp-project-id"
location = "global"                    # or us-central1, europe-west4
credentials_file = "/path/to/service-account.json"
model = "gemini-2.0-flash-exp"
```

**Service Account Setup:**

```bash
# 1. Create service account
PROJECT_ID="your-project-id"
gcloud iam service-accounts create mneme-sa \
    --display-name="Mneme Image Analysis" \
    --project=$PROJECT_ID

# 2. Grant Vertex AI User role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:mneme-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# 3. Create credentials
gcloud iam service-accounts keys create ~/.config/gcloud/mneme-key.json \
    --iam-account=mneme-sa@${PROJECT_ID}.iam.gserviceaccount.com

# 4. Secure file
chmod 600 ~/.config/gcloud/mneme-key.json
```

#### Microsoft AI Gateway
```toml
[image_analysis]
provider = "aigateway"

[aigateway]
url = "https://your-gateway.azure.com"
key = "your-api-key"
model = "gemini-2.5-flash"
```

### Logging Configuration

```toml
[logging]
dir = "logs"
scan_log_file = "logs/scan_{date}.log"      # Daily rotation
watch_log_file = "logs/watch_{datetime}.log" # Per-run logs
level = "INFO"                               # DEBUG | INFO | WARNING | ERROR
enable_scan_logging = false                  # Manual via CLI
enable_watch_logging = true                  # Always log watch mode
```

**Pattern substitution:**
- `{date}` → `20260123` (YYYYMMDD)
- `{datetime}` → `20260123_142030` (YYYYMMDD_HHMMSS)

---

## CLI Reference

### scan Command

Index your vault.

```bash
# Full scan (re-index everything)
mneme scan --full

# Incremental scan (only changed files)
mneme scan

# With parallel workers
mneme scan --full --workers 8

# Sequential (disable parallelization)
mneme scan --full --no-parallel

# With logging
mneme scan --full --log-file logs/scan.log --verbose
```

### watch Command

Continuously monitor vault for changes.

```bash
# Start watcher
mneme watch

# With custom config
mneme watch --config my_config.toml

# With logging
mneme watch --log-file logs/watch.log --verbose

# Batched mode options
mneme watch --batch-size 20 --batch-timeout 10
```

### query Command

Search your indexed vault.

```bash
# Basic search
mneme query "search term"

# More results
mneme query "kubernetes deployment" --k 20

# With reranking (better quality)
mneme query "machine learning" --rerank

# JSON output
mneme query "project status" --json
```

### mcp Command

Start the MCP server for AI agent integration.

```bash
# Start MCP server (stdio transport)
mneme mcp

# With custom config
mneme mcp --config config.toml
```

---

## Watcher Management

### Using the Skill's Management Script

The Mneme skill provides a portable watcher management script:

```bash
# Check status
~/.claude/skills/Mneme/Tools/manage-watcher.sh status

# Start watcher (auto-installs mneme if needed)
~/.claude/skills/Mneme/Tools/manage-watcher.sh start

# Stop watcher
~/.claude/skills/Mneme/Tools/manage-watcher.sh stop

# Restart
~/.claude/skills/Mneme/Tools/manage-watcher.sh restart

# Health check
~/.claude/skills/Mneme/Tools/manage-watcher.sh health

# Check installation
~/.claude/skills/Mneme/Tools/manage-watcher.sh check
```

### Manual Management

```bash
# Check if running
pgrep -f "mneme watch"

# Start in background with logging
mkdir -p logs
nohup mneme watch --config config.toml --log-file logs/watch_$(date +%Y%m%d).log &
echo $! > logs/watcher.pid

# Stop
pkill -f "mneme watch"

# Check logs
tail -f logs/watch_$(date +%Y%m%d).log
```

### What the Watcher Handles

- Individual file changes (create, modify, delete, move)
- Directory operations (delete/move entire folders)
- Batched processing for efficiency
- Automatic index updates

---

## MCP Server Integration

### Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mneme": {
      "command": "/path/to/mneme/.venv/bin/mneme",
      "args": ["mcp", "--config", "/path/to/config.toml"],
      "cwd": "/path/to/mneme"
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `vault_search` | Hybrid search across your vault |
| `vault_open` | Retrieve full content of a file |
| `vault_neighbors` | Get linked notes (outlinks/backlinks) |
| `vault_status` | Index statistics |
| `vault_list` | List configured vaults |

### Example Queries

After setup, ask Claude:
- "Search my vault for notes about machine learning"
- "Find all meeting notes from last week"
- "What are the backlinks to my Project X note?"

---

## Claude Code Skill

### Installation

```bash
# Symlink from repository
ln -s $(pwd)/skills/Mneme ~/.claude/skills/Mneme
```

### What It Provides

The skill follows the PAI (Personal AI Infrastructure) standard with workflow-based routing:

- **SearchVault** - Search and answer questions from vault content with source citations
- **SetupVault** - Initial vault setup and configuration
- **ConfigureImageAnalysis** - Configure image analysis providers (Tesseract/Gemini)
- **ManageWatcher** - Start/stop/status for the watcher service
- **Scan** - Run full or incremental scans
- **Troubleshoot** - Diagnose and fix common issues

Plus context files for detailed documentation on search best practices, configuration, commands, and architecture.

---

## Troubleshooting

### Office Temp Files

**Symptom:** `PackageNotFoundError: Package not found at '~$document.pptx'`

**Solution:**
1. Close all Office applications before scanning
2. Add ignore patterns to config.toml:
   ```toml
   ignore = ["**/~$*", "**/.~lock.*"]
   ```
3. Remove existing temp files:
   ```bash
   find /path/to/vault -name "~$*" -delete
   ```

### Offline Mode Issues

**Symptom:** `OSError: We couldn't connect to 'https://huggingface.co'`

**Solution:**
```bash
# Download model first
export HF_OFFLINE_MODE=0
mneme scan --config config.toml

# Then enable offline mode
export HF_OFFLINE_MODE=1
```

### Gemini Authentication

**Symptom:** `Credentials file not found`

**Solution:**
```bash
# Check file exists
ls -l $GOOGLE_APPLICATION_CREDENTIALS

# Use absolute path in config
credentials_file = "/absolute/path/to/service-account.json"
```

**Symptom:** `PermissionDenied`

**Solution:**
```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:mneme-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### Rate Limit Errors (429)

**Expected behavior:** This is normal with many images. The scan continues and most images are processed successfully.

**Solution:**
- Re-run scan (successful images are cached)
- Try `location = "global"` for better routing
- Consider Tesseract for large image sets

### No Query Results

**Check:**
```bash
# Verify index has documents
sqlite3 ~/.mneme/indexes/myvault/vaultrag.sqlite \
    "SELECT COUNT(*) FROM documents WHERE deleted=0;"

# Verify embeddings exist
sqlite3 ~/.mneme/indexes/myvault/vaultrag.sqlite \
    "SELECT COUNT(*) FROM embeddings;"
```

### Slow Indexing

**Solutions:**
1. Use GPU: `device = "mps"` (Mac) or `device = "cuda"` (NVIDIA)
2. Increase batch size: `batch_size = 64`
3. Use parallel scanning: `--workers 8`
4. Use Tesseract instead of Gemini for images

### High Memory Usage

**Solutions:**
1. Reduce batch size: `batch_size = 16`
2. Use smaller model: `model = "sentence-transformers/all-MiniLM-L6-v2"`

---

## Verifying Indexing

### Scan Output

```
Scan complete: 135 files, 963 chunks in 133.0s
  (3 deleted files removed from index)
```

### Log Messages

```bash
# Scan summary
grep '\[scan\] Complete:' logs/scan_*.log

# Watch indexing
grep '\[watch\] Indexed:' logs/watch_*.log

# Errors
grep -E 'ERROR|Failed' logs/*.log
```

### Database Check

```bash
# Count documents
sqlite3 ~/.mneme/indexes/myvault/vaultrag.sqlite \
    "SELECT COUNT(*) FROM documents WHERE deleted=0;"

# Count chunks
sqlite3 ~/.mneme/indexes/myvault/vaultrag.sqlite \
    "SELECT COUNT(*) FROM chunks;"

# File types
sqlite3 ~/.mneme/indexes/myvault/vaultrag.sqlite \
    "SELECT file_type, COUNT(*) FROM documents WHERE deleted=0 GROUP BY file_type;"
```

---

## Getting Help

1. **Check logs:** `mneme scan 2>&1 | tee scan.log`
2. **Verify config:** `python -c "import tomllib; tomllib.loads(open('config.toml').read())"`
3. **Test components:**
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```
4. **Report issues:** GitHub Issues with config (redacted), error, and version
