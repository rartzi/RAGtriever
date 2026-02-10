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
  - [Query Server](#query-server)
- [MCP Server Integration](#mcp-server-integration)
- [Claude Code Skill](#claude-code-skill)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install
pip install -e .

# Optional: FAISS support for large vaults (>10K chunks)
pip install -e ".[faiss]"

# 2. Create config
mneme init --vault "/path/to/vault" --index "~/.mneme/indexes/myvault"

# 3. Index your vault
mneme scan --full

# 4. Search
mneme query "your search term"

# 5. Watch for changes + query server (optional)
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

### Retrieval Configuration

```toml
[retrieval]
top_k = 10                   # Default results per query
k_vec = 20                   # Candidates from vector search
k_lex = 20                   # Candidates from lexical search
fusion_algorithm = "rrf"     # "rrf" (recommended) or "weighted"
rrf_k = 60                   # RRF constant (higher = more weight to lower-ranked)

# Cross-encoder reranking (20-30% quality improvement, +100-200ms)
use_rerank = false
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_device = "cpu"        # cpu | cuda | mps

# FAISS vector index (for vaults with >10K chunks)
use_faiss = false            # pip install mneme[faiss] required

# Boosts (all enabled by default)
backlink_boost_enabled = true
backlink_boost_weight = 0.1  # +10% per backlink
backlink_boost_cap = 2.0     # Max 2x boost
recency_boost_enabled = true
recency_fresh_days = 14      # +10% for files < 14 days old
recency_recent_days = 60     # +5% for files < 60 days old
recency_old_days = 180       # -2% for files > 180 days old
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

Continuously monitor vault for changes. The watcher also starts a built-in **query server** that provides ~0.1s query response times (vs ~5s cold-start).

```bash
# Start watcher (includes query server)
mneme watch

# With custom config
mneme watch --config my_config.toml

# With logging
mneme watch --log-file logs/watch.log --verbose

# Batched mode options
mneme watch --batch-size 20 --batch-timeout 10
```

When the watcher is running, `mneme query` automatically routes through the query server via unix socket for fast responses. Use `--no-socket` to bypass:

### query Command

Search your indexed vault. When the watcher is running, queries automatically route through its built-in query server (~0.1s vs ~5s cold-start).

```bash
# Basic search (auto-routes through watcher socket if available)
mneme query "search term"

# More results
mneme query "kubernetes deployment" --k 20

# With reranking (better quality, +100-200ms)
mneme query "machine learning" --rerank

# Force cold-start (bypass watcher socket)
mneme query "search term" --no-socket

# JSON output
mneme query "project status" --json
```

### list-docs Command

List indexed documents in the vault.

```bash
# List all files
mneme list-docs

# Filter by path
mneme list-docs --path "projects/"

# Filter by vault (multi-vault)
mneme list-docs --vaults my-thoughts
```

### text-search Command

Search using BM25 lexical matching only (bypasses semantic search). Best for exact phrases.

```bash
# Exact phrase search
mneme text-search "orchestration patterns"

# With path filter
mneme text-search "agent loop" --path "notes/" --k 10

# Force cold-start
mneme text-search "exact phrase" --no-socket
```

### backlinks Command

Show most-linked documents (hub analysis).

```bash
# Top 20 most-linked documents
mneme backlinks

# Top 10
mneme backlinks --limit 10

# Check specific documents
mneme backlinks --paths "projects/alpha.md"
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

### Query Server

The watcher includes a built-in query server that dramatically improves query latency:

| Mode | Latency | How |
|------|---------|-----|
| **Cold-start** | ~3-5s | Python startup + model loading |
| **Query server** | ~0.1-0.3s | Pre-loaded models via unix socket |

The query server starts automatically with the watcher. The `mneme query` CLI transparently routes through the socket when available and falls back to cold-start when the watcher is not running.

```bash
# Check if query server is active
~/.claude/skills/Mneme/Tools/manage-watcher.sh health

# Force cold-start (bypass socket)
mneme query "search term" --no-socket
```

### What the Watcher Handles

- Individual file changes (create, modify, delete, move)
- Directory operations (delete/move entire folders)
- Batched processing for efficiency
- Automatic index updates
- Built-in query server for fast CLI queries

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
| `vault_list_docs` | List indexed documents (with optional path filter) |
| `vault_text_search` | BM25 lexical search only (exact phrase matching) |
| `vault_backlinks` | Get backlink counts (find hub documents) |

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
- **AgenticSearch** - Iterative multi-step search for complex, multi-hop questions
- **SetupVault** - Initial vault setup and configuration
- **ConfigureImageAnalysis** - Configure image analysis providers (Tesseract/Gemini)
- **ManageWatcher** - Start/stop/status for the watcher service
- **Scan** - Run full or incremental scans
- **Troubleshoot** - Diagnose and fix common issues

Plus context files for detailed documentation on search best practices, configuration, commands, and architecture.

### Example Phrases for Claude Code

Just talk naturally - Claude Code will invoke the appropriate workflow:

**Searching your vault:**
- "What does my vault say about project planning?"
- "Search my vault for meeting notes"
- "Find information about authentication in my notes"
- "What do I have on machine learning?"

**Managing the watcher:**
- "Start the watcher"
- "Is the watcher running?"
- "Stop the watcher"
- "Check watcher health"

**Scanning:**
- "Run a full scan"
- "Do an incremental scan"
- "Re-index my vault"

**Setup and configuration:**
- "Setup mneme for my vault at ~/Documents/notes"
- "Configure image analysis with Gemini"
- "Initialize a new vault"

**Troubleshooting:**
- "The watcher isn't working"
- "I'm getting an error with scanning"
- "Help me fix the indexing issue"

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
Scan complete: 149 files, 3106 chunks in 182.3s
  (7 images analyzed, 0 deleted files removed from index)

# Incremental scan (manifest-based skip)
Scan complete: 0 files, 0 chunks in 0.0s
  (149 files skipped unchanged)
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
