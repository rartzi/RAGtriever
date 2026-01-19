---
name: RAGtrieval
description: Answer questions from indexed vault content using RAGtriever search, plus setup, configuration, and troubleshooting
---

# RAGtrieval Skill

Claude Code skill for working with RAGtriever (local-first vault indexer with hybrid retrieval).

**Primary Purpose:** Search and answer questions from the indexed vault content, not the RAGtriever codebase.

## When to Use This Skill

- **Answering questions about vault content** - Always use RAGtriever to search indexed content
- Setting up RAGtriever for a new vault
- Configuring image analysis (Tesseract, Gemini API, or Vertex AI)
- Troubleshooting scanning or indexing issues
- Testing search and retrieval functionality
- Working with embeddings and offline mode

## CRITICAL: Answering Questions from Vault Content

**When the user asks ANY question that could be answered from vault content:**

1. **ALWAYS use `ragtriever query` to search the indexed vault**
2. **NEVER answer from memory or assumptions about the codebase**
3. **The vault content is the source of truth** - not the RAGtriever repository code itself

### Question Types to Search For:
- "What ideas do we have?" → Search vault for ideas
- "List all projects" → Search vault for projects
- "Who is working on X?" → Search vault for people/assignments
- "What's the status of Y?" → Search vault for status updates
- "Find information about Z" → Search vault content

### Example Query Pattern:
```bash
source .venv/bin/activate
ragtriever query --config config.toml "user's question keywords" --k 15
```

### Current Vault Configuration:
Check `test_config.toml` or the active config to see:
- Vault root path (what content is indexed)
- Index directory (where the database lives)
- Which files are included/ignored

## Pre-flight Checks

Before running any RAGtriever commands:

1. **Check virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **For offline mode (corporate proxies):**
   ```bash
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

3. **Verify cached models:**
   ```bash
   ls ~/.cache/huggingface/hub/
   ```

## Configuration Checklist

When setting up or modifying `config.toml`:

### Basic Setup
- [ ] `[vault]` - Set vault root path (absolute path)
- [ ] `[vault]` - Configure ignore patterns (Office temp files, .git, .obsidian/cache)
- [ ] `[index]` - Set index directory (usually `~/.ragtriever/indexes/<vault-name>`)
- [ ] `[embeddings]` - Choose provider and model
- [ ] `[embeddings]` - Set offline_mode (true if behind proxy)

### Image Analysis Provider Options

#### For Local OCR (Fastest, No API):
```toml
[image_analysis]
provider = "tesseract"
# Requires: pip install pytesseract + tesseract-ocr system package
```

#### For Gemini API (Good quality, API key):
```toml
[image_analysis]
provider = "gemini"
gemini_model = "gemini-2.0-flash"
# Set GEMINI_API_KEY environment variable
```

#### For Vertex AI (Best quality, Service account):
```toml
[image_analysis]
provider = "vertex_ai"

[vertex_ai]
project_id = "your-gcp-project-id"  # or set GOOGLE_CLOUD_PROJECT
location = "global"  # Recommended for model availability
credentials_file = "/path/to/service-account.json"  # or set GOOGLE_APPLICATION_CREDENTIALS
model = "gemini-2.0-flash-exp"
```

**Vertex AI Checklist:**
- [ ] Credentials file exists and is readable
- [ ] Use `location = "global"` for best model availability
- [ ] Unset `GEMINI_API_KEY` to avoid conflicts
- [ ] Service account has `roles/aiplatform.user` permission

#### To Disable Image Analysis:
```toml
[image_analysis]
provider = "off"
```

## Common Commands

### Initial Setup
```bash
# Install RAGtriever in development mode
pip install -e ".[dev]"

# Generate starter config
ragtriever init --vault "/path/to/vault" --index "~/.ragtriever/indexes/myvault"

# Edit config.toml to configure image analysis and embeddings
```

### Scanning and Indexing
```bash
# Full scan (re-index everything) - uses parallel processing by default
ragtriever scan --config config.toml --full

# Full scan with explicit parallel settings
ragtriever scan --config config.toml --full --workers 8

# Sequential scan (disable parallelization)
ragtriever scan --config config.toml --full --no-parallel

# Incremental scan (only changed files)
ragtriever scan --config config.toml

# Watch mode (continuous indexing)
ragtriever watch --config config.toml
```

### Parallel Scanning (3.6x faster)
Parallel scanning is enabled by default. Configure in `config.toml`:
```toml
[indexing]
extraction_workers = 4    # Parallel file extraction workers
embed_batch_size = 256    # Cross-file embedding batch size
image_workers = 4         # Parallel image API workers
parallel_scan = true      # Enable/disable parallel scanning
```

### Querying
```bash
# Basic search (hybrid: semantic + lexical)
ragtriever query --config config.toml "search term" --k 10

# More results
ragtriever query --config config.toml "kubernetes deployment" --k 20
```

### MCP Server (Claude Desktop Integration)
```bash
# Start MCP server
ragtriever mcp --config config.toml

# Add to Claude Desktop config.json:
{
  "mcpServers": {
    "ragtriever": {
      "command": "ragtriever",
      "args": ["mcp", "--config", "/full/path/to/config.toml"]
    }
  }
}
```

## Troubleshooting

### Issue: Office Temp Files (`~$*.pptx`)

**Symptom:** `PackageNotFoundError: Package not found at '~$document.pptx'`

**Solution:**
1. Close all Office applications before scanning
2. Add to ignore patterns in config.toml:
   ```toml
   [vault]
   ignore = [
       ".git/**",
       ".obsidian/cache/**",
       "**/.DS_Store",
       "**/~$*",           # Office temp files
       "**/.~lock.*"       # LibreOffice lock files
   ]
   ```
3. Remove existing temp files:
   ```bash
   find /path/to/vault -name "~$*" -delete
   ```

### Issue: Offline Mode with Uncached Model

**Symptom:** `OSError: We couldn't connect to 'https://huggingface.co'`

**Solution:**
```bash
# Option 1: Temporarily disable offline mode
export HF_OFFLINE_MODE=0
ragtriever scan --config config.toml

# Then re-enable offline mode
export HF_OFFLINE_MODE=1

# Option 2: Use a cached model
ls ~/.cache/huggingface/hub/
# Update config.toml to use one of the cached models
```

### Issue: Vertex AI Rate Limits (429 errors)

**Symptom:** `Vertex AI analysis failed: TooManyRequests`

**Expected Behavior:** This is normal with multiple images. The scan continues and processes other files.

**Solutions:**
- Most images will succeed despite some 429 errors
- Re-run scan - successful images are cached
- Consider using Gemini API or Tesseract for large image sets

### Issue: Vertex AI Authentication Errors

**Solution:**
1. Verify credentials file exists:
   ```bash
   test -r $GOOGLE_APPLICATION_CREDENTIALS && echo "OK"
   ```
2. Check environment variables:
   ```bash
   echo $GOOGLE_APPLICATION_CREDENTIALS
   echo $GOOGLE_CLOUD_PROJECT
   ```
3. Ensure `GEMINI_API_KEY` is **unset** (conflicts with Vertex AI):
   ```bash
   unset GEMINI_API_KEY
   ```
4. Test authentication:
   ```bash
   gcloud auth activate-service-account \
       --key-file=$GOOGLE_APPLICATION_CREDENTIALS
   gcloud ai models list --region=global
   ```

### Issue: FTS5 Syntax Errors with Special Characters

**Symptom:** `fts5: syntax error near "-"`

**Solution:** This is handled automatically as of recent versions. Queries with hyphens, slashes, etc. work correctly:
```bash
ragtriever query "T-DXd treatment"     # Works
ragtriever query "CDK4/6 inhibitor"    # Works
ragtriever query "HR+/HER2- cancer"    # Works
```

## Development Workflow

### Making Changes
```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes to code

# 3. Reinstall after code changes
pip install -e ".[dev]"

# 4. Run linting
ruff check src/ tests/

# 5. Run tests
pytest
```

### Testing Image Analysis
```bash
# Test with small vault first (1-3 images)
ragtriever scan --config config.toml --full

# Check logs for errors
# tail -100 scan.log

# Query for image content
ragtriever query --config config.toml "image description" --k 5

# Verify results include image metadata
```

## Key Files Reference

- `src/ragtriever/extractors/image.py` - Image extractor implementations
- `src/ragtriever/config.py` - Configuration management
- `src/ragtriever/indexer/indexer.py` - Main indexer orchestration
- `src/ragtriever/retrieval/retriever.py` - Hybrid search implementation
- `examples/config.toml.example` - Configuration template
- `docs/architecture.md` - Complete system architecture
- `docs/vertex_ai_setup.md` - Vertex AI setup guide
- `docs/troubleshooting.md` - Detailed troubleshooting
- `IMPROVEMENTS.md` - Planned enhancements (inc. Gemini 3)

## Architecture Quick Reference

### Data Flow (Parallel Pipeline)
```
Vault (filesystem)
  → File Discovery (Reconciler)
  → Phase 1: Parallel Extraction + Chunking (ThreadPoolExecutor)
  → Phase 2: Batched Embedding (cross-file batches of 256)
  → Phase 3: Parallel Image Analysis (if enabled)
  → Store (SQLite + FTS5)
  → Retrieval (hybrid search)
```

### Image Analysis Pipeline
```
Image file
  → ImageExtractor (Tesseract/Gemini/Vertex AI)
  → Structured analysis (description, OCR, topics, entities)
  → Chunking
  → Indexing
  → Searchable via query
```

### Execution Modes
- **CLI**: `ragtriever scan/query/watch/mcp`
- **Watch Mode**: Continuous filesystem monitoring
- **MCP Server**: Integration with Claude Desktop
- **Python API**: Programmatic import and use

## Success Criteria

After running a scan, verify:
- [ ] Scan completes without fatal errors
- [ ] Rate limit errors (429) are logged but don't stop scan
- [ ] Database created: `~/.ragtriever/indexes/<vault>/vaultrag.sqlite`
- [ ] Query returns results
- [ ] Image content is searchable (if images in vault)
- [ ] Metadata includes analysis_provider and model name

**Check indexing stats:**
```bash
# Count documents
sqlite3 ~/.ragtriever/indexes/myvault/vaultrag.sqlite \
    "SELECT COUNT(*) FROM documents WHERE deleted=0;"

# Count chunks
sqlite3 ~/.ragtriever/indexes/myvault/vaultrag.sqlite \
    "SELECT COUNT(*) FROM chunks;"

# Show file types
sqlite3 ~/.ragtriever/indexes/myvault/vaultrag.sqlite \
    "SELECT file_type, COUNT(*) FROM documents WHERE deleted=0 GROUP BY file_type;"
```

## Quick Workflows

### Setup New Vault (Tesseract)
```bash
source .venv/bin/activate
ragtriever init --vault ~/vault --index ~/.ragtriever/indexes/myvault
# Edit config.toml: set provider = "tesseract"
ragtriever scan --config config.toml --full
ragtriever query --config config.toml "test query" --k 5
```

### Setup with Vertex AI
```bash
source .venv/bin/activate
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset GEMINI_API_KEY
ragtriever init --vault ~/vault --index ~/.ragtriever/indexes/myvault
# Edit config.toml: configure [vertex_ai] section
ragtriever scan --config config.toml --full
ragtriever query --config config.toml "image content" --k 5
```

### Incremental Updates
```bash
source .venv/bin/activate
ragtriever scan --config config.toml  # Only changed files
```

### Watch Mode (Background)
```bash
source .venv/bin/activate
nohup ragtriever watch --config config.toml > watch.log 2>&1 &
```

## Additional Resources

- **README.md**: User guide and quick start
- **CLAUDE.md**: Project architecture for Claude Code
- **docs/architecture.md**: Complete system documentation
- **docs/vertex_ai_setup.md**: Vertex AI service account setup
- **docs/troubleshooting.md**: Comprehensive troubleshooting guide
- **IMPROVEMENTS.md**: Planned features (Gemini 3, better error handling, etc.)

## Tips for Claude Code Users

When helping users with RAGtriever:

1. **ALWAYS search vault content first** - When user asks questions, use `ragtriever query` to search indexed vault
2. **Vault content ≠ RAGtriever code** - Don't confuse searching the vault (user's content) with RAGtriever repository code
3. **Always check config first** - Most issues stem from configuration
4. **Verify virtual environment** - Commands fail if venv not activated
5. **Check file paths** - Use absolute paths, expand ~
6. **Office temp files** - First thing to check if PPTX extraction fails
7. **Offline mode** - Corporate users need this; verify model is cached
8. **Rate limits are OK** - 429 errors are expected with many images
9. **Test incrementally** - Small vault first, then scale up

## Notes

- RAGtriever is a standalone tool, not a Claude Code skill itself
- This skill provides workflow assistance for using RAGtriever
- RAGtriever runs independently and can work without Claude
- Optional MCP integration enables Claude Desktop to search your vault
