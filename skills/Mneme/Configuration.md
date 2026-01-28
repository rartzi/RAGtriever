# Configuration

## Configuration Checklist

When setting up or modifying `config.toml`:

### Basic Setup
- [ ] `[vault]` - Set vault root path (absolute path)
- [ ] `[vault]` - Configure ignore patterns (Office temp files, .git, .obsidian/cache)
- [ ] `[index]` - Set index directory (usually `~/.mneme/indexes/<vault-name>`)
- [ ] `[embeddings]` - Choose provider and model
- [ ] `[embeddings]` - Set offline_mode (true if behind proxy)

### Current Vault Configuration
Check `test_config.toml` or the active config to see:
- Vault root path (what content is indexed)
- Index directory (where the database lives)
- Which files are included/ignored

## Image Analysis Provider Options

### For Local OCR (Fastest, No API):
```toml
[image_analysis]
provider = "tesseract"
# Requires: pip install pytesseract + tesseract-ocr system package
```

### For Gemini API (Good quality, API key):
```toml
[image_analysis]
provider = "gemini"
gemini_model = "gemini-2.0-flash"
# Set GEMINI_API_KEY environment variable
```

### For Gemini service account (Best quality, Service account):
```toml
[image_analysis]
provider = "gemini-service-account"

[gemini_service_account]
project_id = "your-gcp-project-id"  # or set GOOGLE_CLOUD_PROJECT
location = "global"  # Recommended for model availability
credentials_file = "/path/to/service-account.json"  # or set GOOGLE_APPLICATION_CREDENTIALS
model = "gemini-2.0-flash-exp"
```

**Gemini service account Checklist:**
- [ ] Credentials file exists and is readable
- [ ] Use `location = "global"` for best model availability
- [ ] Unset `GEMINI_API_KEY` to avoid conflicts
- [ ] Service account has `roles/aiplatform.user` permission

### To Disable Image Analysis:
```toml
[image_analysis]
provider = "off"
```

## Logging Configuration (Audit Trail)

Configure automatic logging with date patterns in `config.toml`:
```toml
[logging]
dir = "logs"                                # Log directory
scan_log_file = "logs/scan_{date}.log"      # Daily rotation: scan_20260123.log
watch_log_file = "logs/watch_{datetime}.log" # Per-run: watch_20260123_142030.log
level = "INFO"                              # DEBUG, INFO, WARNING, ERROR
enable_scan_logging = false                 # Auto-enable for scan (false = manual via CLI)
enable_watch_logging = true                 # Auto-enable for watch (true = always log)
```

**Date patterns:**
- `{date}` -> `20260123` (YYYYMMDD - daily rotation)
- `{datetime}` -> `20260123_142030` (YYYYMMDD_HHMMSS - per-run logs)

**Priority:** CLI flags > config settings > no logging

## Parallel Scanning Configuration

Parallel scanning is enabled by default. Configure in `config.toml`:
```toml
[indexing]
extraction_workers = 8    # Parallel file extraction workers (default: 8)
embed_batch_size = 256    # Cross-file embedding batch size
image_workers = 8         # Parallel image API workers (default: 8)
parallel_scan = true      # Enable/disable parallel scanning
```

## Offline Mode

### Smart Offline Mode (Auto-detect cached models)

**ALWAYS run this check first** - it prevents HuggingFace network errors:

```bash
# Check if default model is cached
if ls ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5 2>/dev/null || \
   ls ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2 2>/dev/null; then
    echo "Model cached - enabling offline mode"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    echo "No cached model found - will download on first run"
fi
```

**One-liner:**
```bash
ls ~/.cache/huggingface/hub/models--* &>/dev/null && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && echo "Offline mode enabled"
```

### Force model update (when needed):
```bash
# Temporarily disable offline mode to download/update model
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mneme scan --config config.toml  # Downloads model
# Then re-enable offline mode
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### List cached models:
```bash
ls ~/.cache/huggingface/hub/ | grep "models--"
```

### Why Auto-Offline Matters

HuggingFace network calls fail frequently due to:
- Corporate proxies
- Rate limits
- Network issues
- Redirect responses (HTML instead of JSON)

**If the model is cached, there's no reason to hit the network.**
