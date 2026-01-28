# SetupVault Workflow

Initialize and configure Mneme for a new vault.

## Trigger Phrases
- "setup mneme"
- "initialize vault"
- "create index"
- "configure mneme for my vault"

## Procedure

### 1. Initialize the Vault

```bash
./bin/mneme init --vault "/path/to/vault" --index "~/.mneme/indexes/vaultname"
```

This creates a starter `config.toml` file.

### 2. Configure Settings

Edit `config.toml` to set:

**Required:**
- `[vault]` root path (absolute path)
- `[vault]` ignore patterns
- `[index]` directory
- `[embeddings]` provider and model

**Optional:**
- `[image_analysis]` provider (tesseract/gemini/gemini-service-account/off)
- `[logging]` configuration
- `[indexing]` parallel settings

### 3. Configure Ignore Patterns

Add to `[vault]` section:
```toml
ignore = [
    ".git/**",
    ".obsidian/cache/**",
    "**/.DS_Store",
    "**/~$*",           # Office temp files
    "**/.~lock.*"       # LibreOffice lock files
]
```

### 4. Check Offline Mode (if needed)

For corporate networks or cached models:
```bash
ls ~/.cache/huggingface/hub/models--* &>/dev/null && \
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
    echo "Offline mode enabled"
```

### 5. Run Initial Scan

```bash
./bin/mneme scan --config config.toml --full
```

Monitor output for:
- File count and chunk count
- Any extraction errors
- Rate limit warnings (OK for images)

### 6. Verify Setup

```bash
# Check status
./bin/mneme status

# Test query
./bin/mneme query --config config.toml "test query" --k 5
```

### 7. (Optional) Start Watcher

For continuous indexing:
```bash
./scripts/manage_watcher.sh start
```

## Quick Setup Commands

### With Tesseract (Local OCR)
```bash
./bin/mneme init --vault ~/vault --index ~/.mneme/indexes/myvault
# Edit config.toml: set provider = "tesseract"
./bin/mneme scan --config config.toml --full
./bin/mneme query --config config.toml "test" --k 5
```

### With Gemini Service Account
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"
unset GEMINI_API_KEY

./bin/mneme init --vault ~/vault --index ~/.mneme/indexes/myvault
# Edit config.toml: configure [gemini_service_account] section
./bin/mneme scan --config config.toml --full
```

## Checklist

- [ ] Vault path is absolute
- [ ] Index directory is set
- [ ] Ignore patterns include Office temp files
- [ ] Embedding model is configured
- [ ] Image analysis provider is set (or "off")
- [ ] Initial scan completed successfully
- [ ] Test query returns results
