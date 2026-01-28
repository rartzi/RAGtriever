# Troubleshoot Workflow

Diagnose and resolve common Mneme issues.

## Trigger Phrases
- "error"
- "not working"
- "fix"
- "troubleshoot"
- "help with..."

## Quick Diagnosis

### 1. Check Configuration

```bash
# Verify config exists
test -f config.toml && echo "Config OK" || echo "Config missing"

# Check vault path
grep -A2 '\[vault\]' config.toml
```

### 2. Check Process Status

```bash
# Watcher running?
pgrep -f "mneme watch" && echo "Watcher running" || echo "Watcher not running"
```

### 3. Check Recent Logs

```bash
# Scan errors
grep -i error logs/scan*.log | tail -20

# Watch errors
grep -i error logs/watch*.log | tail -20
```

### 4. Check Database

```bash
mneme status
```

## Common Issues

### Office Temp Files (`~$*.pptx`)

**Symptom:** `PackageNotFoundError: Package not found at '~$document.pptx'`

**Solution:**
1. Close Office applications
2. Add ignore pattern:
   ```toml
   [vault]
   ignore = ["**/~$*"]
   ```
3. Remove temp files:
   ```bash
   find /path/to/vault -name "~$*" -delete
   ```

### Offline Mode Errors

**Symptom:** `OSError: We couldn't connect to 'https://huggingface.co'`

**Solution:**
```bash
# Check if model is cached
ls ~/.cache/huggingface/hub/models--*

# Option 1: Enable offline mode
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Option 2: Temporarily download
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mneme scan --config config.toml
```

### Rate Limit Errors (429)

**Symptom:** `TooManyRequests` with Gemini

**This is normal.** Solutions:
- Re-run scan (cached images skip)
- Use Tesseract for large image sets
- Reduce `image_workers` in config

### Authentication Errors (Gemini)

**Check:**
```bash
# Credentials readable?
test -r $GOOGLE_APPLICATION_CREDENTIALS && echo "OK"

# Env vars set?
echo $GOOGLE_APPLICATION_CREDENTIALS
echo $GOOGLE_CLOUD_PROJECT

# GEMINI_API_KEY unset? (conflicts with service account)
echo $GEMINI_API_KEY  # Should be empty
```

**Fix:**
```bash
unset GEMINI_API_KEY
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"
export GOOGLE_CLOUD_PROJECT="your-project"
```

### FTS5 Syntax Errors

**Symptom:** `fts5: syntax error near "-"`

**Solution:** Update to recent version. Special characters (hyphens, slashes) are now handled automatically.

### Watcher Not Indexing

**Check:**
1. Process running: `pgrep -f "mneme watch"`
2. Recent log activity: `find logs/ -name "watch_*.log" -mmin -5`
3. Vault path accessible: `ls "$(grep root config.toml | cut -d'"' -f2)"`

**Fix:**
```bash
~/.claude/skills/Mneme/Tools/manage-watcher.sh restart
```

### Empty Search Results

**Check:**
1. Database has content: `mneme status`
2. Query is reasonable: try simpler terms
3. Index is fresh: run incremental scan

**Fix:**
```bash
mneme scan --config config.toml
mneme query --config config.toml "simple term" --k 20
```

## Diagnostic Commands

```bash
# Full health check
~/.claude/skills/Mneme/Tools/manage-watcher.sh health

# Count indexed files
sqlite3 ~/.mneme/indexes/*/vaultrag.sqlite "SELECT COUNT(*) FROM documents WHERE deleted=0;"

# Check file types
sqlite3 ~/.mneme/indexes/*/vaultrag.sqlite "SELECT file_type, COUNT(*) FROM documents WHERE deleted=0 GROUP BY file_type;"

# Recent errors
grep -E 'ERROR|Failed' logs/*.log | tail -20
```

## When to Escalate

If issues persist after trying above solutions:
1. Check `docs/troubleshooting.md` for detailed guidance
2. Review `CHANGELOG.md` for recent changes
3. Check GitHub issues for similar problems
