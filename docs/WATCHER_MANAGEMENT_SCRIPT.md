# Watcher Management Script

## Overview

The `scripts/manage_watcher.sh` script provides a robust way to manage the RAGtriever watcher with automatic dependency checking and error handling.

## What Was Fixed

### Problems in Original Version
1. No dependency checking (failed silently if venv not activated)
2. No verification of ragtriever installation
3. No handling of shell-specific issues (zsh vs bash)
4. No offline mode environment variables
5. No directory validation

### Improvements Made
1. ✅ **Automatic dependency checking** before starting
2. ✅ **Virtual environment detection** (.venv or venv)
3. ✅ **ragtriever command verification**
4. ✅ **Config file validation**
5. ✅ **Offline mode environment variables** (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE)
6. ✅ **Clean bash shell execution** (avoids shell-specific issues)
7. ✅ **Project root detection** (works from any directory)
8. ✅ **Help command** for usage information
9. ✅ **Check command** for dependency verification only

## Usage

### Basic Commands

```bash
# Check if watcher is running
./scripts/manage_watcher.sh status

# Start watcher (with dependency checks)
./scripts/manage_watcher.sh start

# Stop watcher
./scripts/manage_watcher.sh stop

# Restart watcher
./scripts/manage_watcher.sh restart

# Health check
./scripts/manage_watcher.sh health

# Check dependencies only
./scripts/manage_watcher.sh check

# Show help
./scripts/manage_watcher.sh help
```

## Dependency Checks

The script automatically checks:

1. **Config file exists** (`config.toml`)
   - Error: "Config file not found: config.toml"
   - Fix: Create config or set CONFIG_FILE env var

2. **Virtual environment exists** (`.venv/` or `venv/`)
   - Error: "Virtual environment not found"
   - Fix: `python -m venv .venv && source .venv/bin/activate && pip install -e .`

3. **ragtriever command available**
   - Error: "ragtriever command not found"
   - Fix: `pip install -e .`

## Output

### Successful Start
```
Checking dependencies...
✓ All dependencies OK

Starting watcher...
✓ Watcher started successfully (PID: 41435)
Log file: logs/watch_20260123.log

Recent log:
2026-01-23 15:06:32 [INFO] ragtriever.indexer.indexer: [watch] Starting batched watch...
2026-01-23 15:06:32 [INFO] ragtriever.indexer.change_detector: File watcher started
```

### Dependency Check Failed
```
Checking dependencies...
✗ Config file not found: config.toml
  Current directory: /some/wrong/directory
✗ Virtual environment not found (.venv/ or venv/)
  Run: python -m venv .venv && source .venv/bin/activate && pip install -e .

✗ Dependency check failed - cannot start watcher
```

### Health Check
```
=== Watcher Health Check ===

1. Process running: ✓ (PID: 41435)
2. Log file exists: ✓ (logs/watch_20260123.log)
3. Recent activity: ✓ (log modified in last 5 min)
4. Recent errors: ✓ (none)

=== Summary ===
✓ Watcher is healthy
```

## Environment Variables

### CONFIG_FILE
Override the default config file:

```bash
CONFIG_FILE=my_config.toml ./scripts/manage_watcher.sh start
```

Default: `config.toml`

### Offline Mode
The script automatically sets:
- `HF_HUB_OFFLINE=1` (prevents HuggingFace API calls)
- `TRANSFORMERS_OFFLINE=1` (uses cached models only)

## Files Created

- **logs/watcher.pid** - Process ID of running watcher
- **logs/watch_YYYYMMDD.log** - Daily log file (if logging enabled)

## Common Scenarios

### Scenario 1: First Time Setup
```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Create config
ragtriever init --vault ~/vault --index ~/.ragtriever/indexes/test

# Start watcher (script checks dependencies automatically)
./scripts/manage_watcher.sh start
```

### Scenario 2: After Code Changes
```bash
# Reinstall
pip install -e .

# Restart watcher
./scripts/manage_watcher.sh restart
```

### Scenario 3: After Config Changes
```bash
# Edit config.toml
vim config.toml

# Restart watcher to pick up changes
./scripts/manage_watcher.sh restart
```

### Scenario 4: Watcher Stuck
```bash
# Check health
./scripts/manage_watcher.sh health

# If unhealthy, restart
./scripts/manage_watcher.sh restart
```

### Scenario 5: Production Deployment
```bash
# Verify dependencies
./scripts/manage_watcher.sh check

# Start watcher
./scripts/manage_watcher.sh start

# Verify it's healthy
./scripts/manage_watcher.sh health
```

## Troubleshooting

### Script fails with "command not found"
**Cause:** Shell issues or venv not activated in script

**Fix:** The script now uses `/bin/bash` explicitly for starting the watcher, which avoids zsh/bash differences.

### Watcher starts but exits immediately
**Check logs:**
```bash
cat logs/watch_stdout.log
tail -20 logs/watch_20260123.log
```

**Common causes:**
- Config file errors
- Model not cached (offline mode)
- Vault path invalid

### Dependencies check fails
**Run:**
```bash
./scripts/manage_watcher.sh check
```

This shows exactly what's missing.

## Integration with Skill

The RAGtrieval skill now teaches using this script:

```bash
# Check if watcher is running
./scripts/manage_watcher.sh status

# Start watcher
./scripts/manage_watcher.sh start

# Restart watcher after changes
./scripts/manage_watcher.sh restart
```

This is more reliable than manual commands.

## Best Practices

1. **Always use the script** instead of manual commands
2. **Run health checks** periodically in production
3. **Check dependencies** before deploying
4. **Monitor logs** for errors
5. **Restart after config changes**

## See Also

- `scripts/README.md` - All available scripts
- `docs/LOGGING_CONFIGURATION.md` - Logging setup
- `docs/WHERE_TO_SEE_INDEXING.md` - Verify indexing success
