# Logging Configuration

## Overview

RAGtriever now supports configurable logging for scan and watch operations via the `[logging]` section in `config.toml`. Logs are automatically generated with date/time patterns in the filename.

## Configuration

Add a `[logging]` section to your `config.toml`:

```toml
[logging]
# Log directory (relative to cwd or absolute path)
dir = "logs"

# Log file paths with date/time patterns
scan_log_file = "logs/scan_{date}.log"       # One log per day
watch_log_file = "logs/watch_{datetime}.log" # New log each run

# Log level: DEBUG, INFO, WARNING, ERROR
level = "INFO"

# Auto-enable logging (if true, logging always enabled)
enable_scan_logging = false   # Manual: use --log-file flag
enable_watch_logging = true   # Automatic: always logs
```

## Pattern Substitution

Log file paths support date/time patterns:

| Pattern | Example | Description |
|---------|---------|-------------|
| `{date}` | `20260123` | YYYYMMDD format |
| `{datetime}` | `20260123_142030` | YYYYMMDD_HHMMSS format |

**Examples:**
```toml
scan_log_file = "logs/scan_{date}.log"
# → logs/scan_20260123.log (one log per day)

watch_log_file = "logs/watch_{datetime}.log"
# → logs/watch_20260123_142030.log (new log each run)

scan_log_file = "logs/{date}/scan.log"
# → logs/20260123/scan.log (organized by date)
```

## Behavior

### Priority (highest to lowest):
1. **CLI flags** (`--log-file`, `--log-level`, `--verbose`)
2. **Config settings** (`[logging]` section)
3. **No logging** (default if not configured)

### Enable/Disable Logging:

**Option 1: Config-driven (recommended for production)**
```toml
enable_scan_logging = false   # Manual control via CLI flags
enable_watch_logging = true   # Always logs (watch mode should always log)
```

**Option 2: CLI-driven (recommended for ad-hoc operations)**
```bash
# Override config, use custom log file
ragtriever scan --log-file logs/my_scan.log

# Use config settings (if enable_scan_logging=true)
ragtriever scan  # Logs to scan_{date}.log automatically

# Disable logging even if config enabled
ragtriever scan  # (set enable_scan_logging=false in config)
```

## Use Cases

### 1. Production Watch Mode (Always Log)
```toml
[logging]
dir = "logs"
watch_log_file = "logs/watch_{date}.log"
level = "INFO"
enable_watch_logging = true  # Always log
```

Watch mode will automatically log to `logs/watch_20260123.log` without needing `--log-file` flag.

### 2. On-Demand Scan Logging
```toml
[logging]
enable_scan_logging = false  # Manual control
```

Use CLI flags when needed:
```bash
ragtriever scan --full --log-file logs/scan.log --verbose
```

### 3. Daily Log Rotation
```toml
[logging]
scan_log_file = "logs/scan_{date}.log"
watch_log_file = "logs/watch_{date}.log"
```

Logs automatically rotate daily:
- `logs/scan_20260123.log`
- `logs/scan_20260124.log`
- etc.

### 4. Per-Run Logs (Debugging)
```toml
[logging]
scan_log_file = "logs/scan_{datetime}.log"
watch_log_file = "logs/watch_{datetime}.log"
```

Each run creates a new log file:
- `logs/scan_20260123_140530.log`
- `logs/scan_20260123_152045.log`
- etc.

## Log Formats

All logs use consistent format:
```
YYYY-MM-DD HH:MM:SS [LEVEL] module: message
```

**Example:**
```
2026-01-23 14:20:25 [INFO] ragtriever.indexer.indexer: [scan] Found 142 files to process
2026-01-23 14:21:06 [INFO] ragtriever.indexer.indexer: [scan] Phase 1: 135 files extracted, 0 failed
2026-01-23 14:21:10 [DEBUG] ragtriever.indexer.indexer: [batch] Stored 135 docs, 963 chunks
```

## CLI Reference

### Scan Command
```bash
# Use config logging
ragtriever scan --full

# Override with CLI flag
ragtriever scan --full --log-file logs/custom.log

# Verbose logging
ragtriever scan --full --log-file logs/scan.log --verbose

# Set log level
ragtriever scan --full --log-file logs/scan.log --log-level DEBUG
```

### Watch Command
```bash
# Use config logging (recommended)
ragtriever watch

# Override with CLI flag
ragtriever watch --log-file logs/custom.log

# Verbose logging
ragtriever watch --log-file logs/watch.log --verbose
```

## Monitoring Logs

### Real-Time Monitoring
```bash
# Terminal 1: Run watcher
ragtriever watch

# Terminal 2: Monitor logs
tail -f logs/watch_20260123.log | grep -E "\[watch\]|\[batch\]"
```

### Search Logs
```bash
# Find all scan operations
grep '\[scan\]' logs/scan_*.log

# Find errors
grep 'ERROR' logs/*.log

# Find specific file operations
grep 'test_file.md' logs/watch_*.log

# Count events
grep -c 'File created' logs/watch_20260123.log
```

## Best Practices

1. **Always enable watch logging** (`enable_watch_logging = true`)
   - Watch mode runs continuously, logs are essential for debugging

2. **Use daily rotation for production** (`{date}` pattern)
   - Easier to manage than per-run logs
   - grep across date range: `grep pattern logs/*_2026012*.log`

3. **Use `INFO` level for production, `DEBUG` for troubleshooting**
   - DEBUG generates high volume
   - INFO captures all significant events

4. **Keep logs for 30-90 days**
   - Useful for debugging issues that appear later
   - Archive or compress older logs

5. **Monitor disk usage**
   - Large vaults can generate MB/day of logs
   - Set up log rotation or cleanup scripts

## Troubleshooting

### Logs not being created
```bash
# Check config
grep -A 10 '\[logging\]' config.toml

# Verify directory exists
mkdir -p logs

# Check permissions
ls -la logs/
```

### Log path not resolving
```bash
# Test pattern substitution
python3 -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M%S'))"
```

### Logs empty
- Check `enable_scan_logging` / `enable_watch_logging` settings
- Verify CLI flags are correct
- Check log level isn't filtering out messages

## Migration from CLI-only Logging

**Before (CLI flags only):**
```bash
ragtriever scan --full --log-file logs/scan.log --verbose
ragtriever watch --log-file logs/watch.log --verbose
```

**After (config-driven):**
```toml
[logging]
scan_log_file = "logs/scan_{date}.log"
watch_log_file = "logs/watch_{date}.log"
level = "INFO"
enable_scan_logging = false  # Manual for scan
enable_watch_logging = true  # Automatic for watch
```

```bash
ragtriever scan --full  # No flags needed if enable_scan_logging=true
ragtriever watch        # Automatically logs to logs/watch_20260123.log
```

CLI flags still work and override config settings.
