# Scripts

Operational scripts for testing, development, and watcher management.

## Available Scripts

### manage_watcher.sh

Watcher management script for starting, stopping, and monitoring the RAGtriever watcher.

**Usage:**
```bash
./scripts/manage_watcher.sh {status|start|stop|restart|health}
```

**Commands:**
- `status` - Check if watcher is running
- `start` - Start the watcher in background
- `stop` - Stop the watcher gracefully
- `restart` - Restart the watcher
- `health` - Run comprehensive health check

**Features:**
- Automatic virtual environment activation
- PID file management (`logs/watcher.pid`)
- Graceful shutdown with fallback to force kill
- Health checks: process, log activity, errors
- Color-coded status messages

**Examples:**
```bash
# Check if watcher is running
./scripts/manage_watcher.sh status

# Start in background
./scripts/manage_watcher.sh start

# Check health
./scripts/manage_watcher.sh health

# Restart after config change
./scripts/manage_watcher.sh restart
```

**Environment:**
- Uses `CONFIG_FILE` env var (default: `config.toml`)
- Saves PID to `logs/watcher.pid`
- Checks for logs in `logs/watch_YYYYMMDD.log`

---

### test_scan_and_watch.sh

Automated test script for scanning and watching with logging and profiling.

**Usage:**
```bash
./scripts/test_scan_and_watch.sh
```

**What it does:**
1. Cleans the database (deletes existing index)
2. Runs a full scan with 10 workers and profiling
3. Displays profiling summary
4. Displays scan log summary
5. Prompts to start the watcher for testing

**Output:**
- Logs: `logs/scan_YYYYMMDD_HHMMSS.log`
- Profile: `logs/scan_profile_YYYYMMDD_HHMMSS.txt`
- Watch logs: `logs/watch_YYYYMMDD_HHMMSS.log`

All logs are timestamped and saved to the `logs/` directory.

---

## See Also

**Documentation:**
- `docs/SCAN_AND_WATCH_QUICKSTART.md` - Quick reference for manual commands
- `docs/SCAN_AND_WATCH_TESTING.md` - Comprehensive testing guide
- `docs/LOGGING_CONFIGURATION.md` - Logging setup and configuration
- `docs/WHERE_TO_SEE_INDEXING.md` - How to verify indexing success
