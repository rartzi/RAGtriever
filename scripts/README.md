# Scripts

Development and testing scripts.

## Watcher Management

**Watcher management has moved to the skill's Tools directory:**

```bash
~/.claude/skills/Mneme/Tools/manage-watcher.sh {status|start|stop|restart|health|check|help}
```

See `skills/Mneme/WatcherManagement.md` for full documentation.

---

## Test Scripts

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

---

### test_parallel_gemini_providers.sh

Test script for comparing Gemini provider performance.

---

### generate_provider_report.py

Generate performance reports for image analysis providers.

---

## See Also

- `skills/Mneme/Tools/` - Portable watcher and CLI wrapper scripts
- `docs/USERGUIDE.md` - Comprehensive user guide
- `docs/ARCHITECTURE.md` - System architecture
