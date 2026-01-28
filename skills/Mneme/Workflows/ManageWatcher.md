# ManageWatcher Workflow

Manage the continuous indexing watcher service.

## Trigger Phrases
- "start watcher"
- "stop watcher"
- "watcher status"
- "is the watcher running"
- "restart watcher"

## IMPORTANT: Logging is Mandatory

The watcher **always writes to log files** in `logs/watch_YYYYMMDD.log` for audit purposes.
The `manage_watcher.sh` script handles this automatically.

## Procedure

### Check Status

```bash
./scripts/manage_watcher.sh status
```

Or manually:
```bash
pgrep -f "mneme watch" && echo "Running" || echo "Not running"
```

### Start Watcher (with automatic logging)

```bash
./scripts/manage_watcher.sh start
```

The script automatically:
1. Creates `logs/` directory if needed
2. Starts watcher with logging to `logs/watch_YYYYMMDD.log`
3. Saves PID to `logs/watcher.pid`
4. Sets offline mode environment variables

Or manually (with explicit logging):
```bash
mkdir -p logs
nohup ./bin/mneme watch --config config.toml \
    --log-file logs/watch_$(date +%Y%m%d).log \
    > logs/watcher_stdout.log 2>&1 &
echo $! > logs/watcher.pid
```

### Stop Watcher

```bash
./scripts/manage_watcher.sh stop
```

Or manually:
```bash
pkill -f "mneme watch"
```

### Restart Watcher

```bash
./scripts/manage_watcher.sh restart
```

### Health Check

```bash
./scripts/manage_watcher.sh health
```

This checks:
1. Process is running
2. Log file exists and is recent
3. No errors in recent log entries

## What the Watcher Handles

- Individual file changes (create, modify, delete, move)
- Directory operations (delete entire folders, move/rename folders)
- All files within deleted or moved directories
- Catch-up on files modified while watcher was stopped

## When to Restart

Restart the watcher when:
- Config changes have been made
- Code has been updated
- Watcher stops responding
- After system reboot
- Log errors indicate issues

## Monitoring

**Check recent activity:**
```bash
tail -20 logs/watch_$(date +%Y%m%d).log
```

**Real-time monitoring:**
```bash
tail -f logs/watch_$(date +%Y%m%d).log | grep '\[watch\] Indexed:'
```

**Check for errors:**
```bash
grep -i error logs/watch_$(date +%Y%m%d).log
```

## Troubleshooting

**Watcher not starting:**
1. Check if already running: `pgrep -f "mneme watch"`
2. Check config exists: `test -f config.toml`
3. Check logs directory: `mkdir -p logs`

**Watcher not indexing:**
1. Verify vault path is accessible
2. Check file permissions
3. Review logs for errors

**High CPU/memory:**
1. Check ignore patterns for large files
2. Reduce image_workers if using API providers
