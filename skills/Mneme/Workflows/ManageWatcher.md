# ManageWatcher Workflow

Manage the continuous indexing watcher service.

## Trigger Phrases
- "start watcher"
- "stop watcher"
- "watcher status"
- "is the watcher running"
- "restart watcher"

## Procedure

### Check Status

```bash
./scripts/manage_watcher.sh status
```

Or manually:
```bash
pgrep -f "mneme watch" && echo "Running" || echo "Not running"
```

### Start Watcher

```bash
./scripts/manage_watcher.sh start
```

Or manually:
```bash
nohup ./bin/mneme watch --config config.toml > /dev/null 2>&1 &
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
2. Recent log activity
3. No errors in logs

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
