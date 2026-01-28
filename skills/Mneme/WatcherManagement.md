# Watcher Management

**The watcher automatically handles:**
- Individual file changes (create, modify, delete, move)
- Directory operations (delete entire folders, move/rename folders)
- All files within deleted or moved directories are updated in the index
- No manual intervention needed for vault reorganization

## Check if Watcher is Running

```bash
# Check for running watcher process
ps aux | grep -E "[r]agtriever watch"

# Or more precise:
pgrep -f "mneme watch"

# If output: process is running
# If no output: watcher is not running
```

## Start the Watcher

```bash
# Preferred: Use the management script (handles everything)
./scripts/manage_watcher.sh start

# Or use the wrapper directly (foreground)
./bin/mneme watch --config config.toml

# Background with logging
nohup ./bin/mneme watch --config config.toml > /dev/null 2>&1 &
echo $! > logs/watcher.pid  # Save PID for later
```

## Stop the Watcher

```bash
# Method 1: Kill all mneme watch processes
pkill -f "mneme watch"

# Method 2: Kill specific PID (if you saved it)
kill $(cat logs/watcher.pid)

# Method 3: Find and kill interactively
ps aux | grep "[r]agtriever watch"
kill <PID>

# Verify it stopped
ps aux | grep -E "[r]agtriever watch" || echo "Watcher stopped"
```

## Restart the Watcher

```bash
# Preferred: Use the management script
./scripts/manage_watcher.sh restart

# Or manually:
pkill -f "mneme watch"
sleep 2
nohup ./bin/mneme watch --config config.toml &
echo $! > logs/watcher.pid
```

## Check Watcher Status

```bash
# Is it running?
if pgrep -f "mneme watch" > /dev/null; then
    echo "Watcher is running"
    ps aux | grep "[r]agtriever watch" | grep -v grep
else
    echo "Watcher is not running"
fi

# Check recent activity (if logging enabled)
tail -20 logs/watch_$(date +%Y%m%d).log
```

## Watcher Health Check

```bash
# 1. Check if process is running
pgrep -f "mneme watch" > /dev/null || echo "ERROR: Watcher not running"

# 2. Check if it's actually indexing (log activity in last 5 minutes)
find logs/ -name "watch_*.log" -mmin -5 | grep -q . && echo "Recent activity" || echo "No recent log activity"

# 3. Check for errors in logs
tail -100 logs/watch_$(date +%Y%m%d).log | grep -i error && echo "Errors found" || echo "No recent errors"
```

## Watcher Management Script

**For convenience, use the included management script:**

```bash
# Check status
./scripts/manage_watcher.sh status

# Start watcher
./scripts/manage_watcher.sh start

# Stop watcher
./scripts/manage_watcher.sh stop

# Restart watcher
./scripts/manage_watcher.sh restart

# Run health check
./scripts/manage_watcher.sh health
```

**The script handles:**
- Virtual environment activation
- PID file management
- Graceful shutdown
- Health checks (process, logs, errors)
- Log file detection

## When to Restart the Watcher

Restart the watcher when:
- Config changes have been made
- Code has been updated
- Watcher stops responding
- After system reboot
- Log errors indicate issues

## Watcher Management Workflow

When user mentions watcher issues:
```bash
# Use the management script - handles everything automatically
./scripts/manage_watcher.sh status   # Check if running
./scripts/manage_watcher.sh start    # Start (auto-creates venv if needed)
./scripts/manage_watcher.sh restart  # Restart
./scripts/manage_watcher.sh health   # Full health check

# Manual alternative:
pgrep -f "mneme watch" || echo "Not running"
nohup ./bin/mneme watch --config config.toml &
tail -20 logs/watch_$(date +%Y%m%d).log
```
