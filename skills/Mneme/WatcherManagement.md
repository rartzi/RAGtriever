# Watcher Management

**The watcher automatically handles:**
- Individual file changes (create, modify, delete, move)
- Directory operations (delete entire folders, move/rename folders)
- All files within deleted or moved directories are updated in the index
- Catch-up on files modified while watcher was stopped
- No manual intervention needed for vault reorganization

## Using the Skill's Management Script

The skill provides a portable management script in `Tools/manage-watcher.sh`:

```bash
# Check status
~/.claude/skills/Mneme/Tools/manage-watcher.sh status

# Start watcher (auto-installs mneme if needed)
~/.claude/skills/Mneme/Tools/manage-watcher.sh start

# Stop watcher
~/.claude/skills/Mneme/Tools/manage-watcher.sh stop

# Restart watcher
~/.claude/skills/Mneme/Tools/manage-watcher.sh restart

# Run health check
~/.claude/skills/Mneme/Tools/manage-watcher.sh health

# Check installation
~/.claude/skills/Mneme/Tools/manage-watcher.sh check
```

**The script handles:**
- Auto-installing mneme if not found
- Log file creation with daily rotation
- PID file management
- Graceful shutdown
- Health checks (process, logs, errors)
- Offline mode environment variables

## Check if Watcher is Running

```bash
# Using the skill's script (recommended)
~/.claude/skills/Mneme/Tools/manage-watcher.sh status

# Or manually:
pgrep -f "mneme watch" && echo "Running" || echo "Not running"
```

## Start the Watcher

```bash
# Preferred: Use the skill's script (handles everything)
~/.claude/skills/Mneme/Tools/manage-watcher.sh start

# Or manually with logging (REQUIRED for audit):
mkdir -p logs
nohup mneme watch --config config.toml --log-file logs/watch_$(date +%Y%m%d).log &
echo $! > logs/watcher.pid
```

## Stop the Watcher

```bash
# Using the skill's script
~/.claude/skills/Mneme/Tools/manage-watcher.sh stop

# Or manually:
pkill -f "mneme watch"
```

## Restart the Watcher

```bash
# Using the skill's script
~/.claude/skills/Mneme/Tools/manage-watcher.sh restart

# Or manually:
pkill -f "mneme watch"
sleep 2
~/.claude/skills/Mneme/Tools/manage-watcher.sh start
```

## Health Check

```bash
# Full health check
~/.claude/skills/Mneme/Tools/manage-watcher.sh health

# Manual checks:
# 1. Check if process is running
pgrep -f "mneme watch" > /dev/null || echo "ERROR: Watcher not running"

# 2. Check if it's actually indexing (log activity in last 5 minutes)
find logs/ -name "watch_*.log" -mmin -5 | grep -q . && echo "Recent activity" || echo "No recent log activity"

# 3. Check for errors in logs
tail -100 logs/watch_$(date +%Y%m%d).log | grep -i error && echo "Errors found" || echo "No recent errors"
```

## When to Restart the Watcher

Restart the watcher when:
- Config changes have been made
- Code has been updated
- Watcher stops responding
- After system reboot
- Log errors indicate issues

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | `config.toml` | Config file path |
| `LOG_DIR` | `logs` | Log output directory |
| `MNEME_HOME` | `~/.mneme` | Mneme installation directory |

Example with custom config:
```bash
CONFIG_FILE=production.toml ~/.claude/skills/Mneme/Tools/manage-watcher.sh start
```
