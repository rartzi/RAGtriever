# Skill Improvements: CLI Discovery & Watcher Management

## What Was Improved

The RAGtrieval skill has been enhanced with two major improvements to make it more elegant, maintainable, and practical.

---

## 1. CLI Discovery via --help

### Problem
The skill was hardcoding all CLI commands, flags, and options. This:
- Became outdated when features changed
- Required manual skill updates
- Could provide incorrect information

### Solution
Teach the skill to **discover CLI options dynamically** using `--help`.

### What Was Added

**New section: "Discovering CLI Options"**

```bash
# Main help
ragtriever --help

# Command-specific help
ragtriever scan --help
ragtriever watch --help
ragtriever query --help
```

**Example workflow:**
```bash
# User asks: "How do I enable logging?"
# Instead of guessing, run:
ragtriever scan --help | grep -A 2 log

# Output shows current, accurate flags:
#   --log-file TEXT    Log file path for audit trail
#   --verbose, -v      Enable verbose (DEBUG) logging
```

### Benefits
- ✅ Always accurate (reflects current code)
- ✅ Self-documenting (no manual updates needed)
- ✅ Shows all available options
- ✅ Includes default values
- ✅ Documents new features automatically

### Updated Tip
**Tip #4:** "Use --help to discover options" - Run `ragtriever <command> --help` instead of guessing

---

## 2. Watcher Management

### Problem
The skill didn't teach how to:
- Check if watcher is running
- Start/stop the watcher reliably
- Restart the watcher
- Monitor watcher health

### Solution
Added comprehensive **watcher management section** with commands and a helper script.

### What Was Added

#### Manual Commands Section

**Check if running:**
```bash
pgrep -f "ragtriever watch"
```

**Start watcher:**
```bash
source .venv/bin/activate
nohup ragtriever watch --config config.toml &
echo $! > logs/watcher.pid
```

**Stop watcher:**
```bash
pkill -f "ragtriever watch"
```

**Restart watcher:**
```bash
pkill -f "ragtriever watch"
sleep 2
source .venv/bin/activate
nohup ragtriever watch --config config.toml &
```

**Health check:**
```bash
# 1. Process running?
pgrep -f "ragtriever watch" > /dev/null

# 2. Recent log activity?
find logs/ -name "watch_*.log" -mmin -5 | grep -q .

# 3. Any errors?
tail -100 logs/watch_$(date +%Y%m%d).log | grep -i error
```

#### Management Script

**New file:** `scripts/manage_watcher.sh`

**Usage:**
```bash
./scripts/manage_watcher.sh status    # Check if running
./scripts/manage_watcher.sh start     # Start watcher
./scripts/manage_watcher.sh stop      # Stop watcher
./scripts/manage_watcher.sh restart   # Restart watcher
./scripts/manage_watcher.sh health    # Health check
```

**Features:**
- Automatic virtual environment activation
- PID file management (`logs/watcher.pid`)
- Graceful shutdown with force-kill fallback
- Comprehensive health checks (process, logs, errors)
- Color-coded status messages

### Benefits
- ✅ Easy to check watcher status
- ✅ Reliable start/stop/restart
- ✅ Health monitoring built-in
- ✅ Handles edge cases (already running, force kill, etc.)
- ✅ Works with config-driven logging

### Updated Tips

**Tip #5:** "Check if watcher is running" - Use `pgrep -f "ragtriever watch"` before starting/restarting

**Tip #17:** "Restart watcher when needed" - After config changes, code updates, or if it stops responding

**New subsection:** "Watcher Management Workflow" with step-by-step commands

---

## Files Modified/Created

### Modified
1. **skills/RAGtrieval/skill.md**
   - Added "Discovering CLI Options" section (lines 109-142)
   - Added "Managing the Watcher" section (lines 144-260)
   - Updated "Tips for Claude Code Users" (tips #4, #5, #17)
   - Added "Watcher Management Workflow" subsection

### Created
2. **scripts/manage_watcher.sh**
   - New executable script for watcher management
   - 5 commands: status, start, stop, restart, health
   - ~200 lines with error handling

3. **scripts/README.md** (updated)
   - Documented manage_watcher.sh script
   - Added usage examples and features

---

## Example Usage

### Before (Hardcoded)
**User:** "What flags does scan support?"
**Skill:** Lists hardcoded flags (might be outdated)

### After (Dynamic Discovery)
**User:** "What flags does scan support?"
**Skill:** 
```bash
ragtriever scan --help
```
Shows current, accurate flags.

---

### Before (No Watcher Management)
**User:** "Is the watcher running?"
**Skill:** No guidance provided

### After (Watcher Management)
**User:** "Is the watcher running?"
**Skill:**
```bash
./scripts/manage_watcher.sh status
# Or manually:
pgrep -f "ragtriever watch"
```

**User:** "Restart the watcher"
**Skill:**
```bash
./scripts/manage_watcher.sh restart
```

---

## Impact

### Maintainability
- ✅ Skill doesn't need updates when CLI changes
- ✅ --help always provides current information
- ✅ Single source of truth (the code itself)

### Usability
- ✅ Users can discover options themselves
- ✅ Watcher management is straightforward
- ✅ Health checks catch issues early
- ✅ Scripts handle edge cases

### Elegance
- ✅ Self-documenting approach
- ✅ Less hardcoded information
- ✅ More teachable patterns
- ✅ Works with future features automatically

---

## Summary

The skill is now **more elegant and maintainable** by:

1. **Teaching discovery** instead of hardcoding information
2. **Providing watcher management** tools and commands
3. **Making patterns reusable** across commands
4. **Self-documenting** via --help

**Result:** The skill is now more resilient to change and more practical for day-to-day use.
