#!/bin/bash
# Watcher Management Script - Portable version for Mneme skill
#
# This script manages the mneme watcher service using the skill's wrapper.
#
# Usage:
#   ./Tools/manage-watcher.sh status   # Check if running
#   ./Tools/manage-watcher.sh start    # Start watcher
#   ./Tools/manage-watcher.sh stop     # Stop watcher
#   ./Tools/manage-watcher.sh restart  # Restart watcher
#   ./Tools/manage-watcher.sh health   # Health check
#
# Environment Variables:
#   CONFIG_FILE     - Config file path (default: config.toml)
#   LOG_DIR         - Log directory (default: logs)
#   MNEME_HOME      - Mneme installation directory (passed to wrapper)

set -e

# Colors
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    GREEN=''
    RED=''
    YELLOW=''
    NC=''
fi

# Find script directory and wrapper
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNEME_WRAPPER="$SCRIPT_DIR/mneme-wrapper.sh"

# Configuration
CONFIG_FILE="${CONFIG_FILE:-config.toml}"
LOG_DIR="${LOG_DIR:-logs}"
PID_FILE="$LOG_DIR/watcher.pid"

# Ensure wrapper exists
if [ ! -x "$MNEME_WRAPPER" ]; then
    echo -e "${RED}Error:${NC} mneme-wrapper.sh not found at $MNEME_WRAPPER"
    exit 1
fi

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Get mneme command (installs if needed)
get_mneme_cmd() {
    local mneme_path
    mneme_path=$("$MNEME_WRAPPER" --where 2>/dev/null)
    if [ -z "$mneme_path" ] || [ "$mneme_path" = "Not found" ]; then
        echo -e "${YELLOW}mneme not installed. Installing...${NC}"
        "$MNEME_WRAPPER" --install
        mneme_path=$("$MNEME_WRAPPER" --where)
    fi
    echo "$mneme_path"
}

# Check if watcher is running
is_running() {
    pgrep -f "mneme watch" > /dev/null 2>&1
}

# Get PID if running
get_pid() {
    pgrep -f "mneme watch" 2>/dev/null
}

# Status command
cmd_status() {
    if is_running; then
        PID=$(get_pid)
        echo -e "${GREEN}✓${NC} Watcher is running (PID: $PID)"
        return 0
    else
        echo -e "${RED}✗${NC} Watcher is not running"
        return 1
    fi
}

# Start command
cmd_start() {
    # Check config file
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}✗${NC} Config file not found: $CONFIG_FILE"
        echo "  Create config.toml or set CONFIG_FILE environment variable"
        return 1
    fi

    if is_running; then
        echo -e "${YELLOW}⚠${NC} Watcher is already running"
        cmd_status
        return 0
    fi

    echo "Starting watcher..."

    # Get mneme command
    MNEME_CMD=$(get_mneme_cmd)
    echo "  Using: $MNEME_CMD"

    # Log file with daily rotation
    TODAY=$(date +%Y%m%d)
    WATCH_LOG="$LOG_DIR/watch_$TODAY.log"

    # Set offline mode for corporate environments
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    # Start watcher in background
    nohup "$MNEME_CMD" watch --config "$CONFIG_FILE" --log-file "$WATCH_LOG" >> "$WATCH_LOG" 2>&1 &
    echo $! > "$PID_FILE"

    # Wait for startup
    sleep 3

    # Verify it started
    if is_running; then
        ACTUAL_PID=$(get_pid)
        echo -e "${GREEN}✓${NC} Watcher started (PID: $ACTUAL_PID)"
        echo "  Log: $WATCH_LOG"
    else
        echo -e "${RED}✗${NC} Failed to start watcher"
        echo "  Check log: $WATCH_LOG"
        return 1
    fi
}

# Stop command
cmd_stop() {
    if ! is_running; then
        echo -e "${YELLOW}⚠${NC} Watcher is not running"
        return 0
    fi

    echo "Stopping watcher..."
    pkill -f "mneme watch" || true

    # Wait for graceful shutdown
    for i in {1..5}; do
        if ! is_running; then
            break
        fi
        sleep 1
    done

    # Force kill if still running
    if is_running; then
        echo "Force killing..."
        pkill -9 -f "mneme watch" || true
        sleep 1
    fi

    if ! is_running; then
        echo -e "${GREEN}✓${NC} Watcher stopped"
        rm -f "$PID_FILE"
        # Clean up query server socket if present
        if [ -f "$CONFIG_FILE" ]; then
            INDEX_DIR=$(python3 -c "
import tomllib; c=tomllib.load(open('$CONFIG_FILE','rb'))
print(c.get('index',{}).get('dir','~/.mneme/indexes'))
" 2>/dev/null || true)
            if [ -n "$INDEX_DIR" ]; then
                SOCK_PATH="$(eval echo "$INDEX_DIR")/query.sock"
                rm -f "$SOCK_PATH" 2>/dev/null && echo "  Cleaned up query socket"
            fi
        fi
    else
        echo -e "${RED}✗${NC} Failed to stop watcher"
        return 1
    fi
}

# Restart command
cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

# Health check
cmd_health() {
    echo "=== Watcher Health Check ==="
    echo ""

    # 1. Process check
    echo -n "1. Process running: "
    if is_running; then
        PID=$(get_pid)
        echo -e "${GREEN}✓${NC} (PID: $PID)"
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi

    # 2. Log file check
    echo -n "2. Log file exists: "
    TODAY=$(date +%Y%m%d)
    LOG_FILE="$LOG_DIR/watch_$TODAY.log"
    if [ -f "$LOG_FILE" ]; then
        echo -e "${GREEN}✓${NC} ($LOG_FILE)"
    else
        echo -e "${YELLOW}⚠${NC} (not found)"
    fi

    # 3. Recent activity
    echo -n "3. Recent activity: "
    if [ -f "$LOG_FILE" ] && find "$LOG_FILE" -mmin -5 | grep -q .; then
        echo -e "${GREEN}✓${NC} (log modified in last 5 min)"
    else
        echo -e "${YELLOW}⚠${NC} (no recent activity)"
    fi

    # 4. Error check
    echo -n "4. Recent errors: "
    if [ -f "$LOG_FILE" ]; then
        ERROR_COUNT=$(tail -100 "$LOG_FILE" | grep -c "ERROR" || true)
        if [ "$ERROR_COUNT" -eq 0 ]; then
            echo -e "${GREEN}✓${NC} (none)"
        else
            echo -e "${YELLOW}⚠${NC} ($ERROR_COUNT in last 100 lines)"
        fi
    else
        echo -e "${YELLOW}⚠${NC} (no log to check)"
    fi

    # 5. Query server check
    echo -n "5. Query server: "
    if [ -f "$CONFIG_FILE" ]; then
        INDEX_DIR=$(python3 -c "
import tomllib; c=tomllib.load(open('$CONFIG_FILE','rb'))
print(c.get('index',{}).get('dir','~/.mneme/indexes'))
" 2>/dev/null || true)
        if [ -n "$INDEX_DIR" ]; then
            SOCK_PATH="$(eval echo "$INDEX_DIR")/query.sock"
            if [ -S "$SOCK_PATH" ]; then
                echo -e "${GREEN}✓${NC} (socket: $SOCK_PATH)"
            else
                echo -e "${YELLOW}⚠${NC} (no socket found)"
            fi
        else
            echo -e "${YELLOW}⚠${NC} (could not determine index dir)"
        fi
    else
        echo -e "${YELLOW}⚠${NC} (no config file)"
    fi

    echo ""
    echo "=== Summary ==="
    if is_running; then
        echo -e "${GREEN}✓${NC} Watcher is healthy"
    else
        echo -e "${RED}✗${NC} Watcher has issues"
    fi
}

# Install check
cmd_check() {
    echo "Checking mneme installation..."
    MNEME_CMD=$(get_mneme_cmd)
    echo -e "${GREEN}✓${NC} mneme found: $MNEME_CMD"

    if [ -f "$CONFIG_FILE" ]; then
        echo -e "${GREEN}✓${NC} Config found: $CONFIG_FILE"
    else
        echo -e "${RED}✗${NC} Config not found: $CONFIG_FILE"
    fi
}

# Help
cmd_help() {
    echo "Mneme Watcher Management Script"
    echo ""
    echo "Usage: $0 {status|start|stop|restart|health|check|help}"
    echo ""
    echo "Commands:"
    echo "  status   - Check if watcher is running"
    echo "  start    - Start the watcher (installs mneme if needed)"
    echo "  stop     - Stop the watcher gracefully"
    echo "  restart  - Restart the watcher"
    echo "  health   - Run comprehensive health check"
    echo "  check    - Check mneme installation"
    echo "  help     - Show this help"
    echo ""
    echo "Environment:"
    echo "  CONFIG_FILE=$CONFIG_FILE"
    echo "  LOG_DIR=$LOG_DIR"
    echo ""
    echo "Output files:"
    echo "  $LOG_DIR/watcher.pid        - Process ID"
    echo "  $LOG_DIR/watch_YYYYMMDD.log - Daily log"
}

# Main
case "${1:-help}" in
    status)  cmd_status ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    health)  cmd_health ;;
    check)   cmd_check ;;
    help|--help|-h) cmd_help ;;
    *)
        echo -e "${RED}Error:${NC} Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
