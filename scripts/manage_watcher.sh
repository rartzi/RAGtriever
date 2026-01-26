#!/bin/bash
# Watcher management script for RAGtriever
#
# Usage:
#   ./scripts/manage_watcher.sh status   # Check if running
#   ./scripts/manage_watcher.sh start    # Start watcher
#   ./scripts/manage_watcher.sh stop     # Stop watcher
#   ./scripts/manage_watcher.sh restart  # Restart watcher
#   ./scripts/manage_watcher.sh health   # Health check

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Configuration
CONFIG_FILE="${CONFIG_FILE:-config.toml}"
PID_FILE="logs/watcher.pid"
LOG_DIR="logs"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Check dependencies
check_dependencies() {
    local errors=0

    # Check if we're in the right directory
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}✗${NC} Config file not found: $CONFIG_FILE"
        echo "  Current directory: $(pwd)"
        errors=$((errors + 1))
    fi

    # Check if venv exists
    if [ ! -f ".venv/bin/activate" ] && [ ! -f "venv/bin/activate" ]; then
        echo -e "${RED}✗${NC} Virtual environment not found (.venv/ or venv/)"
        echo "  Run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
        errors=$((errors + 1))
    fi

    # Check if mneme is installed (after activating venv)
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi

    if ! command -v mneme &> /dev/null; then
        echo -e "${RED}✗${NC} mneme command not found"
        echo "  Run: pip install -e ."
        errors=$((errors + 1))
    fi

    return $errors
}

# Check if watcher is running
is_running() {
    pgrep -f "mneme watch" > /dev/null
}

# Get PID if running
get_pid() {
    pgrep -f "mneme watch"
}

# Status command
cmd_status() {
    if is_running; then
        PID=$(get_pid)
        echo -e "${GREEN}✓${NC} Watcher is running (PID: $PID)"
        ps aux | grep "[r]agtriever watch"
        return 0
    else
        echo -e "${RED}✗${NC} Watcher is not running"
        return 1
    fi
}

# Start command
cmd_start() {
    # Check dependencies first
    echo "Checking dependencies..."
    if ! check_dependencies; then
        echo ""
        echo -e "${RED}✗${NC} Dependency check failed - cannot start watcher"
        return 1
    fi
    echo -e "${GREEN}✓${NC} All dependencies OK"
    echo ""

    if is_running; then
        echo -e "${YELLOW}⚠${NC} Watcher is already running"
        cmd_status
        return 0
    fi

    echo "Starting watcher..."

    # Activate venv (already checked in dependencies)
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi

    # Set offline mode environment variables (for corporate proxies)
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    # Start watcher in background using /bin/bash explicitly
    /bin/bash -c "source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; nohup mneme watch --config '$CONFIG_FILE' > /dev/null 2>&1 &
    echo \$!" > "$PID_FILE"

    # Get PID from file
    if [ -f "$PID_FILE" ]; then
        WATCHER_PID=$(cat "$PID_FILE")
    else
        echo -e "${RED}✗${NC} Failed to save PID"
        return 1
    fi

    # Wait a moment for startup
    sleep 3

    # Verify it started
    if is_running; then
        ACTUAL_PID=$(get_pid)
        echo -e "${GREEN}✓${NC} Watcher started successfully (PID: $ACTUAL_PID)"

        # Check for log file
        TODAY=$(date +%Y%m%d)
        if [ -f "$LOG_DIR/watch_$TODAY.log" ]; then
            echo "Log file: $LOG_DIR/watch_$TODAY.log"
        else
            echo "Log file will be created: $LOG_DIR/watch_$TODAY.log"
        fi

        # Show recent log lines if available
        if [ -f "$LOG_DIR/watch_$TODAY.log" ]; then
            echo ""
            echo "Recent log:"
            tail -5 "$LOG_DIR/watch_$TODAY.log"
        fi
    else
        echo -e "${RED}✗${NC} Failed to start watcher"
        echo "Check if mneme is installed: pip install -e ."
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
    pkill -f "mneme watch"

    # Wait for graceful shutdown
    for i in {1..5}; do
        if ! is_running; then
            break
        fi
        sleep 1
    done

    # Force kill if still running
    if is_running; then
        echo "Force killing watcher..."
        pkill -9 -f "mneme watch"
        sleep 1
    fi

    if ! is_running; then
        echo -e "${GREEN}✓${NC} Watcher stopped"
        rm -f "$PID_FILE"
    else
        echo -e "${RED}✗${NC} Failed to stop watcher"
        return 1
    fi
}

# Restart command
cmd_restart() {
    echo "Restarting watcher..."
    cmd_stop
    sleep 2
    cmd_start
}

# Health check command
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

    # 3. Recent activity check (last 5 minutes)
    echo -n "3. Recent activity: "
    if find "$LOG_DIR" -name "watch_*.log" -mmin -5 | grep -q .; then
        echo -e "${GREEN}✓${NC} (log modified in last 5 min)"
    else
        echo -e "${YELLOW}⚠${NC} (no recent log activity)"
    fi

    # 4. Error check (last 100 lines)
    echo -n "4. Recent errors: "
    if [ -f "$LOG_FILE" ]; then
        ERROR_COUNT=$(tail -100 "$LOG_FILE" | grep -c "ERROR" || true)
        if [ "$ERROR_COUNT" -eq 0 ]; then
            echo -e "${GREEN}✓${NC} (none)"
        else
            echo -e "${YELLOW}⚠${NC} ($ERROR_COUNT errors in last 100 lines)"
            echo ""
            echo "Recent errors:"
            tail -100 "$LOG_FILE" | grep "ERROR" | tail -3
        fi
    else
        echo -e "${YELLOW}⚠${NC} (no log file to check)"
    fi

    echo ""
    echo "=== Summary ==="
    if is_running && [ -f "$LOG_FILE" ]; then
        echo -e "${GREEN}✓${NC} Watcher is healthy"
    else
        echo -e "${YELLOW}⚠${NC} Watcher may have issues"
    fi
}

# Main
case "${1:-status}" in
    status)
        cmd_status
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    health)
        cmd_health
        ;;
    check)
        check_dependencies
        ;;
    help|--help|-h)
        echo "RAGtriever Watcher Management Script"
        echo ""
        echo "Usage: $0 {status|start|stop|restart|health|check}"
        echo ""
        echo "Commands:"
        echo "  status   - Check if watcher is running"
        echo "  start    - Start the watcher (with dependency checks)"
        echo "  stop     - Stop the watcher gracefully"
        echo "  restart  - Restart the watcher"
        echo "  health   - Run comprehensive health check"
        echo "  check    - Check dependencies only"
        echo ""
        echo "Environment:"
        echo "  CONFIG_FILE - Config file path (default: config.toml)"
        echo "  Example: CONFIG_FILE=my_config.toml $0 start"
        echo ""
        echo "Output files:"
        echo "  logs/watcher.pid        - Watcher process ID"
        echo "  logs/watch_YYYYMMDD.log - Daily log file"
        exit 0
        ;;
    *)
        echo -e "${RED}Error:${NC} Unknown command: $1"
        echo ""
        echo "Usage: $0 {status|start|stop|restart|health|check}"
        echo "Run '$0 help' for more information"
        exit 1
        ;;
esac
