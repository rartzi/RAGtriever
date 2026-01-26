#!/bin/bash
# Test script for scanning and watching with logging and profiling
#
# Usage: Run from project root directory:
#   ./scripts/test_scan_and_watch.sh
#
# Or from scripts directory:
#   cd scripts && ./test_scan_and_watch.sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${BLUE}Working directory: $PROJECT_ROOT${NC}"
echo ""

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓${NC} Activated virtual environment"
    echo ""
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓${NC} Activated virtual environment"
    echo ""
else
    echo -e "${YELLOW}Warning: No virtual environment found${NC}"
    echo ""
fi

# Configuration
CONFIG_FILE="config.toml"
LOG_DIR="logs"
SCAN_LOG="$LOG_DIR/scan_$(date +%Y%m%d_%H%M%S).log"
WATCH_LOG="$LOG_DIR/watch_$(date +%Y%m%d_%H%M%S).log"
PROFILE_FILE="$LOG_DIR/scan_profile_$(date +%Y%m%d_%H%M%S).txt"

# Get index directory from config
INDEX_DIR=$(grep -A 1 '^\[index\]' "$CONFIG_FILE" | grep '^dir' | cut -d '"' -f 2)
# Expand tilde
INDEX_DIR="${INDEX_DIR/#\~/$HOME}"
DB_FILE="$INDEX_DIR/vaultrag.sqlite"

echo -e "${BLUE}=== RAGtriever Test: Scan & Watch ===${NC}"
echo ""

# Create logs directory
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓${NC} Created log directory: $LOG_DIR"

# Step 1: Clean database
echo ""
echo -e "${YELLOW}Step 1: Cleaning database${NC}"
if [ -f "$DB_FILE" ]; then
    rm "$DB_FILE"
    echo -e "${GREEN}✓${NC} Deleted: $DB_FILE"
else
    echo -e "${YELLOW}!${NC} Database not found (already clean): $DB_FILE"
fi

# Also clean FAISS index if exists
FAISS_INDEX="$INDEX_DIR/faiss.index"
if [ -f "$FAISS_INDEX" ]; then
    rm "$FAISS_INDEX"
    echo -e "${GREEN}✓${NC} Deleted FAISS index: $FAISS_INDEX"
fi

# Step 2: Run full scan with profiling and logging
echo ""
echo -e "${YELLOW}Step 2: Running full scan with 10 workers (profiled)${NC}"
echo -e "  Log file: $SCAN_LOG"
echo -e "  Profile file: $PROFILE_FILE"
echo ""

# Set offline mode environment variables to prevent HuggingFace API calls
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mneme scan \
    --config "$CONFIG_FILE" \
    --full \
    --workers 10 \
    --log-file "$SCAN_LOG" \
    --verbose \
    --profile "$PROFILE_FILE"

echo ""
echo -e "${GREEN}✓${NC} Scan complete!"

# Step 3: Show profiling summary
echo ""
echo -e "${YELLOW}Step 3: Profiling Summary${NC}"
echo ""
echo "Top 10 functions by cumulative time:"
head -n 60 "$PROFILE_FILE" | tail -n 15
echo ""
echo -e "Full profile: ${BLUE}$PROFILE_FILE${NC}"

# Step 4: Show scan log summary
echo ""
echo -e "${YELLOW}Step 4: Scan Log Summary${NC}"
echo ""
echo "Key scan events:"
grep -E '\[scan\]|Failed:|Phase' "$SCAN_LOG" | tail -n 20
echo ""
echo -e "Full log: ${BLUE}$SCAN_LOG${NC}"

# Step 5: Prompt for watcher test
echo ""
echo -e "${YELLOW}Step 5: Watch Mode Test${NC}"
echo ""
echo "Ready to test the watcher. This will:"
echo "  - Start watching the vault for changes"
echo "  - Log all events to: $WATCH_LOG"
echo "  - Run in verbose mode for detailed tracing"
echo ""
echo -e "${BLUE}Press Enter to start the watcher, or Ctrl+C to skip${NC}"
read -r

echo ""
echo "Starting watcher... (Press Ctrl+C to stop)"
echo ""

# Run watcher with logging
mneme watch \
    --config "$CONFIG_FILE" \
    --log-file "$WATCH_LOG" \
    --verbose

# This part only runs if watcher is stopped gracefully
echo ""
echo -e "${GREEN}✓${NC} Watcher stopped"
echo -e "Watch log: ${BLUE}$WATCH_LOG${NC}"
