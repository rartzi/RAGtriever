#!/bin/bash
# Mneme CLI Wrapper - Portable, auto-installing wrapper for mneme
#
# This wrapper finds or installs mneme automatically:
#   1. Checks if mneme is in PATH (pip installed globally)
#   2. Checks ~/.mneme/venv/ (user-wide installation)
#   3. Checks ./bin/mneme (project-local installation)
#   4. Auto-installs to ~/.mneme/ if not found
#
# Usage: ./Tools/mneme-wrapper.sh <command> [options]
#   ./Tools/mneme-wrapper.sh query "search term" --k 10
#   ./Tools/mneme-wrapper.sh scan --config config.toml --full
#
# Environment Variables:
#   MNEME_HOME      - Override installation directory (default: ~/.mneme)
#   MNEME_REPO      - Override git repository URL
#   MNEME_BRANCH    - Override git branch (default: main)
#   MNEME_AUTO_INSTALL - Set to "0" to disable auto-install prompt

set -e

# Colors (only if terminal supports it)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN=''
    RED=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Configuration
MNEME_HOME="${MNEME_HOME:-$HOME/.mneme}"
MNEME_REPO="${MNEME_REPO:-https://github.com/rartzi/RAGtriever.git}"
MNEME_BRANCH="${MNEME_BRANCH:-main}"
MNEME_VENV="$MNEME_HOME/venv"
MNEME_SOURCE="$MNEME_HOME/source"

# Find suitable Python (3.11+)
find_python() {
    for cmd in python3.13 python3.12 python3.11 python3; do
        if command -v "$cmd" &> /dev/null; then
            local py_version
            py_version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            local py_major=$(echo "$py_version" | cut -d. -f1)
            local py_minor=$(echo "$py_version" | cut -d. -f2)
            if [ "$py_major" -ge 3 ] && [ "$py_minor" -ge 11 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# Check if mneme is available in a location
find_mneme() {
    # 1. Check PATH (pip installed globally or in active venv)
    if command -v mneme &> /dev/null; then
        echo "$(command -v mneme)"
        return 0
    fi

    # 2. Check user-wide installation
    if [ -x "$MNEME_VENV/bin/mneme" ]; then
        echo "$MNEME_VENV/bin/mneme"
        return 0
    fi

    # 3. Check project-local installation (relative to where script is called from)
    if [ -x "./bin/mneme" ]; then
        echo "./bin/mneme"
        return 0
    fi

    # 4. Check if we're inside a RAGtriever project (script location)
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_bin="$script_dir/../../../bin/mneme"
    if [ -x "$project_bin" ]; then
        echo "$project_bin"
        return 0
    fi

    return 1
}

# Install mneme to ~/.mneme/
install_mneme() {
    echo -e "${BLUE}Installing mneme to $MNEME_HOME${NC}"
    echo ""

    # Check Python
    local python_cmd
    python_cmd=$(find_python) || {
        echo -e "${RED}Error:${NC} Python 3.11+ not found"
        echo "  Install Python 3.11+:"
        echo "    macOS:  brew install python@3.11"
        echo "    Ubuntu: sudo apt install python3.11"
        exit 1
    }
    local py_version=$("$python_cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "  Python: $python_cmd ($py_version)"

    # Check git
    if ! command -v git &> /dev/null; then
        echo -e "${RED}Error:${NC} git not found"
        echo "  Install git and try again"
        exit 1
    fi

    # Create directory
    mkdir -p "$MNEME_HOME"

    # Clone or update source
    if [ -d "$MNEME_SOURCE/.git" ]; then
        echo "  Updating source from $MNEME_BRANCH..."
        cd "$MNEME_SOURCE"
        git fetch origin
        git checkout "$MNEME_BRANCH"
        git pull origin "$MNEME_BRANCH"
        cd - > /dev/null
    else
        echo "  Cloning from $MNEME_REPO ($MNEME_BRANCH)..."
        rm -rf "$MNEME_SOURCE"
        git clone --branch "$MNEME_BRANCH" --depth 1 "$MNEME_REPO" "$MNEME_SOURCE"
    fi
    echo -e "  ${GREEN}✓${NC} Source ready"

    # Create virtual environment
    if [ ! -f "$MNEME_VENV/bin/activate" ]; then
        echo "  Creating virtual environment..."
        "$python_cmd" -m venv "$MNEME_VENV"
    fi
    echo -e "  ${GREEN}✓${NC} Virtual environment ready"

    # Install mneme
    echo "  Installing dependencies (this may take a minute)..."
    source "$MNEME_VENV/bin/activate"
    pip install --upgrade pip > /dev/null 2>&1
    pip install -e "$MNEME_SOURCE[dev]" > /dev/null 2>&1 || {
        echo -e "${YELLOW}Retrying with verbose output...${NC}"
        pip install -e "$MNEME_SOURCE[dev]"
    }
    echo -e "  ${GREEN}✓${NC} Dependencies installed"

    # Verify installation
    if [ -x "$MNEME_VENV/bin/mneme" ]; then
        echo ""
        echo -e "${GREEN}✓ mneme installed successfully!${NC}"
        echo "  Location: $MNEME_VENV/bin/mneme"
        echo "  Source:   $MNEME_SOURCE"
        echo ""
        echo "  To update later: $0 --update"
        return 0
    else
        echo -e "${RED}Error:${NC} Installation failed"
        exit 1
    fi
}

# Update existing installation
update_mneme() {
    if [ ! -d "$MNEME_SOURCE/.git" ]; then
        echo -e "${RED}Error:${NC} No installation found at $MNEME_HOME"
        echo "  Run without --update to install"
        exit 1
    fi

    echo -e "${BLUE}Updating mneme...${NC}"
    cd "$MNEME_SOURCE"
    git fetch origin
    git checkout "$MNEME_BRANCH"
    git pull origin "$MNEME_BRANCH"
    cd - > /dev/null

    source "$MNEME_VENV/bin/activate"
    pip install -e "$MNEME_SOURCE[dev]" > /dev/null 2>&1

    echo -e "${GREEN}✓ mneme updated!${NC}"
    "$MNEME_VENV/bin/mneme" --version 2>/dev/null || echo "  (version command not available)"
}

# Show help
show_help() {
    echo "Mneme Wrapper - Portable CLI wrapper with auto-install"
    echo ""
    echo "Usage: $0 [wrapper-options] <mneme-command> [options]"
    echo ""
    echo "Wrapper Options:"
    echo "  --install     Force (re)install mneme to ~/.mneme/"
    echo "  --update      Update existing installation"
    echo "  --where       Show mneme location"
    echo "  --help        Show this help"
    echo ""
    echo "Mneme Commands:"
    echo "  scan          Index vault files"
    echo "  query         Search the index"
    echo "  watch         Watch for file changes"
    echo "  mcp           Start MCP server"
    echo "  status        Show index status"
    echo ""
    echo "Examples:"
    echo "  $0 query \"AI agents\" --k 10"
    echo "  $0 scan --config config.toml --full"
    echo "  $0 --install"
    echo ""
    echo "Environment:"
    echo "  MNEME_HOME=$MNEME_HOME"
    echo "  MNEME_REPO=$MNEME_REPO"
    echo "  MNEME_BRANCH=$MNEME_BRANCH"
}

# Main
case "${1:-}" in
    --install)
        install_mneme
        exit 0
        ;;
    --update)
        update_mneme
        exit 0
        ;;
    --where)
        mneme_path=$(find_mneme) && echo "$mneme_path" || echo "Not found"
        exit 0
        ;;
    --help|-h)
        show_help
        exit 0
        ;;
esac

# Find or install mneme
mneme_path=$(find_mneme) || {
    echo -e "${YELLOW}mneme not found.${NC}"
    echo ""
    echo "Installation locations checked:"
    echo "  1. PATH (global pip install)"
    echo "  2. $MNEME_VENV/bin/mneme (user-wide)"
    echo "  3. ./bin/mneme (project-local)"
    echo ""

    if [ "${MNEME_AUTO_INSTALL:-1}" = "0" ]; then
        echo "Auto-install disabled. Set MNEME_AUTO_INSTALL=1 or run:"
        echo "  $0 --install"
        exit 1
    fi

    # Auto-install: prompt if interactive, auto-proceed if not
    if [ -t 0 ]; then
        # Interactive terminal - ask user
        read -p "Install mneme to $MNEME_HOME? [Y/n] " answer
        if [[ "$answer" =~ ^[Nn] ]]; then
            echo "Aborted. Install manually with: pip install mneme"
            exit 1
        fi
    else
        # Non-interactive (e.g., from Claude Code) - auto-install
        echo "Non-interactive mode: auto-installing to $MNEME_HOME"
    fi

    install_mneme
    mneme_path="$MNEME_VENV/bin/mneme"
}

# Run mneme
exec "$mneme_path" "$@"
