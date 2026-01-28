#!/bin/bash
# Mneme CLI Wrapper - Portable, auto-installing wrapper for mneme
#
# This wrapper finds or installs mneme automatically:
#   1. Checks if mneme is in PATH (pip installed globally)
#   2. Checks ~/.mneme/venv/ (user-wide installation)
#   3. Checks ./.mneme/venv/ (project-local installation)
#   4. Auto-installs from bundled source or git clone
#
# Installation sources (in priority order):
#   1. Bundled source in skill directory (skills/Mneme/source/)
#   2. Git clone from repository
#
# Usage: ./Tools/mneme-wrapper.sh <command> [options]
#   ./Tools/mneme-wrapper.sh query "search term" --k 10
#   ./Tools/mneme-wrapper.sh scan --config config.toml --full
#
# Environment Variables:
#   MNEME_HOME         - Override installation directory (default: ~/.mneme)
#   MNEME_PROJECT_LOCAL - Set to "1" to install to ./.mneme instead of ~/.mneme
#   MNEME_REPO         - Override git repository URL
#   MNEME_BRANCH       - Override git branch (default: main)
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

# Script location (for finding bundled source)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLED_SOURCE="$SKILL_DIR/source"

# Configuration
MNEME_REPO="${MNEME_REPO:-https://github.com/rartzi/RAGtriever.git}"
MNEME_BRANCH="${MNEME_BRANCH:-main}"

# Determine installation directory
get_install_dir() {
    if [ -n "${MNEME_HOME:-}" ]; then
        echo "$MNEME_HOME"
    elif [ "${MNEME_PROJECT_LOCAL:-0}" = "1" ]; then
        echo "./.mneme"
    else
        echo "$HOME/.mneme"
    fi
}

MNEME_HOME="$(get_install_dir)"
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

    # 2. Check user-wide installation (~/.mneme/venv/)
    if [ -x "$HOME/.mneme/venv/bin/mneme" ]; then
        echo "$HOME/.mneme/venv/bin/mneme"
        return 0
    fi

    # 3. Check project-local installation (./.mneme/venv/)
    if [ -x "./.mneme/venv/bin/mneme" ]; then
        echo "./.mneme/venv/bin/mneme"
        return 0
    fi

    # 4. Check project-local bin (legacy)
    if [ -x "./bin/mneme" ]; then
        echo "./bin/mneme"
        return 0
    fi

    # 5. Check project .venv (development)
    if [ -x "./.venv/bin/mneme" ]; then
        echo "./.venv/bin/mneme"
        return 0
    fi

    return 1
}

# Check if bundled source exists
has_bundled_source() {
    [ -f "$BUNDLED_SOURCE/pyproject.toml" ] && [ -d "$BUNDLED_SOURCE/src/mneme" ]
}

# Install mneme
install_mneme() {
    local install_dir="$1"
    local venv_dir="$install_dir/venv"
    local source_dir="$install_dir/source"

    echo -e "${BLUE}Installing mneme to $install_dir${NC}"
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

    # Create directory
    mkdir -p "$install_dir"

    # Get source: bundled or git clone
    if has_bundled_source; then
        echo -e "  Using bundled source from skill"
        if [ ! -d "$source_dir" ] || [ "$BUNDLED_SOURCE" -nt "$source_dir" ]; then
            rm -rf "$source_dir"
            cp -r "$BUNDLED_SOURCE" "$source_dir"
        fi
        echo -e "  ${GREEN}✓${NC} Source ready (bundled)"
    else
        # Need git for cloning
        if ! command -v git &> /dev/null; then
            echo -e "${RED}Error:${NC} git not found and no bundled source available"
            echo "  Install git and try again, or use a skill package with bundled source"
            exit 1
        fi

        # Clone or update source
        if [ -d "$source_dir/.git" ]; then
            echo "  Updating source from $MNEME_BRANCH..."
            cd "$source_dir"
            git fetch origin
            git checkout "$MNEME_BRANCH"
            git pull origin "$MNEME_BRANCH"
            cd - > /dev/null
        else
            echo "  Cloning from $MNEME_REPO ($MNEME_BRANCH)..."
            rm -rf "$source_dir"
            git clone --branch "$MNEME_BRANCH" --depth 1 "$MNEME_REPO" "$source_dir"
        fi
        echo -e "  ${GREEN}✓${NC} Source ready (git)"
    fi

    # Create virtual environment
    if [ ! -f "$venv_dir/bin/activate" ]; then
        echo "  Creating virtual environment..."
        "$python_cmd" -m venv "$venv_dir"
    fi
    echo -e "  ${GREEN}✓${NC} Virtual environment ready"

    # Install mneme
    echo "  Installing dependencies (this may take a minute)..."
    source "$venv_dir/bin/activate"
    pip install --upgrade pip > /dev/null 2>&1
    pip install -e "$source_dir" > /dev/null 2>&1 || {
        echo -e "${YELLOW}Retrying with verbose output...${NC}"
        pip install -e "$source_dir"
    }
    echo -e "  ${GREEN}✓${NC} Dependencies installed"

    # Verify installation
    if [ -x "$venv_dir/bin/mneme" ]; then
        echo ""
        echo -e "${GREEN}✓ mneme installed successfully!${NC}"
        echo "  Location: $venv_dir/bin/mneme"
        echo "  Source:   $source_dir"
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
    local install_dir="$MNEME_HOME"
    local venv_dir="$install_dir/venv"
    local source_dir="$install_dir/source"

    if [ ! -d "$source_dir" ]; then
        echo -e "${RED}Error:${NC} No installation found at $install_dir"
        echo "  Run without --update to install"
        exit 1
    fi

    echo -e "${BLUE}Updating mneme...${NC}"

    # Update from bundled or git
    if has_bundled_source; then
        echo "  Updating from bundled source..."
        rm -rf "$source_dir"
        cp -r "$BUNDLED_SOURCE" "$source_dir"
        echo -e "  ${GREEN}✓${NC} Source updated (bundled)"
    elif [ -d "$source_dir/.git" ]; then
        cd "$source_dir"
        git fetch origin
        git checkout "$MNEME_BRANCH"
        git pull origin "$MNEME_BRANCH"
        cd - > /dev/null
        echo -e "  ${GREEN}✓${NC} Source updated (git)"
    else
        echo -e "${YELLOW}Warning:${NC} Cannot update - no git repo and no bundled source"
        echo "  Reinstalling from scratch..."
        install_mneme "$install_dir"
        return
    fi

    source "$venv_dir/bin/activate"
    pip install -e "$source_dir" > /dev/null 2>&1

    echo -e "${GREEN}✓ mneme updated!${NC}"
    "$venv_dir/bin/mneme" --version 2>/dev/null || echo "  (version command not available)"
}

# Show help
show_help() {
    echo "Mneme Wrapper - Portable CLI wrapper with auto-install"
    echo ""
    echo "Usage: $0 [wrapper-options] <mneme-command> [options]"
    echo ""
    echo "Wrapper Options:"
    echo "  --install         Install mneme (default: ~/.mneme/)"
    echo "  --install-local   Install mneme to ./.mneme/ (project-local)"
    echo "  --update          Update existing installation"
    echo "  --where           Show mneme location"
    echo "  --help            Show this help"
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
    echo "  $0 --install-local"
    echo ""
    echo "Installation Sources:"
    if has_bundled_source; then
        echo "  ${GREEN}✓${NC} Bundled source available at $BUNDLED_SOURCE"
    else
        echo "  • No bundled source - will clone from $MNEME_REPO"
    fi
    echo ""
    echo "Environment:"
    echo "  MNEME_HOME=$MNEME_HOME"
    echo "  MNEME_PROJECT_LOCAL=${MNEME_PROJECT_LOCAL:-0}"
    echo "  MNEME_REPO=$MNEME_REPO"
    echo "  MNEME_BRANCH=$MNEME_BRANCH"
}

# Main
case "${1:-}" in
    --install)
        install_mneme "$HOME/.mneme"
        exit 0
        ;;
    --install-local)
        install_mneme "./.mneme"
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
    echo "  2. ~/.mneme/venv/bin/mneme (user-wide)"
    echo "  3. ./.mneme/venv/bin/mneme (project-local)"
    echo "  4. ./bin/mneme (legacy)"
    echo "  5. ./.venv/bin/mneme (development)"
    echo ""

    if has_bundled_source; then
        echo -e "Bundled source: ${GREEN}available${NC}"
    else
        echo "Bundled source: not available (will clone from git)"
    fi
    echo ""

    if [ "${MNEME_AUTO_INSTALL:-1}" = "0" ]; then
        echo "Auto-install disabled. Set MNEME_AUTO_INSTALL=1 or run:"
        echo "  $0 --install"
        exit 1
    fi

    # Determine install location
    local install_dir="$HOME/.mneme"

    # Auto-install: prompt if interactive, auto-proceed if not
    if [ -t 0 ]; then
        # Interactive terminal - ask user
        echo "Install options:"
        echo "  1) ~/.mneme/ (user-wide, shared across projects)"
        echo "  2) ./.mneme/ (project-local)"
        echo "  3) Cancel"
        read -p "Choose [1]: " choice
        case "${choice:-1}" in
            1) install_dir="$HOME/.mneme" ;;
            2) install_dir="./.mneme" ;;
            *) echo "Aborted."; exit 1 ;;
        esac
    else
        # Non-interactive (e.g., from Claude Code) - auto-install user-wide
        echo "Non-interactive mode: auto-installing to $install_dir"
    fi

    install_mneme "$install_dir"

    # Update paths after install
    if [ "$install_dir" = "$HOME/.mneme" ]; then
        mneme_path="$HOME/.mneme/venv/bin/mneme"
    else
        mneme_path="./.mneme/venv/bin/mneme"
    fi
}

# Run mneme
exec "$mneme_path" "$@"
