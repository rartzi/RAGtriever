# Mneme Skill Deployment Guide

This guide explains how to deploy the Mneme skill to any project for use with Claude Code.

## Overview

The Mneme skill is **self-contained and portable**. It can:
- Auto-install mneme if not found
- Work with existing project-local installations
- Share a single user-wide installation across projects

## Quick Start

### Option 1: Symlink (Recommended for Development)

```bash
# From any project directory
ln -s /path/to/RAGtriever/skills/Mneme ~/.claude/skills/Mneme
```

### Option 2: Copy (For Distribution)

```bash
# Copy entire skill directory
cp -r /path/to/RAGtriever/skills/Mneme ~/.claude/skills/Mneme
```

### Option 3: Git Submodule (For Teams)

```bash
# Add as submodule in your project
git submodule add https://github.com/rartzi/RAGtriever.git vendor/RAGtriever
ln -s vendor/RAGtriever/skills/Mneme .claude/skills/Mneme
```

## Installation Locations

The skill's `mneme-wrapper.sh` searches for mneme in this order:

| Priority | Location | Description |
|----------|----------|-------------|
| 1 | `PATH` | Global pip install (`pip install mneme`) |
| 2 | `~/.mneme/venv/bin/mneme` | User-wide auto-install |
| 3 | `./bin/mneme` | Project-local (RAGtriever dev) |

If not found, the wrapper offers to **auto-install to `~/.mneme/`**.

## Directory Structure

```
~/.mneme/                    # User-wide installation (auto-created)
├── venv/                    # Python virtual environment
│   └── bin/mneme           # The mneme CLI
└── source/                  # Cloned RAGtriever source
    ├── pyproject.toml
    └── src/mneme/

~/.claude/skills/Mneme/      # The skill (symlinked or copied)
├── SKILL.md                 # Main routing file
├── DEPLOYMENT.md            # This file
├── Workflows/               # Execution procedures
├── Tools/                   # Portable scripts
│   ├── mneme-wrapper.sh    # Smart CLI wrapper
│   └── manage-watcher.sh   # Watcher management
└── ...context files
```

## First-Time Setup

### 1. Deploy the Skill

```bash
# Symlink to Claude Code skills directory
ln -s /path/to/RAGtriever/skills/Mneme ~/.claude/skills/Mneme
```

### 2. Create Project Config

In your project directory, create `config.toml`:

```toml
[[vaults]]
name = "my-vault"
root = "/path/to/your/obsidian/vault"
ignore = [".git/**", ".obsidian/cache/**", "**/.DS_Store"]

[index]
dir = "~/.mneme/indexes/my-vault"

[embeddings]
provider = "sentence_transformers"
model = "BAAI/bge-small-en-v1.5"
device = "cpu"
offline_mode = true

[image_analysis]
provider = "tesseract"  # or "gemini", "off"

[logging]
dir = "logs"
level = "INFO"
```

### 3. Run Initial Scan

```bash
# Using the skill's wrapper (auto-installs mneme if needed)
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh scan --config config.toml --full
```

Or via Claude Code - just ask:
> "Run a full scan of my vault"

## Using the Skill

Once deployed, Claude Code automatically uses the skill when you:

| You Say | Skill Does |
|---------|------------|
| "What does my vault say about X?" | Searches vault, returns answer with sources |
| "Run a full scan" | Executes scan with logging |
| "Start the watcher" | Starts continuous indexing |
| "Setup mneme for my vault" | Guides through configuration |

## Manual Commands

### Via Wrapper (Recommended)

```bash
# Query
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh query "search term" --k 10

# Scan
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh scan --config config.toml --full

# Status
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh status
```

### Via Watcher Manager

```bash
# Check status
~/.claude/skills/Mneme/Tools/manage-watcher.sh status

# Start
~/.claude/skills/Mneme/Tools/manage-watcher.sh start

# Stop
~/.claude/skills/Mneme/Tools/manage-watcher.sh stop

# Health check
~/.claude/skills/Mneme/Tools/manage-watcher.sh health
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEME_HOME` | `~/.mneme` | Installation directory |
| `MNEME_REPO` | `https://github.com/rartzi/RAGtriever.git` | Source repository |
| `MNEME_BRANCH` | `main` | Git branch to install |
| `CONFIG_FILE` | `config.toml` | Config file for watcher |
| `LOG_DIR` | `logs` | Log output directory |

## Updating Mneme

```bash
# Update user-wide installation
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --update
```

Or manually:
```bash
cd ~/.mneme/source
git pull origin main
~/.mneme/venv/bin/pip install -e .[dev]
```

## Multi-Project Setup

The same `~/.mneme/` installation works for multiple projects:

```
~/projects/
├── project-a/
│   └── config.toml          # Points to project-a vault
├── project-b/
│   └── config.toml          # Points to project-b vault
└── project-c/
    └── config.toml          # Points to project-c vault

~/.mneme/                     # Shared installation (used by all)
~/.claude/skills/Mneme/       # Shared skill (used by all)
```

Each project just needs its own `config.toml` with vault paths.

## Troubleshooting

### "mneme not found"

The wrapper should auto-install, but if it fails:

```bash
# Manual install
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --install

# Or check where it's looking
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --where
```

### "Config file not found"

```bash
# Create config.toml in your project directory
# See examples/config.toml.example in RAGtriever repo
```

### "Python 3.11+ not found"

```bash
# macOS
brew install python@3.11

# Ubuntu
sudo apt install python3.11
```

### Permission Denied

```bash
# Ensure scripts are executable
chmod +x ~/.claude/skills/Mneme/Tools/*.sh
```

## Offline Installation

For air-gapped environments:

1. On a connected machine:
   ```bash
   # Clone and package
   git clone https://github.com/rartzi/RAGtriever.git
   cd RAGtriever
   pip download -d wheels/ .[dev]
   tar -czf mneme-offline.tar.gz .
   ```

2. Transfer `mneme-offline.tar.gz` to target machine

3. On target machine:
   ```bash
   mkdir -p ~/.mneme/source
   tar -xzf mneme-offline.tar.gz -C ~/.mneme/source
   python3 -m venv ~/.mneme/venv
   ~/.mneme/venv/bin/pip install --no-index --find-links=~/.mneme/source/wheels/ -e ~/.mneme/source[dev]
   ```

## Uninstalling

```bash
# Remove user-wide installation
rm -rf ~/.mneme

# Remove skill
rm -rf ~/.claude/skills/Mneme
```
