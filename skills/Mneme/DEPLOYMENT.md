# Mneme Skill Deployment Guide

Deploy the Mneme skill for use with Claude Code. Choose the deployment level that fits your needs.

## Deployment Levels

| Level | Skill Location | Mneme Location | Use Case |
|-------|---------------|----------------|----------|
| **Global** | `~/.claude/skills/Mneme/` | `~/.mneme/` | Single user, multiple projects |
| **User** | `~/.claude/skills/Mneme/` | `~/.mneme/` | Same as global (recommended) |
| **Project** | `./skills/Mneme/` | `./.mneme/` | Isolated project, specific version |

---

## Global / User-Level Deployment (Recommended)

Best for most users. One skill installation shared across all projects.

### Step 1: Install the Skill

**Option A: Download Release (Non-Developers)**
```bash
# Download and extract skill
curl -L https://github.com/rartzi/RAGtriever/releases/latest/download/mneme-skill.zip -o mneme-skill.zip
unzip mneme-skill.zip -d ~/.claude/skills/
chmod +x ~/.claude/skills/Mneme/Tools/*.sh
```

**Option B: Symlink (Developers)**
```bash
# Clone repo and symlink skill
git clone https://github.com/rartzi/RAGtriever.git ~/RAGtriever
ln -s ~/RAGtriever/skills/Mneme ~/.claude/skills/Mneme
```

### Step 2: Install Mneme (Auto or Manual)

**Automatic:** First use triggers auto-install from bundled source.

**Manual:**
```bash
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --install
```

This installs to `~/.mneme/` using bundled source (no git required).

### Step 3: Create Project Config

In each project directory, create `config.toml`:

```toml
[vault]
root = "/path/to/your/obsidian/vault"
ignore = [".git/**", ".obsidian/cache/**", "**/.DS_Store", "**/~$*"]

[index]
dir = "~/.mneme/indexes/my-project"

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

### Step 4: Run Initial Scan

```bash
cd ~/my-project
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh scan --config config.toml --full
```

Or via Claude Code: "Run a full scan of my vault"

### Directory Structure (Global)

```
~/.claude/skills/Mneme/          # Skill (shared)
├── SKILL.md
├── Tools/
│   ├── mneme-wrapper.sh
│   └── manage-watcher.sh
├── Workflows/
├── source/                      # Bundled source (~776KB)
│   ├── pyproject.toml
│   └── src/mneme/
├── docs/                       # Documentation
└── examples/                   # Example configs

~/.mneme/                        # Mneme installation (shared)
├── venv/bin/mneme              # The CLI
└── source/                      # Installed source

~/project-a/config.toml          # Project configs (separate)
~/project-b/config.toml
~/project-c/config.toml
```

---

## Project-Level Deployment

For isolated projects that need their own skill and mneme version.

### Step 1: Add Skill to Project

**Option A: Copy**
```bash
cd ~/my-project
mkdir -p skills
cp -r /path/to/RAGtriever/skills/Mneme ./skills/
chmod +x ./skills/Mneme/Tools/*.sh
```

**Option B: Git Submodule**
```bash
cd ~/my-project
git submodule add https://github.com/rartzi/RAGtriever.git vendor/RAGtriever
ln -s vendor/RAGtriever/skills/Mneme ./skills/Mneme
```

### Step 2: Install Mneme Locally

```bash
./skills/Mneme/Tools/mneme-wrapper.sh --install-local
```

This installs to `./.mneme/` within the project.

### Step 3: Create Config

Create `config.toml` in project root (same format as above).

### Step 4: Run Initial Scan

```bash
./skills/Mneme/Tools/mneme-wrapper.sh scan --config config.toml --full
```

### Directory Structure (Project)

```
~/my-project/
├── config.toml                  # Project config
├── logs/                        # Scan/watch logs
├── skills/
│   └── Mneme/                   # Skill (project-local)
│       ├── SKILL.md
│       ├── Tools/
│       ├── Workflows/
│       └── source/              # Bundled source
└── .mneme/                      # Mneme installation (project-local)
    ├── venv/bin/mneme
    └── source/
```

### Add to .gitignore

```gitignore
# Mneme
.mneme/
logs/
```

---

## Config File Location

**Config is always relative to your current working directory:**

```bash
cd ~/project-a
mneme query "search"              # Uses ~/project-a/config.toml

cd ~/project-b
mneme query "search"              # Uses ~/project-b/config.toml

# Or specify explicitly:
mneme query --config /path/to/config.toml "search"
```

---

## Using the Skill with Claude Code

Once deployed, just talk naturally:

| You Say | Skill Does |
|---------|------------|
| "What does my vault say about X?" | Searches vault, returns answer with sources |
| "Search my vault for meeting notes" | Searches and cites sources |
| "Run a full scan" | Executes scan with logging |
| "Start the watcher" | Starts continuous indexing |
| "Is the watcher running?" | Checks watcher status |
| "Setup mneme for my vault" | Guides through configuration |

---

## Manual Commands Reference

### Via Wrapper

```bash
# Query
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh query "search term" --k 10

# Scan
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh scan --config config.toml --full

# Status
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh status

# Check installation
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --where
```

### Via Watcher Manager

```bash
~/.claude/skills/Mneme/Tools/manage-watcher.sh status
~/.claude/skills/Mneme/Tools/manage-watcher.sh start
~/.claude/skills/Mneme/Tools/manage-watcher.sh stop
~/.claude/skills/Mneme/Tools/manage-watcher.sh health
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEME_HOME` | `~/.mneme` | Override installation directory |
| `MNEME_PROJECT_LOCAL` | `0` | Set to `1` to install to `./.mneme` |
| `MNEME_REPO` | GitHub URL | Override git repository |
| `MNEME_BRANCH` | `main` | Git branch for updates |
| `MNEME_AUTO_INSTALL` | `1` | Set to `0` to disable auto-install |
| `CONFIG_FILE` | `config.toml` | Config file for watcher |

---

## Updating

### Automatic Updates (Default)

**Mneme automatically updates when skill source changes!** No manual action needed.

When you run any mneme command, the wrapper checks if the bundled source in your skill directory is newer than the installed version. If so, it auto-updates transparently before running your command.

**How it works:**
1. Updates skill: `cd ~/RAGtriever && git pull` (if symlinked) or re-download (if copied)
2. Next mneme command auto-detects changes and updates runtime installation
3. See message: `⟳ Detecting updated source - auto-updating...`
4. Takes 1-2 seconds, happens once per update

**To disable auto-updates:**
```bash
export MNEME_AUTO_UPDATE=0  # Add to ~/.bashrc or ~/.zshrc
```

### Manual Update

If you disabled auto-updates or want to force an update:

```bash
# From bundled source (re-copies from skill)
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --update

# From git (if installed from git)
cd ~/.mneme/source && git pull origin main
~/.mneme/venv/bin/pip install -e .
```

### Update Skill Source

```bash
# If symlinked - update the source repo
cd ~/RAGtriever && git pull

# If copied - re-download
curl -L https://github.com/rartzi/RAGtriever/releases/latest/download/mneme-skill.zip -o mneme-skill.zip
unzip -o mneme-skill.zip -d ~/.claude/skills/
```

---

## Troubleshooting

### "mneme not found"

```bash
# Check where wrapper is looking
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --where

# Manual install
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh --install
```

### "Python 3.11+ not found"

```bash
# macOS
brew install python@3.11

# Ubuntu
sudo apt install python3.11
```

### "Config file not found"

Create `config.toml` in your current directory. See examples above.

### "Permission denied"

```bash
chmod +x ~/.claude/skills/Mneme/Tools/*.sh
```

---

## Offline Installation

The skill includes bundled source, so no network is needed after downloading the skill.

For fully air-gapped environments with no pip access:

```bash
# On connected machine
pip download -d wheels/ sentence-transformers torch
tar -czf mneme-wheels.tar.gz wheels/

# Transfer and install
tar -xzf mneme-wheels.tar.gz
~/.mneme/venv/bin/pip install --no-index --find-links=wheels/ sentence-transformers torch
```

---

## Uninstalling

```bash
# Remove mneme installation
rm -rf ~/.mneme      # or ./.mneme for project-local

# Remove skill
rm -rf ~/.claude/skills/Mneme   # or ./skills/Mneme
```
