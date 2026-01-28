<p align="center">
  <img src="assets/hero.jpg" alt="Mneme - Memory for Your Second Brain" width="800"/>
</p>

<h1 align="center">Mneme</h1>

<p align="center">
  <strong>Memory for your Second Brain</strong><br/>
  <em>(pronounced NEE-mee, after the Greek Muse of Memory)</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platform"/>
</p>

---

Mneme is a **Claude Code skill** that indexes your Obsidian-compatible vault into a powerful hybrid retrieval system combining **semantic search**, **lexical search (FTS5)**, and **link-graph awareness**. All data stays local on your machine.

## Features

- **Hybrid Retrieval** - Combines vector embeddings with full-text search
- **RRF Fusion** - Reciprocal Rank Fusion for robust result merging
- **Backlink & Recency Boost** - Hub documents and fresh content rank higher
- **Obsidian-Aware** - Understands YAML frontmatter, `[[wikilinks]]`, `![[embeds]]`, and `#tags`
- **Multi-Format Support** - Markdown, PDF, PPTX, XLSX, and images
- **AI Image Analysis** - Tesseract OCR, Gemini Vision, or Vertex AI
- **Watch Mode** - Continuously index changes as you edit
- **Self-Contained Skill** - Auto-installs, works offline after setup
- **100% Local** - Your data never leaves your machine

---

## Installation

### For Claude Code Users (Recommended)

```bash
# Download and install the skill
curl -L https://github.com/rartzi/RAGtriever/releases/latest/download/mneme-skill.zip -o mneme-skill.zip
unzip mneme-skill.zip -d ~/.claude/skills/
chmod +x ~/.claude/skills/Mneme/Tools/*.sh
```

Or clone and symlink:
```bash
git clone https://github.com/rartzi/RAGtriever.git
ln -s $(pwd)/RAGtriever/skills/Mneme ~/.claude/skills/Mneme
```

That's it! First use will auto-install mneme from bundled source.

### For Developers

```bash
git clone https://github.com/rartzi/RAGtriever.git
cd RAGtriever/skills/Mneme/source

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Run tests
pytest
```

---

## Usage

### With Claude Code (Natural Language)

Just talk to Claude:

```
"Search my vault for meeting notes"
"What does my vault say about project planning?"
"Start the watcher"
"Run a full scan"
```

### Manual Commands

```bash
# Create config
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh init --vault ~/vault --index ~/.mneme/indexes/myvault

# Scan
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh scan --config config.toml --full

# Query
~/.claude/skills/Mneme/Tools/mneme-wrapper.sh query "search term" --k 10

# Watcher
~/.claude/skills/Mneme/Tools/manage-watcher.sh start
~/.claude/skills/Mneme/Tools/manage-watcher.sh status
```

---

## Configuration

Create `config.toml` in your project directory:

```toml
[vault]
root = "/path/to/your/obsidian/vault"
ignore = [".git/**", ".obsidian/cache/**", "**/.DS_Store", "**/~$*"]

[index]
dir = "~/.mneme/indexes/myvault"

[embeddings]
provider = "sentence_transformers"
model = "BAAI/bge-small-en-v1.5"
device = "cpu"
offline_mode = true

[image_analysis]
provider = "tesseract"  # or "gemini", "off"
```

See [examples/config.toml.example](skills/Mneme/examples/config.toml.example) for full options.

---

## Documentation

| Document | Description |
|----------|-------------|
| [DEPLOYMENT.md](skills/Mneme/DEPLOYMENT.md) | Installation guide (global, user, project) |
| [SKILL.md](skills/Mneme/SKILL.md) | Skill routing and quick reference |
| [docs/USERGUIDE.md](skills/Mneme/docs/USERGUIDE.md) | Comprehensive user guide |
| [docs/ARCHITECTURE.md](skills/Mneme/docs/ARCHITECTURE.md) | System architecture |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

## Project Structure

```
RAGtriever/
├── skills/Mneme/              # The skill (self-contained)
│   ├── SKILL.md               # Routing + quick reference
│   ├── DEPLOYMENT.md          # Installation guide
│   ├── Tools/                 # CLI wrappers
│   │   ├── mneme-wrapper.sh   # Auto-installing CLI
│   │   └── manage-watcher.sh  # Watcher management
│   ├── Workflows/             # Execution procedures
│   ├── source/                # Bundled source code
│   │   ├── pyproject.toml
│   │   ├── src/mneme/
│   │   └── tests/
│   ├── docs/                  # Documentation
│   └── examples/              # Example configs
├── README.md                  # This file
├── CHANGELOG.md               # Release history
├── CLAUDE.md                  # Claude Code instructions
└── LICENSE                    # MIT License
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
