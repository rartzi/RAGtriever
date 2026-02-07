# Development Workflow

## Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes to code

# 3. Reinstall after code changes
pip install -e ".[dev]"

# 4. Run linting
ruff check src/ tests/

# 5. Run tests
pytest
```

## Testing Image Analysis

```bash
# Test with small vault first (1-3 images)
mneme scan --config config.toml --full

# Check logs for errors
# tail -100 scan.log

# Query for image content
mneme query --config config.toml "image description" --k 5

# Verify results include image metadata
```

## Test Scan & Watch

```bash
# Run full scan with logging
mkdir -p logs
mneme scan --config config.toml --full --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log

# Start watcher with logging
~/.claude/skills/Mneme/Tools/manage-watcher.sh start

# Check watcher health
~/.claude/skills/Mneme/Tools/manage-watcher.sh health

# Stop watcher
~/.claude/skills/Mneme/Tools/manage-watcher.sh stop
```

## Key Files Reference

### Code
- `src/mneme/extractors/image.py` - Image extractor implementations
- `src/mneme/config.py` - Configuration management (includes [logging] support)
- `src/mneme/cli.py` - CLI commands (scan, watch, query with logging/profiling)
- `src/mneme/indexer/indexer.py` - Main indexer orchestration
- `src/mneme/retrieval/retriever.py` - Hybrid search implementation

### Configuration
- `examples/config.toml.example` - Configuration template with [logging] section
- `config.toml` - User configuration

### Documentation
- `docs/ARCHITECTURE.md` - Complete system architecture
- `docs/LOGGING_CONFIGURATION.md` - Logging setup and usage guide
- `docs/WHERE_TO_SEE_INDEXING.md` - How to verify indexing success
- `docs/SCAN_AND_WATCH_TESTING.md` - Testing guide with profiling
- `docs/gemini_service_account_setup.md` - Gemini service account setup guide
- `docs/troubleshooting.md` - Detailed troubleshooting
- `IMPROVEMENTS.md` - Planned enhancements

### Skill Tools
- `skills/Mneme/Tools/mneme-wrapper.sh` - Auto-installing CLI wrapper
- `skills/Mneme/Tools/manage-watcher.sh` - Watcher management (start/stop/restart/health)

## Additional Resources

- **README.md**: User guide and quick start
- **CLAUDE.md**: Project architecture for Claude Code
- **docs/architecture.md**: Complete system documentation
- **docs/gemini_service_account_setup.md**: Gemini service account service account setup
- **docs/troubleshooting.md**: Comprehensive troubleshooting guide
- **IMPROVEMENTS.md**: Planned features

## Notes

- Mneme is a standalone tool, not a Claude Code skill itself
- This skill provides workflow assistance for using Mneme
- Mneme runs independently and can work without Claude
- Optional MCP integration enables Claude Desktop to search your vault
