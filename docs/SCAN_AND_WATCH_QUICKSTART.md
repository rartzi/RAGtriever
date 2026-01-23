# Quick Start: Test Scan & Watch

## Option 1: Automated (Recommended)

```bash
./scripts/test_scan_and_watch.sh
```

This handles everything automatically. The script can be run from anywhere in the project.

## Option 2: Manual Commands

### Clean Database
```bash
rm -f ~/.ragtriever/indexes/test_vault/vaultrag.sqlite
```

### Run Profiled Scan (10 workers)
```bash
mkdir -p logs

ragtriever scan \
    --full \
    --workers 10 \
    --log-file logs/scan.log \
    --profile logs/profile.txt \
    --verbose
```

### View Results
```bash
# View profile summary
head -n 60 logs/profile.txt

# View scan log
tail -n 50 logs/scan.log

# Grep for scan phases
grep '\[scan\]' logs/scan.log
```

### Test Watcher
```bash
# Terminal 1: Start watcher
ragtriever watch --log-file logs/watch.log --verbose

# Terminal 2: Watch logs
tail -f logs/watch.log

# Terminal 3: Make changes in vault
cd /Users/kjzc236/workrelated/odsp/innovation_group/obsidian_vaults/test_vault_comprehensive
echo "# Test" > test.md
echo "Updated" >> test.md
rm test.md
```

## New CLI Flags

**Scan command:**
- `--log-file` / `-l` : Write logs to file
- `--profile` / `-p` : Generate profiling report
- `--verbose` / `-v` : DEBUG level logging
- `--log-level` : Set log level (DEBUG/INFO/WARNING/ERROR)

**Watch command (existing):**
- `--log-file` / `-l` : Write logs to file
- `--verbose` / `-v` : DEBUG level logging
- `--log-level` : Set log level

## What to Look For

**Profiling Report:**
- Top time-consuming functions
- Extraction vs embedding vs storage time
- Parallel efficiency

**Scan Log:**
- `[scan] Found N files to process`
- `[scan] Phase 1: N files extracted, M failed`
- `[scan] Phase 2: N chunks, M embeddings`
- `[scan] Phase 3: N images processed`
- `[scan] Complete: N files indexed in Xs`

**Watch Log:**
- `[watch] File created: path`
- `[watch] File modified: path`
- `[watch] File deleted: path`
- `[watch] Batch processed: N files, M chunks`
