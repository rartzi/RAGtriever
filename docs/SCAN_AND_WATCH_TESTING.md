# RAGtriever Testing Guide

## Quick Start: Automated Testing

Run the automated test script:

```bash
./scripts/test_scan_and_watch.sh
```

This script will:
1. Clean the database (delete existing index)
2. Run a full scan with 10 workers
3. Generate profiling report
4. Create detailed scan logs
5. Prompt you to test the watcher with logging

All logs and profiles are saved to the `logs/` directory with timestamps.

## Manual Commands

### 1. Clean Database

```bash
# Get index directory from config
INDEX_DIR=$(grep -A 1 '^\[index\]' config.toml | grep '^dir' | cut -d '"' -f 2)
INDEX_DIR="${INDEX_DIR/#\~/$HOME}"

# Delete database
rm -f "$INDEX_DIR/vaultrag.sqlite"
rm -f "$INDEX_DIR/faiss.index"

echo "Database cleaned: $INDEX_DIR"
```

### 2. Run Full Scan with Profiling and Logging

```bash
# Create logs directory
mkdir -p logs

# Run scan
ragtriever scan \
    --config config.toml \
    --full \
    --workers 10 \
    --log-file logs/scan.log \
    --verbose \
    --profile logs/scan_profile.txt
```

**New CLI flags added:**
- `--log-file` / `-l`: Write logs to file for audit trail
- `--verbose` / `-v`: Enable DEBUG level logging
- `--log-level`: Set log level (DEBUG, INFO, WARNING, ERROR)
- `--profile` / `-p`: Generate cProfile profiling report

### 3. View Profiling Report

```bash
# View top functions by cumulative time
head -n 60 logs/scan_profile.txt

# View full report
less logs/scan_profile.txt
```

The profiling report shows:
- Top 50 functions by cumulative time
- Top 20 functions with their callers
- Useful for identifying bottlenecks

### 4. View Scan Logs

```bash
# View all scan events
grep '\[scan\]' logs/scan.log

# View errors and failures
grep -E 'Failed:|ERROR' logs/scan.log

# View phase summaries
grep 'Phase' logs/scan.log

# Tail logs in real-time (if running scan in background)
tail -f logs/scan.log
```

**Scan log messages:**
- `[scan] Found N files to process`
- `[scan] Deleted: path`
- `[scan] Phase 0: Removed N deleted file(s)`
- `[scan] Failed: path - error`
- `[scan] Phase 1: N files extracted, M failed`
- `[scan] Phase 2: N chunks, M embeddings`
- `[scan] Phase 3: N images processed`
- `[scan] Complete: N files indexed in Xs`

### 5. Test Watcher with Logging

```bash
# Start watcher with logging
ragtriever watch \
    --config config.toml \
    --log-file logs/watch.log \
    --verbose

# In another terminal, tail the log
tail -f logs/watch.log
```

**Watch log messages:**
- `[watch] Starting file watcher on: path`
- `[watch] File created: path`
- `[watch] File modified: path`
- `[watch] File deleted: path`
- `[watch] File moved: old -> new`
- `[watch] Batch processed: N files, M chunks`

### 6. Test Watch Mode

While the watcher is running, test these scenarios in your vault:

```bash
VAULT="/Users/kjzc236/workrelated/odsp/innovation_group/obsidian_vaults/test_vault_comprehensive"

# Test file creation
echo "# Test Note" > "$VAULT/test_create.md"

# Test file modification
echo "Updated content" >> "$VAULT/test_create.md"

# Test file deletion
rm "$VAULT/test_create.md"

# Test file move
echo "# Another Test" > "$VAULT/test_move.md"
mv "$VAULT/test_move.md" "$VAULT/test_moved.md"

# Test directory creation with files
mkdir -p "$VAULT/new_folder"
echo "# File in new folder" > "$VAULT/new_folder/note.md"
```

Check the watch log to see these events being detected and processed.

## Analyzing Results

### Profiling Analysis

Key metrics to look for:
- **Total scan time**: Compare with/without profiling (profiling adds ~5-10% overhead)
- **Extraction time**: Time spent in `_process_file()` and extractors
- **Embedding time**: Time in `embed()` calls
- **Storage time**: Time in SQLite operations
- **Parallel efficiency**: Compare 1 worker vs 10 workers

### Log Analysis

```bash
# Count files processed
grep 'Phase 1:' logs/scan.log

# Check for failures
grep -c 'Failed:' logs/scan.log

# View timing breakdown
grep 'Phase' logs/scan.log

# Watch mode: count events by type
grep '\[watch\]' logs/watch.log | cut -d: -f3 | sort | uniq -c
```

## Troubleshooting

### Database locked errors
If you see "database is locked" errors:
```bash
# Kill any hanging processes
pkill -f ragtriever

# Wait a moment for locks to release
sleep 2

# Clean and retry
rm -f ~/.ragtriever/indexes/test_vault/vaultrag.sqlite-shm
rm -f ~/.ragtriever/indexes/test_vault/vaultrag.sqlite-wal
```

### Missing logs
If logs aren't being created:
```bash
# Check if log directory exists
mkdir -p logs

# Check file permissions
ls -la logs/

# Verify logging is enabled in command
ragtriever scan --help | grep -A 2 log-file
```

### Profiling overhead
Profiling adds overhead. For accurate timing:
1. Run once WITH profiling to identify bottlenecks
2. Run again WITHOUT profiling for true performance numbers

## Performance Baselines

Expected performance on test vault (119 files, 16 images):
- **Sequential scan**: ~180s (estimated)
- **Parallel scan (10 workers)**: ~45-50s (3.6x speedup)
- **Bottlenecks**: Image API calls (Gemini), embedding batching

Watch mode should detect and process file changes within 1-2 seconds.
