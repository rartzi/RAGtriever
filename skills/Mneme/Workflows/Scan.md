# Scan Workflow

Run scanning operations to index vault content.

## Trigger Phrases
- "run scan"
- "full scan"
- "incremental scan"
- "reindex"

## Procedure

### Full Scan (Re-index Everything)

```bash
./bin/mneme scan --config config.toml --full
```

### Incremental Scan (Changed Files Only)

```bash
./bin/mneme scan --config config.toml
```

### Scan with Logging

```bash
./bin/mneme scan --config config.toml --full --log-file logs/scan.log
```

### Scan with Verbose Logging

```bash
./bin/mneme scan --config config.toml --full --log-file logs/scan.log --verbose
```

### Scan with Profiling

```bash
./bin/mneme scan --config config.toml --full --profile logs/profile.txt
```

### Scan with Everything

```bash
./bin/mneme scan --config config.toml --full \
    --log-file logs/scan.log \
    --profile logs/profile.txt \
    --verbose
```

## Parallel Settings

Default is parallel processing. Override with:

```bash
# Explicit workers
./bin/mneme scan --config config.toml --full --workers 8

# Disable parallelization
./bin/mneme scan --config config.toml --full --no-parallel
```

Configure in `config.toml`:
```toml
[indexing]
extraction_workers = 8    # Parallel file extraction
embed_batch_size = 256    # Embedding batch size
image_workers = 8         # Parallel image API workers
parallel_scan = true      # Enable/disable parallel
```

## Verifying Success

**Console output:**
```
Scan complete: 135 files, 963 chunks in 133.0s
```

**Log file:**
```bash
grep '\[scan\] Complete:' logs/scan.log
grep '\[scan\] Phase' logs/scan.log
```

**Database check:**
```bash
./bin/mneme status
```

## Expected Output

**Phase 1:** File extraction
```
[scan] Phase 1: 135 files extracted, 0 failed
```

**Phase 2:** Embedding
```
[scan] Phase 2: 963 chunks, 963 embeddings
```

**Complete:**
```
[scan] Complete: 135 files indexed in 133.0s
```

## Handling Errors

**Rate limit errors (429):** Normal for image APIs. Re-run scan - cached images are skipped.

**Office temp files:** Close Office apps, add `**/~$*` to ignore patterns.

**Network errors:** Enable offline mode if model is cached.

## Automated Testing

```bash
./scripts/test_scan_and_watch.sh
```

This will:
1. Clean database
2. Run full scan with profiling
3. Show summaries
4. Prompt for watcher test
