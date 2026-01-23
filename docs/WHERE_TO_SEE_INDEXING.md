# Where to See Successful Indexing

## Quick Answer

### Scan Mode
**Log Messages:**
- `[scan] Phase 1: N files extracted, M failed` - Shows files successfully extracted
- `[scan] Phase 2: N chunks, M embeddings` - Shows chunks stored in database
- `[scan] Complete: N files indexed in Xs` - Final summary

**Console Output:**
```bash
Scan complete: 135 files, 963 chunks in 133.0s
```

### Watch Mode
**Log Messages:**
- `[watch] Indexed: filepath` - Each file successfully processed (INFO level)
- `[batch] Stored N docs, M chunks, N embeddings` - Batch written to database (DEBUG level)

**To see indexing confirmations:**
```bash
grep "\[watch\] Indexed:" logs/watch_20260123.log
```

---

## Detailed Guide

### Scan Mode: Where to Look

#### 1. Console Output (Always Visible)
```bash
ragtriever scan --full --workers 10
```

**Output:**
```
Scan complete: 135 files, 963 chunks in 133.0s
  (116 images processed)
```

This tells you:
- ✅ 135 files were indexed successfully
- ✅ 963 chunks created in database
- ✅ 116 images analyzed

#### 2. Log File (if logging enabled)

**Phase-by-phase breakdown:**
```bash
grep '\[scan\]' logs/scan_20260123.log
```

**Output:**
```
2026-01-23 14:20:25 [INFO] ragtriever.indexer.indexer: [scan] Found 142 files to process
2026-01-23 14:21:06 [INFO] ragtriever.indexer.indexer: [scan] Phase 1: 135 files extracted, 0 failed
2026-01-23 14:21:10 [INFO] ragtriever.indexer.indexer: [scan] Phase 2: 963 chunks, 963 embeddings
2026-01-23 14:22:38 [INFO] ragtriever.indexer.indexer: [scan] Phase 3: 116 images processed
2026-01-23 14:22:38 [INFO] ragtriever.indexer.indexer: [scan] Complete: 135 files indexed in 133.0s
```

**What each phase means:**
- **Phase 0**: Reconciliation (deleted files removed)
- **Phase 1**: Extraction (files → chunks)
- **Phase 2**: Embedding (chunks → vectors, stored in database)
- **Phase 3**: Image analysis (images → metadata)

#### 3. Check for Failures
```bash
grep -E 'Failed:|ERROR' logs/scan_20260123.log
```

If no output → All files indexed successfully!

#### 4. Verify in Database
```bash
ragtriever status
```

**Output:**
```
Indexed files: 135
Indexed chunks: 963
```

---

### Watch Mode: Where to Look

#### 1. Individual File Indexing (INFO level)

**Command:**
```bash
grep "\[watch\] Indexed:" logs/watch_20260123.log
```

**Output:**
```
2026-01-23 14:25:20 [INFO] ragtriever.indexer.indexer: [watch] Indexed: 102-KnowledeBases/Navari/Core decks for sharing outside the Navari team/Navari Comms Strategy FINAL 15 JAN for sharing.pdf
2026-01-23 14:25:20 [INFO] ragtriever.indexer.indexer: [watch] Indexed: 102-KnowledeBases/Navari/Decks presented to SET/AC_Navari_Executive_Brief_v2.pptx
2026-01-23 14:25:23 [INFO] ragtriever.indexer.indexer: [watch] Indexed: 102-KnowledeBases/Navari/Comms and messaging decks/Logo files-JPEGs-1x-Logo-100.jpg
```

Each `[watch] Indexed:` line = one file successfully indexed.

#### 2. Batch Database Writes (DEBUG level)

**Command:**
```bash
grep "\[batch\] Stored" logs/watch_20260123.log
```

**Output:**
```
2026-01-23 14:25:27 [DEBUG] ragtriever.indexer.indexer: [batch] Stored 8 docs, 335 chunks, 335 embeddings
```

This confirms files were written to the database.

#### 3. Real-Time Monitoring

**Terminal 1: Run watcher**
```bash
ragtriever watch
```

**Terminal 2: Monitor indexing**
```bash
tail -f logs/watch_20260123.log | grep -E "\[watch\] Indexed:|\[batch\] Stored"
```

**Output (real-time):**
```
2026-01-23 14:30:15 [INFO] ragtriever.indexer.indexer: [watch] Indexed: new_file.md
2026-01-23 14:30:20 [DEBUG] ragtriever.indexer.indexer: [batch] Stored 1 docs, 5 chunks, 5 embeddings
```

#### 4. Count Successfully Indexed Files

**Command:**
```bash
grep -c "\[watch\] Indexed:" logs/watch_20260123.log
```

**Output:**
```
8
```

= 8 files successfully indexed by the watcher.

---

## Log Level Comparison

### INFO Level (Default)
**What you see:**
```
[scan] Found 142 files to process
[scan] Phase 1: 135 files extracted, 0 failed
[scan] Complete: 135 files indexed in 133.0s
[watch] Indexed: file.md
```

**Good for:** Production, normal operations

### DEBUG Level (Verbose)
**What you see:**
```
[scan] Found 142 files to process
[scan] Phase 1: 135 files extracted, 0 failed
[batch] Stored 135 docs, 963 chunks, 963 embeddings  ← Extra detail
[scan] Complete: 135 files indexed in 133.0s
[watch] Indexed: file.md
[batch] Stored 1 docs, 5 chunks, 5 embeddings  ← Extra detail
```

**Good for:** Debugging, troubleshooting

**Enable with:**
```bash
ragtriever scan --full --log-file logs/scan.log --verbose
ragtriever watch --log-file logs/watch.log --verbose
```

---

## Summary Table

| Operation | Log Message | Meaning | Level |
|-----------|-------------|---------|-------|
| **Scan** | `[scan] Phase 1: N files extracted, M failed` | N files successfully extracted | INFO |
| **Scan** | `[scan] Phase 2: N chunks, M embeddings` | Chunks stored in database | INFO |
| **Scan** | `[scan] Complete: N files indexed in Xs` | Final count | INFO |
| **Watch** | `[watch] Indexed: filepath` | File successfully processed | INFO |
| **Watch** | `[batch] Stored N docs, M chunks` | Batch written to database | DEBUG |
| **Both** | `Failed: filepath - error` | File failed to process | INFO |
| **Both** | `[ERROR]` | Something went wrong | ERROR |

---

## Quick Commands Reference

### Check Scan Results
```bash
# View summary
tail -5 logs/scan_20260123.log

# Count successful files
grep '\[scan\] Complete:' logs/scan_20260123.log

# Check for failures
grep 'Failed:' logs/scan_20260123.log
```

### Check Watch Results
```bash
# List all indexed files
grep '\[watch\] Indexed:' logs/watch_20260123.log

# Count indexed files
grep -c '\[watch\] Indexed:' logs/watch_20260123.log

# Check for failures
grep -E 'Failed:|ERROR' logs/watch_20260123.log

# See batch writes
grep '\[batch\] Stored' logs/watch_20260123.log
```

### Real-Time Monitoring
```bash
# Watch indexing as it happens
tail -f logs/watch_20260123.log | grep '\[watch\] Indexed:'

# Watch everything
tail -f logs/watch_20260123.log
```

### Verify in Database
```bash
# Check total indexed
ragtriever status

# Query for specific file
ragtriever query "filename" --k 1
```

---

## Examples

### Example 1: Scan Success
```bash
$ ragtriever scan --full
Scan complete: 135 files, 963 chunks in 133.0s
```

**Verify in log:**
```bash
$ grep '\[scan\] Complete:' logs/scan_20260123.log
2026-01-23 14:22:38 [INFO] ragtriever.indexer.indexer: [scan] Complete: 135 files indexed in 133.0s
```

✅ All 135 files indexed successfully

### Example 2: Watch Success
```bash
$ ragtriever watch
# (running...)
```

**Check log:**
```bash
$ grep '\[watch\] Indexed:' logs/watch_20260123.log
2026-01-23 14:25:20 [INFO] ragtriever.indexer.indexer: [watch] Indexed: file1.pdf
2026-01-23 14:25:21 [INFO] ragtriever.indexer.indexer: [watch] Indexed: file2.md
2026-01-23 14:25:23 [INFO] ragtriever.indexer.indexer: [watch] Indexed: image.jpg
```

✅ 3 files indexed successfully

### Example 3: Partial Failure
```bash
$ ragtriever scan --full
Scan complete: 133 files, 950 chunks in 130.0s
  (2 files failed)
```

**Check what failed:**
```bash
$ grep 'Failed:' logs/scan_20260123.log
2026-01-23 14:21:05 [INFO] ragtriever.indexer.indexer: [scan] Failed: corrupted_file.pdf - error: Invalid PDF structure
2026-01-23 14:21:06 [INFO] ragtriever.indexer.indexer: [scan] Failed: locked_file.docx - error: PermissionError
```

⚠️ 133 files indexed, 2 failed (see log for details)

---

## Troubleshooting

### "I don't see [watch] Indexed: messages"

**Check log level:**
```bash
# These are INFO level messages
grep 'level = ' config.toml
```

Should be `INFO` or `DEBUG`, not `WARNING` or `ERROR`.

**Check if watch logging is enabled:**
```bash
grep 'enable_watch_logging' config.toml
```

Should be `true`.

### "Files detected but not in [watch] Indexed:"

**Check for errors:**
```bash
grep 'ERROR' logs/watch_20260123.log
```

**Check if files were deleted before indexing:**
```bash
grep 'File deleted' logs/watch_20260123.log
```

Files deleted within batch timeout (5s) won't be indexed.

### "Scan says N files but I see fewer in log"

Scan reports **successfully indexed** files. Check for failures:
```bash
grep 'files failed' logs/scan_20260123.log
```
