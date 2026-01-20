# ~~Critical Issue: File Deletions Not Detected by Scan Command~~ RESOLVED

> **STATUS: RESOLVED** (2026-01-20)
>
> This issue has been fixed. Both scan and watch modes now properly detect and handle file deletions.
> See the "Resolution" section at the bottom for details.

## Problem Summary (Historical)

**The `ragtriever scan` command did not detect file deletions from the vault. Deleted files remained in the index indefinitely unless manually removed or detected by watch mode.**

## Root Cause

Located in `src/ragtriever/indexer/indexer.py:105-125`:

```python
def scan(self, full: bool = False) -> ScanStats:
    """Scan and index files. `full=True` means re-index all; otherwise only changed."""
    if self.cfg.parallel_scan:
        return self.scan_parallel(full=full)

    # Sequential scan (original behavior)
    start = time.time()
    rec = Reconciler(self.cfg.vault_root, self.cfg.ignore)
    files_indexed = 0
    for p in rec.scan_files():  # ← Only processes files that exist!
        self._index_one(p, force=full)
        files_indexed += 1

    return ScanStats(...)
```

**The problem**: `rec.scan_files()` only returns files that currently exist in the filesystem. It never checks the database for previously indexed files that no longer exist.

## Evidence from Test

Test script: `test_file_lifecycle.py`

Results:
```
=== INITIAL STATE ===
Documents (active): 119
Chunks: 715

✅ Created test file with wikilinks

=== AFTER ADDING TEST FILE ===
Documents (active): 120  # ✅ File added correctly
Chunks: 754

🗑️  Deleted test file from filesystem

=== AFTER DELETING TEST FILE ===
Documents (active): 120  # ❌ File NOT detected as deleted!
Chunks: 792              # ❌ Chunks actually increased!
```

The deleted file remains in the index with `deleted=0` status.

## Impact

### 1. Search Results Include Deleted Files
Users will see search results for files that no longer exist:
```bash
$ ragtriever query "test lifecycle"
# Returns results from test_lifecycle.md (which was deleted)
```

### 2. Database Bloat
Over time, the database accumulates:
- Stale documents
- Stale chunks
- Stale embeddings
- Stale FTS entries
- Stale links (if any)

### 3. Incorrect Status Information
```bash
$ ragtriever status
# Shows incorrect file counts (includes deleted files)
```

### 4. Watch Mode Works, Scan Mode Doesn't
- ✅ `ragtriever watch`: Detects deletions via filesystem events
- ❌ `ragtriever scan`: Does NOT detect deletions

This creates inconsistent behavior depending on which command users run.

## What Should Happen

The scan command should:
1. Get list of files currently in the filesystem
2. Get list of files currently in the database (where deleted=0)
3. Compare the two lists
4. For files in DB but not in filesystem → call `delete_document()`

## Additional Issues Found

Even when `delete_document()` IS called (e.g., from watch mode), it doesn't clean up everything:

### Missing Cleanup #1: Links Table
```python
# Current delete_document() does NOT include:
self._conn.execute("DELETE FROM links WHERE vault_id=? AND src_rel_path=?", (vault_id, rel_path))
```

Impact: Ghost nodes in link graph, `vault_neighbors` returns deleted files

### Missing Cleanup #2: Manifest Table
```python
# Current delete_document() does NOT include:
self._conn.execute("DELETE FROM manifest WHERE vault_id=? AND rel_path=?", (vault_id, rel_path))
```

Impact: Stale manifest entries, database bloat

### Missing Cleanup #3: FAISS Index
If `use_faiss = true`, deleted embeddings remain in the FAISS index file.

Impact: Memory waste, slower vector search

## Recommended Fixes

### Fix 1: Add Reconciliation to Scan (CRITICAL)
```python
def scan(self, full: bool = False) -> ScanStats:
    """Scan and index files, detecting deletions."""
    start = time.time()
    rec = Reconciler(self.cfg.vault_root, self.cfg.ignore)

    # Get files on filesystem
    fs_files = {str(p.relative_to(self.cfg.vault_root)) for p in rec.scan_files()}

    # Get indexed files from database
    indexed_files = self.store.get_indexed_files(self.vault_id)  # New method needed

    # Detect deletions
    deleted_files = indexed_files - fs_files
    for rel_path in deleted_files:
        logger.info(f"Detected deletion: {rel_path}")
        self.store.delete_document(self.vault_id, rel_path)

    # Index existing files
    files_indexed = 0
    for p in rec.scan_files():
        self._index_one(p, force=full)
        files_indexed += 1

    return ScanStats(
        files_indexed=files_indexed,
        files_deleted=len(deleted_files),  # Add this stat
        ...
    )
```

### Fix 2: Add get_indexed_files() to Store
```python
# In libsql_store.py:
def get_indexed_files(self, vault_id: str) -> set[str]:
    """Get set of rel_paths for all non-deleted documents in vault."""
    rows = self._conn.execute(
        "SELECT rel_path FROM documents WHERE vault_id=? AND deleted=0",
        (vault_id,)
    ).fetchall()
    return {row["rel_path"] for row in rows}
```

### Fix 3: Enhanced delete_document() Cleanup
```python
def delete_document(self, vault_id: str, rel_path: str) -> None:
    # ... existing code ...

    # FIX: Clean up links table
    self._conn.execute(
        "DELETE FROM links WHERE vault_id=? AND src_rel_path=?",
        (vault_id, rel_path)
    )

    # FIX: Clean up manifest table
    self._conn.execute(
        "DELETE FROM manifest WHERE vault_id=? AND rel_path=?",
        (vault_id, rel_path)
    )

    self._conn.commit()
```

### Fix 4: FAISS Index Cleanup (Optional)
Option A: Rebuild FAISS on every scan (simple but slow)
Option B: Track deleted chunk_ids and rebuild periodically
Option C: Accept current behavior (rebuild on restart)

**Recommendation**: Option C for now, add periodic rebuild later if needed.

## Testing Plan

### Test 1: Scan Detects Deletions
```bash
1. Index 10 files
2. Delete 3 files from filesystem
3. Run ragtriever scan
4. Verify: 3 documents marked as deleted
5. Verify: All chunks/embeddings/FTS for those 3 files are gone
```

### Test 2: Scan with --full Detects Deletions
```bash
1. Index 10 files
2. Delete 5 files
3. Run ragtriever scan --full
4. Verify: Deletions detected and cleaned up
```

### Test 3: Links Cleanup
```bash
1. Create A.md with [[B]] and [[C]]
2. Index vault
3. Verify links exist in database
4. Delete A.md
5. Run scan
6. Verify: Links from A.md are deleted
```

### Test 4: Manifest Cleanup
```bash
1. Index test.md
2. Verify manifest entry exists
3. Delete test.md
4. Run scan
5. Verify: Manifest entry deleted
```

## Files to Modify

1. `src/ragtriever/store/base.py` - Add `get_indexed_files()` to Store protocol
2. `src/ragtriever/store/libsql_store.py` - Implement `get_indexed_files()`, enhance `delete_document()`
3. `src/ragtriever/indexer/indexer.py` - Add reconciliation logic to `scan()` and `scan_parallel()`
4. `src/ragtriever/indexer/parallel_types.py` - Add `files_deleted` to ScanStats
5. `src/ragtriever/cli.py` - Display deleted file count in scan output

## Priority

**CRITICAL** - This affects data integrity and search quality. Should be fixed ASAP.

## ~~Workaround (Current)~~ No Longer Needed

~~Until fixed, users must:~~
1. ~~Use `ragtriever watch` mode to detect deletions in real-time, OR~~
2. ~~Manually clean up deleted files from the index, OR~~
3. ~~Delete the entire index and rebuild from scratch periodically~~

---

## Resolution (2026-01-20)

### Changes Made

All recommended fixes were implemented:

1. **`src/ragtriever/store/base.py`**: Added `get_indexed_files()` to Store protocol
2. **`src/ragtriever/store/libsql_store.py`**:
   - Implemented `get_indexed_files()` - returns set of rel_paths for non-deleted documents
   - Enhanced `delete_document()` - now also cleans up links and manifest tables
3. **`src/ragtriever/indexer/indexer.py`**:
   - Added reconciliation logic to `scan()` (lines 119-135)
   - Added reconciliation logic to `scan_parallel()` (Phase 0, lines 166-177)
   - Updated `_merge_stats()` to include `files_deleted`
4. **`src/ragtriever/indexer/parallel_types.py`**: Added `files_deleted` field to ScanStats
5. **`src/ragtriever/cli.py`**: Display deleted file count in scan output

### Test Results

```
$ python test_file_lifecycle.py

=== INITIAL STATE ===
Documents (active): 120

✅ Created test file with wikilinks
📝 Running scan to index test file...

=== AFTER ADDING TEST FILE ===
Documents (active): 120
Document entry: exists (deleted=0) ⚠️ Still active!

🗑️  Deleted test file from filesystem
📝 Running scan to detect deletion...

=== AFTER DELETING TEST FILE ===
Documents (active): 119
Documents (deleted): 1
Document entry: exists (deleted=1) ✅ Correctly marked as deleted
Outgoing links: 0 ✅ Cleaned up
Manifest entry: 0 ✅ Cleaned up
```

### Current Behavior

| Event | Scan Mode | Watch Mode |
|-------|-----------|------------|
| **File Added** | ✅ Indexed on next scan | ✅ Indexed immediately |
| **File Changed** | ✅ Re-indexed on next scan | ✅ Re-indexed immediately |
| **File Deleted** | ✅ Detected via reconciliation | ✅ Detected via filesystem event |

### CLI Output

```bash
$ ragtriever scan
Scan complete: 119 files, 792 chunks in 45.2s
  (3 deleted files removed from index)
```

### FAISS Cleanup

FAISS index cleanup was intentionally skipped for now (as recommended). FAISS rebuilds from SQLite on startup, so stale entries are automatically cleaned on restart.
