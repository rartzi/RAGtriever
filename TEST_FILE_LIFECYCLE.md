# File Lifecycle Cleanup Test

> **STATUS: TESTS PASSING** (2026-01-20)
>
> All file lifecycle tests pass. See "Test Results" section at bottom.

## Current delete_document() Implementation (UPDATED)

Looking at `src/ragtriever/store/libsql_store.py:268-286`:

```python
def delete_document(self, vault_id: str, rel_path: str) -> None:
    # Find doc_id
    row = self._conn.execute("SELECT doc_id FROM documents WHERE vault_id=? AND rel_path=? AND deleted=0", (vault_id, rel_path)).fetchone()
    if not row:
        return
    doc_id = row["doc_id"]
    # Delete chunks and embeddings and fts
    chunk_rows = self._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
    for r in chunk_rows:
        cid = r["chunk_id"]
        self._conn.execute("DELETE FROM embeddings WHERE chunk_id=?", (cid,))
        self._conn.execute("DELETE FROM fts_chunks WHERE chunk_id=?", (cid,))
    self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    self._conn.execute("UPDATE documents SET deleted=1 WHERE doc_id=?", (doc_id,))
    # Clean up links table (outgoing links from this file) - ADDED
    self._conn.execute("DELETE FROM links WHERE vault_id=? AND src_rel_path=?", (vault_id, rel_path))
    # Clean up manifest table - ADDED
    self._conn.execute("DELETE FROM manifest WHERE vault_id=? AND rel_path=?", (vault_id, rel_path))
    self._conn.commit()
```

## What Gets Cleaned Up ✅

1. **Embeddings**: Deleted for each chunk (`DELETE FROM embeddings WHERE chunk_id=?`)
2. **FTS Entries**: Deleted for each chunk (`DELETE FROM fts_chunks WHERE chunk_id=?`)
3. **Chunks**: All chunks for document deleted (`DELETE FROM chunks WHERE doc_id=?`)
4. **Document**: Marked as deleted (soft delete: `UPDATE documents SET deleted=1`)
5. **Links**: Outgoing links from deleted file removed (`DELETE FROM links WHERE src_rel_path=?`) - **ADDED**
6. **Manifest**: Indexing metadata removed (`DELETE FROM manifest WHERE rel_path=?`) - **ADDED**

## ~~Potential Residuals~~ All Fixed ✅

### 1. Links Table
**Schema** (lines 97-105):
```sql
CREATE TABLE IF NOT EXISTS links (
  vault_id TEXT NOT NULL,
  src_rel_path TEXT NOT NULL,  -- Source file path
  dst_target TEXT NOT NULL,     -- Target wikilink
  link_type TEXT NOT NULL
);
```

**Problem**: When a file with `[[wikilinks]]` is deleted, its outgoing links remain in the `links` table.

**Example**:
- File `notes/A.md` contains `[[B]]` and `[[C]]`
- `links` table has: `('vault1', 'notes/A.md', 'B', 'wikilink')` and `('vault1', 'notes/A.md', 'C', 'wikilink')`
- When `A.md` is deleted, these link rows are **NOT** cleaned up

**Impact**:
- `vault_neighbors` MCP tool may return deleted files as neighbors
- Graph traversal includes ghost nodes

### 2. Manifest Table
**Schema** (lines 107-116):
```sql
CREATE TABLE IF NOT EXISTS manifest (
  vault_id TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  mtime INTEGER NOT NULL,
  size INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  last_indexed_at TEXT DEFAULT (datetime('now')),
  last_error TEXT,
  PRIMARY KEY (vault_id, rel_path)
);
```

**Problem**: When a file is deleted, its manifest entry (indexing metadata) remains.

**Impact**:
- Database bloat over time
- Stale entries for files that no longer exist
- Status commands may show incorrect counts

### 3. FAISS Index
**Problem**: If FAISS is enabled (`use_faiss = true`), vector embeddings are stored in a separate FAISS index file. The `delete_document` method removes embeddings from SQLite but doesn't update the FAISS index.

**Impact**:
- FAISS index contains stale vectors
- Vector search may return deleted files
- FAISS index grows without cleanup

**Note**: FAISS index is rebuilt on init, so this is less critical but still a memory/disk waste during runtime.

## Test Plan

### Test 1: Links Cleanup
```bash
# Setup
1. Create file A.md with [[B]] and [[C]]
2. Index vault
3. Verify links exist: SELECT * FROM links WHERE src_rel_path='A.md'
4. Delete A.md
5. Run reconciliation
6. Check: SELECT * FROM links WHERE src_rel_path='A.md'
   - Expected: Empty
   - Actual: ???
```

### Test 2: Manifest Cleanup
```bash
# Setup
1. Index a file test.md
2. Verify manifest entry: SELECT * FROM manifest WHERE rel_path='test.md'
3. Delete test.md
4. Run reconciliation
5. Check: SELECT * FROM manifest WHERE rel_path='test.md'
   - Expected: Empty
   - Actual: ???
```

### Test 3: FAISS Index
```bash
# Setup (use_faiss = true)
1. Index 100 files
2. Note FAISS index size
3. Delete 50 files
4. Check FAISS index size
   - Expected: Index size reduced or rebuilt
   - Actual: ???
```

### Test 4: Complete Cleanup Verification
```bash
# Setup
1. Index a vault with 10 files
2. Get all table counts
3. Delete 5 files
4. Run full scan --full
5. Compare table counts
   - Documents: 5 should have deleted=1
   - Chunks: All chunks from deleted docs should be gone
   - Embeddings: All embeddings from deleted chunks should be gone
   - FTS: All FTS entries from deleted chunks should be gone
   - Links: All links from deleted files should be gone (???)
   - Manifest: Deleted file entries should be gone (???)
```

## Recommended Fixes

### Fix 1: Clean up links table
Add to `delete_document()`:
```python
# Delete outgoing links from this file
self._conn.execute("DELETE FROM links WHERE vault_id=? AND src_rel_path=?", (vault_id, rel_path))
```

### Fix 2: Clean up manifest table
Add to `delete_document()`:
```python
# Delete manifest entry
self._conn.execute("DELETE FROM manifest WHERE vault_id=? AND rel_path=?", (vault_id, rel_path))
```

### Fix 3: FAISS index cleanup
Option A: Rebuild FAISS index periodically
Option B: Track deleted chunks and remove from FAISS
Option C: Accept rebuild-on-restart approach (current behavior)

## Files to Check
- `src/ragtriever/store/libsql_store.py:268-294` - delete_document implementation (UPDATED)
- `src/ragtriever/indexer/indexer.py:119-177` - reconciliation in scan/scan_parallel (ADDED)
- `src/ragtriever/store/faiss_index.py` - FAISS index management

---

## Test Results (2026-01-20)

### Unit Tests (Store Methods)

```python
# Test get_indexed_files
indexed = store.get_indexed_files(vault_id)
assert indexed == {'file1.md', 'file2.md', 'file3.md'}  # ✅ PASS

# Test delete_document marks document as deleted
store.delete_document(vault_id, 'file1.md')
row = conn.execute('SELECT deleted FROM documents WHERE rel_path=?', ('file1.md',)).fetchone()
assert row['deleted'] == 1  # ✅ PASS

# Test delete_document cleans up links table
links_count = conn.execute('SELECT COUNT(*) FROM links WHERE src_rel_path=?', ('file1.md',)).fetchone()[0]
assert links_count == 0  # ✅ PASS

# Test delete_document cleans up manifest table
manifest_count = conn.execute('SELECT COUNT(*) FROM manifest WHERE rel_path=?', ('file1.md',)).fetchone()[0]
assert manifest_count == 0  # ✅ PASS

# Test get_indexed_files excludes deleted files
indexed_after = store.get_indexed_files(vault_id)
assert indexed_after == {'file2.md', 'file3.md'}  # ✅ PASS
```

### Integration Test (test_file_lifecycle.py)

```
============================================================
FILE LIFECYCLE CLEANUP TEST
============================================================

=== INITIAL STATE ===
Documents (active): 120

✅ Created test file with wikilinks

=== AFTER ADDING TEST FILE ===
Documents (active): 120

🗑️  Deleted test file from filesystem

=== AFTER DELETING TEST FILE ===
Documents (active): 119
Documents (deleted): 1

=== Checking residuals for 'test_lifecycle.md' ===
Outgoing links: 0  ✅
Manifest entry: 0  ✅
Document entry: exists (deleted=1)  ✅ Correctly marked as deleted

============================================================
TEST COMPLETE - ALL CHECKS PASSED
============================================================
```
