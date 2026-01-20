#!/usr/bin/env python3
"""Test script to verify file lifecycle cleanup (add/remove files)."""

import sqlite3
import tempfile
from pathlib import Path
import shutil
import sys

def check_table_counts(db_path: Path, label: str):
    """Print counts of all tables."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print(f"\n=== {label} ===")

    # Documents
    docs = conn.execute("SELECT COUNT(*) as cnt FROM documents WHERE deleted=0").fetchone()["cnt"]
    docs_deleted = conn.execute("SELECT COUNT(*) as cnt FROM documents WHERE deleted=1").fetchone()["cnt"]
    print(f"Documents (active): {docs}")
    print(f"Documents (deleted): {docs_deleted}")

    # Chunks
    chunks = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()["cnt"]
    print(f"Chunks: {chunks}")

    # Embeddings
    embeddings = conn.execute("SELECT COUNT(*) as cnt FROM embeddings").fetchone()["cnt"]
    print(f"Embeddings: {embeddings}")

    # FTS
    fts = conn.execute("SELECT COUNT(*) as cnt FROM fts_chunks").fetchone()["cnt"]
    print(f"FTS entries: {fts}")

    # Links
    links = conn.execute("SELECT COUNT(*) as cnt FROM links").fetchone()["cnt"]
    print(f"Links: {links}")

    # Manifest
    manifest = conn.execute("SELECT COUNT(*) as cnt FROM manifest").fetchone()["cnt"]
    print(f"Manifest entries: {manifest}")

    conn.close()

def check_specific_file(db_path: Path, rel_path: str):
    """Check if specific file has residuals."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print(f"\n=== Checking residuals for '{rel_path}' ===")

    # Links where this file is the source
    links_out = conn.execute("SELECT COUNT(*) as cnt FROM links WHERE src_rel_path=?", (rel_path,)).fetchone()["cnt"]
    print(f"Outgoing links: {links_out}")
    if links_out > 0:
        print("  ❌ RESIDUAL: Links not cleaned up!")
        rows = conn.execute("SELECT dst_target, link_type FROM links WHERE src_rel_path=?", (rel_path,)).fetchall()
        for row in rows:
            print(f"    - [{row['link_type']}] -> {row['dst_target']}")

    # Links where this file is the target
    links_in = conn.execute("SELECT COUNT(*) as cnt FROM links WHERE dst_target LIKE ?", (f"%{rel_path}%",)).fetchone()["cnt"]
    print(f"Incoming links: {links_in} (expected - these should remain)")

    # Manifest entry
    manifest_entry = conn.execute("SELECT COUNT(*) as cnt FROM manifest WHERE rel_path=?", (rel_path,)).fetchone()["cnt"]
    print(f"Manifest entry: {manifest_entry}")
    if manifest_entry > 0:
        print("  ❌ RESIDUAL: Manifest entry not cleaned up!")

    # Document entry
    doc = conn.execute("SELECT deleted FROM documents WHERE rel_path=?", (rel_path,)).fetchone()
    if doc:
        print(f"Document entry: exists (deleted={doc['deleted']})")
        if doc['deleted'] == 1:
            print("  ✅ Correctly marked as deleted")
        else:
            print("  ⚠️ Still active!")
    else:
        print("Document entry: not found (hard deleted)")

    conn.close()

def main():
    print("=" * 60)
    print("FILE LIFECYCLE CLEANUP TEST")
    print("=" * 60)

    # Use the existing test vault and index
    config_path = Path("config.toml")
    if not config_path.exists():
        print("❌ config.toml not found!")
        print("Run this from the project root directory.")
        sys.exit(1)

    # Parse config to get index location
    import tomllib
    config = tomllib.loads(config_path.read_text())
    index_dir = Path(config["index"]["dir"]).expanduser()
    db_path = index_dir / "vaultrag.sqlite"

    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        print("Run 'ragtriever scan --full' first")
        sys.exit(1)

    vault_root = Path(config["vault"]["root"])

    # Step 1: Initial state
    check_table_counts(db_path, "INITIAL STATE")

    # Step 2: Create a test file with wikilinks
    test_file = vault_root / "test_lifecycle.md"
    test_file.write_text("""# Test File for Lifecycle

This file contains wikilinks: [[Projects/Cloud Migration]] and [[AI Gateway Access Initiative]]

And some regular content to ensure chunks are created.
""")

    print(f"\n✅ Created test file: {test_file}")

    # Step 3: Index it
    print("\n📝 Running scan to index test file...")
    import subprocess
    result = subprocess.run(
        ["ragtriever", "scan", "--config", "config.toml"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Scan failed: {result.stderr}")
        sys.exit(1)

    # Step 4: Check that it was indexed
    check_table_counts(db_path, "AFTER ADDING TEST FILE")
    check_specific_file(db_path, "test_lifecycle.md")

    # Step 5: Delete the test file
    test_file.unlink()
    print(f"\n🗑️  Deleted test file: {test_file}")

    # Step 6: Run scan again (should detect deletion)
    print("\n📝 Running scan to detect deletion...")
    result = subprocess.run(
        ["ragtriever", "scan", "--config", "config.toml"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Scan failed: {result.stderr}")
        sys.exit(1)

    # Step 7: Check residuals
    check_table_counts(db_path, "AFTER DELETING TEST FILE")
    check_specific_file(db_path, "test_lifecycle.md")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nSummary:")
    print("- Check above for ❌ RESIDUAL warnings")
    print("- Links and manifest entries should be cleaned up")
    print("- Document should be marked as deleted=1")

if __name__ == "__main__":
    main()
