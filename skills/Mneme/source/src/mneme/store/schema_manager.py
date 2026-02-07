"""Schema migration manager for Mneme SQLite databases."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1


class SchemaManager:
    """Manages schema versioning and migrations for the Mneme database.

    Uses a `schema_metadata` table to track the current schema version.
    Migrations run sequentially from current version to CURRENT_SCHEMA_VERSION.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def ensure_current(self) -> None:
        """Ensure the database schema is at the current version.

        Creates the schema_metadata table if needed, detects pre-migration
        databases, and runs any pending migrations sequentially.
        """
        self._ensure_metadata_table()
        current = self._get_version()

        if current == CURRENT_SCHEMA_VERSION:
            logger.debug(f"Schema already at version {current}")
            return

        if current == 0 and self._has_existing_schema():
            logger.info("Detected pre-migration database, stamping as v1")
            self._stamp_version(1)
            return

        if current == 0 and not self._has_existing_schema():
            # Fresh database — stamp current version after tables are created
            self._stamp_version(CURRENT_SCHEMA_VERSION)
            return

        # Run migrations sequentially
        for target in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
            migration = _MIGRATIONS.get(target)
            if migration is None:
                raise RuntimeError(
                    f"No migration found for version {target}"
                )
            logger.info(f"Running migration to v{target}")
            migration(self._conn)
            self._stamp_version(target)
            logger.info(f"Migration to v{target} complete")

    def _ensure_metadata_table(self) -> None:
        """Create the schema_metadata table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _get_version(self) -> int:
        """Get current schema version (0 if not set)."""
        row = self._conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            return 0
        return int(row[0] if isinstance(row, tuple) else row["value"])

    def _has_existing_schema(self) -> bool:
        """Detect pre-migration databases by checking for the documents table."""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()
        return row is not None

    def _stamp_version(self, version: int) -> None:
        """Record the current schema version."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO schema_metadata (key, value, applied_at)
               VALUES ('schema_version', ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, applied_at=excluded.applied_at
            """,
            (str(version), now),
        )
        self._conn.commit()


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> None:
    """Migration from v0 to v1: stamp version on existing schema.

    The v0 schema already has all tables; this migration just records
    the version so future migrations can proceed.
    """
    # No structural changes needed — the schema is already correct.
    # Version stamping is handled by SchemaManager._stamp_version().
    pass


# Registry of migrations: target_version -> migration_function
_MIGRATIONS: dict[int, callable] = {
    1: _migrate_v0_to_v1,
}
