from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING

from .queue import Job, JobQueue
from .reconciler import matches_ignore_pattern

if TYPE_CHECKING:
    from ..store.base import Store

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetector:
    """Filesystem change detector using watchdog.

    Watches a vault directory for file changes and queues indexing jobs.
    Respects ignore patterns to skip files that shouldn't be indexed.
    """
    root: Path
    q: JobQueue
    store: Store
    vault_id: str
    ignore: list[str] = field(default_factory=list)
    debounce_ms: int = 500

    def queue_stale_files(self) -> int:
        """Queue files modified since last index for reprocessing.

        Compares filesystem mtimes against manifest to detect files that
        changed while the watcher was stopped. Queues them for reindex.

        Returns:
            Number of stale files queued.
        """
        root = self.root.resolve()
        queued = 0
        new_files = 0

        # Get last-indexed mtimes from manifest
        manifest_mtimes = self.store.get_manifest_mtimes(self.vault_id)

        logger.info("[watch] Checking for files modified since last index...")

        # Walk filesystem and compare
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue

            try:
                rel = str(file_path.relative_to(root)).replace("\\", "/")
            except ValueError:
                continue

            # Skip ignored files
            if matches_ignore_pattern(rel, self.ignore):
                continue

            # Get filesystem mtime
            try:
                fs_mtime = int(file_path.stat().st_mtime)
            except OSError:
                continue

            # Check if file is new or modified since last index
            manifest_mtime = manifest_mtimes.get(rel)

            if manifest_mtime is None:
                # New file not in manifest
                logger.debug(f"[watch] New file (not in index): {rel}")
                self.q.put(Job(kind="upsert", rel_path=rel))
                new_files += 1
                queued += 1
            elif fs_mtime > manifest_mtime:
                # File modified since last index
                logger.debug(f"[watch] Stale file (modified): {rel}")
                self.q.put(Job(kind="upsert", rel_path=rel))
                queued += 1

        if queued > 0:
            logger.info(
                f"[watch] Queued {queued} stale files for reindex "
                f"({new_files} new, {queued - new_files} modified)"
            )
        else:
            logger.info("[watch] No stale files found - index is up to date")

        return queued

    def watch(self) -> None:
        try:
            from watchdog.observers import Observer  # type: ignore
            from watchdog.events import FileSystemEventHandler  # type: ignore
        except Exception as e:
            raise RuntimeError("watchdog required for watch mode") from e

        # Resolve symlinks to match watchdog's resolved paths (e.g., /tmp -> /private/tmp on macOS)
        root = self.root.resolve()
        ignore_patterns = self.ignore

        logger.info(f"Starting file watcher on: {root}")
        if ignore_patterns:
            logger.info(f"Ignore patterns: {ignore_patterns}")
        logger.debug(f"Debounce interval: {self.debounce_ms}ms")

        class Handler(FileSystemEventHandler):
            def __init__(self, outer: "ChangeDetector") -> None:
                self.outer = outer
                self._last: dict[str, float] = {}

            def _debounced(self, rel: str) -> bool:
                now = time.time()
                last = self._last.get(rel, 0.0)
                if (now - last) * 1000 < self.outer.debounce_ms:
                    self._last[rel] = now
                    return True
                self._last[rel] = now
                return False

            def _should_ignore(self, rel: str) -> bool:
                """Check if the path should be ignored based on patterns."""
                return matches_ignore_pattern(rel, ignore_patterns)

            def on_created(self, event):  # noqa
                p = Path(event.src_path)

                # Handle new directory: scan for files inside
                if event.is_directory:
                    logger.info(f"Directory created: {p}")
                    self._scan_directory(p)
                    return

                rel = str(p.relative_to(root)).replace("\\", "/")
                if self._should_ignore(rel):
                    logger.debug(f"Ignoring created file (matches pattern): {rel}")
                    return
                if self._debounced(rel):
                    logger.debug(f"Debounced created event: {rel}")
                    return
                logger.info(f"File created: {rel}")
                self.outer.q.put(Job(kind="upsert", rel_path=rel))

            def _scan_directory(self, dir_path: Path) -> None:
                """Scan a newly created directory for files to index.

                When a folder is copied/created with files inside, watchdog may
                only fire a directory event without individual file events.
                This ensures all files in the new directory are queued.
                """
                queued_count = 0
                ignored_count = 0
                try:
                    # Use rglob to recursively find all files
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file():
                            try:
                                rel = str(file_path.relative_to(root)).replace("\\", "/")
                                if self._should_ignore(rel):
                                    ignored_count += 1
                                    logger.debug(f"Ignoring file in new directory (matches pattern): {rel}")
                                    continue
                                if self._debounced(rel):
                                    logger.debug(f"Debounced file in new directory: {rel}")
                                    continue
                                self.outer.q.put(Job(kind="upsert", rel_path=rel))
                                queued_count += 1
                                logger.debug(f"Queued file from new directory: {rel}")
                            except ValueError:
                                # File not under root (shouldn't happen)
                                logger.warning(f"File not under vault root: {file_path}")
                                continue
                    logger.info(
                        f"Directory scan complete: {dir_path.name} - "
                        f"queued {queued_count} files, ignored {ignored_count}"
                    )
                except Exception as e:
                    logger.error(f"Error scanning new directory {dir_path}: {e}")

            def on_modified(self, event):  # noqa
                if event.is_directory:
                    return
                p = Path(event.src_path)
                rel = str(p.relative_to(root)).replace("\\", "/")
                if self._should_ignore(rel):
                    logger.debug(f"Ignoring modified file (matches pattern): {rel}")
                    return
                if self._debounced(rel):
                    logger.debug(f"Debounced modified event: {rel}")
                    return
                logger.info(f"File modified: {rel}")
                self.outer.q.put(Job(kind="upsert", rel_path=rel))

            def on_deleted(self, event):  # noqa
                p = Path(event.src_path)
                rel = str(p.relative_to(root)).replace("\\", "/")

                if event.is_directory:
                    # Directory deleted: find all indexed files under this path
                    logger.info(f"Directory deleted: {rel}")
                    try:
                        files_under = self.outer.store.get_files_under_path(
                            self.outer.vault_id, rel
                        )
                        if files_under:
                            for file_rel in files_under:
                                if not self._should_ignore(file_rel):
                                    self.outer.q.put(Job(kind="delete", rel_path=file_rel))
                            logger.info(
                                f"Directory deleted: {rel} - queued {len(files_under)} files for deletion"
                            )
                        else:
                            logger.debug(f"Directory deleted: {rel} - no indexed files found")
                    except Exception as e:
                        logger.error(f"Error handling directory deletion {rel}: {e}")
                    return

                if self._should_ignore(rel):
                    logger.debug(f"Ignoring deleted file (matches pattern): {rel}")
                    return
                logger.info(f"File deleted: {rel}")
                self.outer.q.put(Job(kind="delete", rel_path=rel))

            def on_moved(self, event):  # noqa
                src = Path(event.src_path)
                dst = Path(event.dest_path)
                rel_src = str(src.relative_to(root)).replace("\\", "/")
                rel_dst = str(dst.relative_to(root)).replace("\\", "/")

                if event.is_directory:
                    # Directory moved: find all indexed files under source path
                    logger.info(f"Directory moved: {rel_src} -> {rel_dst}")
                    try:
                        files_under = self.outer.store.get_files_under_path(
                            self.outer.vault_id, rel_src
                        )
                        if files_under:
                            for file_rel in files_under:
                                # Compute new path by replacing source prefix with dest prefix
                                new_file_rel = file_rel.replace(rel_src, rel_dst, 1)

                                # Check ignore patterns for old and new paths
                                src_ignored = self._should_ignore(file_rel)
                                dst_ignored = self._should_ignore(new_file_rel)

                                if src_ignored and dst_ignored:
                                    continue
                                elif src_ignored:
                                    # Moving from ignored to non-ignored = treat as create
                                    self.outer.q.put(Job(kind="upsert", rel_path=new_file_rel))
                                elif dst_ignored:
                                    # Moving from non-ignored to ignored = treat as delete
                                    self.outer.q.put(Job(kind="delete", rel_path=file_rel))
                                else:
                                    # Normal move
                                    self.outer.q.put(Job(kind="move", rel_path=file_rel, new_rel_path=new_file_rel))

                            logger.info(
                                f"Directory moved: {rel_src} -> {rel_dst} - queued {len(files_under)} files"
                            )
                        else:
                            logger.debug(f"Directory moved: {rel_src} -> {rel_dst} - no indexed files found")
                    except Exception as e:
                        logger.error(f"Error handling directory move {rel_src} -> {rel_dst}: {e}")
                    return

                # Check if source or destination should be ignored
                src_ignored = self._should_ignore(rel_src)
                dst_ignored = self._should_ignore(rel_dst)

                if src_ignored and dst_ignored:
                    # Both ignored, skip entirely
                    logger.debug(f"Ignoring move (both paths match patterns): {rel_src} -> {rel_dst}")
                    return
                elif src_ignored:
                    # Moving from ignored to non-ignored = treat as create
                    logger.info(f"File moved from ignored location (treating as create): {rel_dst}")
                    self.outer.q.put(Job(kind="upsert", rel_path=rel_dst))
                elif dst_ignored:
                    # Moving from non-ignored to ignored = treat as delete
                    logger.info(f"File moved to ignored location (treating as delete): {rel_src}")
                    self.outer.q.put(Job(kind="delete", rel_path=rel_src))
                else:
                    # Normal move
                    logger.info(f"File moved: {rel_src} -> {rel_dst}")
                    self.outer.q.put(Job(kind="move", rel_path=rel_src, new_rel_path=rel_dst))

        observer = Observer()
        observer.schedule(Handler(self), str(root), recursive=True)
        observer.start()
        logger.info("File watcher started - monitoring for changes")
        try:
            while True:
                time.sleep(0.25)
        finally:
            logger.info("Stopping file watcher...")
            observer.stop()
            observer.join()
            logger.info("File watcher stopped")
