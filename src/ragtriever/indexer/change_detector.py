from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

from .queue import Job, JobQueue
from .reconciler import matches_ignore_pattern


@dataclass
class ChangeDetector:
    """Filesystem change detector using watchdog.

    Watches a vault directory for file changes and queues indexing jobs.
    Respects ignore patterns to skip files that shouldn't be indexed.
    """
    root: Path
    q: JobQueue
    ignore: list[str] = field(default_factory=list)
    debounce_ms: int = 500

    def watch(self) -> None:
        try:
            from watchdog.observers import Observer  # type: ignore
            from watchdog.events import FileSystemEventHandler  # type: ignore
        except Exception as e:
            raise RuntimeError("watchdog required for watch mode") from e

        # Resolve symlinks to match watchdog's resolved paths (e.g., /tmp -> /private/tmp on macOS)
        root = self.root.resolve()
        ignore_patterns = self.ignore

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
                if event.is_directory:
                    return
                p = Path(event.src_path)
                rel = str(p.relative_to(root)).replace("\\", "/")
                if self._should_ignore(rel):
                    return
                if self._debounced(rel):
                    return
                self.outer.q.put(Job(kind="upsert", rel_path=rel))

            def on_modified(self, event):  # noqa
                if event.is_directory:
                    return
                p = Path(event.src_path)
                rel = str(p.relative_to(root)).replace("\\", "/")
                if self._should_ignore(rel):
                    return
                if self._debounced(rel):
                    return
                self.outer.q.put(Job(kind="upsert", rel_path=rel))

            def on_deleted(self, event):  # noqa
                if event.is_directory:
                    return
                p = Path(event.src_path)
                rel = str(p.relative_to(root)).replace("\\", "/")
                if self._should_ignore(rel):
                    return
                self.outer.q.put(Job(kind="delete", rel_path=rel))

            def on_moved(self, event):  # noqa
                if event.is_directory:
                    return
                src = Path(event.src_path)
                dst = Path(event.dest_path)
                rel_src = str(src.relative_to(root)).replace("\\", "/")
                rel_dst = str(dst.relative_to(root)).replace("\\", "/")

                # Check if source or destination should be ignored
                src_ignored = self._should_ignore(rel_src)
                dst_ignored = self._should_ignore(rel_dst)

                if src_ignored and dst_ignored:
                    # Both ignored, skip entirely
                    return
                elif src_ignored:
                    # Moving from ignored to non-ignored = treat as create
                    self.outer.q.put(Job(kind="upsert", rel_path=rel_dst))
                elif dst_ignored:
                    # Moving from non-ignored to ignored = treat as delete
                    self.outer.q.put(Job(kind="delete", rel_path=rel_src))
                else:
                    # Normal move
                    self.outer.q.put(Job(kind="move", rel_path=rel_src, new_rel_path=rel_dst))

        observer = Observer()
        observer.schedule(Handler(self), str(root), recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(0.25)
        finally:
            observer.stop()
            observer.join()
