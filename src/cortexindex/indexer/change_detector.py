from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import time

from .queue import Job, JobQueue

@dataclass
class ChangeDetector:
    """Filesystem change detector.

    TODO: Implement watchdog observer for create/modify/delete/move.
    This skeleton provides a no-op placeholder.
    """
    root: Path
    q: JobQueue
    debounce_ms: int = 500

    def watch(self) -> None:
        try:
            from watchdog.observers import Observer  # type: ignore
            from watchdog.events import FileSystemEventHandler  # type: ignore
        except Exception as e:
            raise RuntimeError("watchdog required for watch mode") from e

        root = self.root

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

            def on_created(self, event):  # noqa
                if event.is_directory:
                    return
                p = Path(event.src_path)
                rel = str(p.relative_to(root)).replace("\\", "/")
                if self._debounced(rel):
                    return
                self.outer.q.put(Job(kind="upsert", rel_path=rel))

            def on_modified(self, event):  # noqa
                if event.is_directory:
                    return
                p = Path(event.src_path)
                rel = str(p.relative_to(root)).replace("\\", "/")
                if self._debounced(rel):
                    return
                self.outer.q.put(Job(kind="upsert", rel_path=rel))

            def on_deleted(self, event):  # noqa
                if event.is_directory:
                    return
                p = Path(event.src_path)
                rel = str(p.relative_to(root)).replace("\\", "/")
                self.outer.q.put(Job(kind="delete", rel_path=rel))

            def on_moved(self, event):  # noqa
                if event.is_directory:
                    return
                src = Path(event.src_path)
                dst = Path(event.dest_path)
                rel_src = str(src.relative_to(root)).replace("\\", "/")
                rel_dst = str(dst.relative_to(root)).replace("\\", "/")
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
