from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Job:
    kind: str  # upsert|delete|move
    rel_path: str
    new_rel_path: Optional[str] = None

class JobQueue:
    def __init__(self) -> None:
        self._q: "queue.Queue[Job]" = queue.Queue()

    def put(self, job: Job) -> None:
        self._q.put(job)

    def get(self, timeout: float | None = None) -> Job:
        return self._q.get(timeout=timeout)

    def task_done(self) -> None:
        self._q.task_done()
