from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass
class Reconciler:
    root: Path
    ignore: list[str]

    def scan_files(self) -> list[Path]:
        # TODO: implement ignore pattern matching (glob / pathspec)
        paths: list[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file():
                paths.append(p)
        return paths
