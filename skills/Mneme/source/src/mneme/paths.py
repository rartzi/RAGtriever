from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def relpath(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")

def is_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        logger.debug(f"Path {path} is not within {root}")
        return False
