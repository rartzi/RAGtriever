from __future__ import annotations

import hashlib
from pathlib import Path

def blake2b_hex(data: bytes) -> str:
    h = hashlib.blake2b(digest_size=32)
    h.update(data)
    return h.hexdigest()

def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a stable hash of a file's bytes."""
    p = Path(path)
    h = hashlib.blake2b(digest_size=32)
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
