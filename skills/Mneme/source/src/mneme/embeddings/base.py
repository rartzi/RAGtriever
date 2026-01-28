from __future__ import annotations

from typing import Protocol, Sequence
import numpy as np

class Embedder(Protocol):
    model_id: str
    dims: int

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def embed_query(self, query: str) -> np.ndarray:
        ...
