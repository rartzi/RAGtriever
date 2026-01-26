from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .base import Embedder

@dataclass
class SentenceTransformersEmbedder:
    model_id: str
    device: str = "cpu"
    batch_size: int = 32
    use_query_prefix: bool = True
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    def __post_init__(self) -> None:
        # Suppress harmless multiprocessing resource tracker warnings on macOS
        import warnings
        warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*leaked semaphore")

        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(self.model_id, device=self.device)
        # Determine dims from a small encode
        v = self._model.encode(["dimension_probe"], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
        self.dims = int(v.shape[1])

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents (no prefix)."""
        return self._model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with optional instruction prefix for asymmetric retrieval."""
        if self.use_query_prefix and self.query_prefix:
            query = self.query_prefix + query
        return self._model.encode([query], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)[0]
