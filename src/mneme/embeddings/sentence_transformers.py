from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .base import Embedder

logger = logging.getLogger(__name__)


def _find_cached_model(model_id: str) -> str | None:
    """Check if model exists in HuggingFace cache (Artifactory download format).

    Artifactory downloads use: ~/.cache/huggingface/hub/{org}/{model}/main/
    Standard HF cache uses: ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/

    Returns the path if found, None otherwise.
    """
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")

    # Check Artifactory format: {cache}/{model_id}/main/
    artifactory_path = os.path.join(hf_cache, model_id, "main")
    if os.path.exists(artifactory_path):
        # Verify it has model files (not just empty dir)
        contents = os.listdir(artifactory_path)
        if any(f.endswith(('.safetensors', '.bin', '.json')) for f in contents):
            return artifactory_path

    return None


@dataclass
class SentenceTransformersEmbedder:
    model_id: str
    model_path: str | None = None  # Explicit local path for model (overrides model_id for loading)
    device: str = "cpu"
    batch_size: int = 32
    use_query_prefix: bool = True
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    def __post_init__(self) -> None:
        # Suppress harmless multiprocessing resource tracker warnings on macOS
        import warnings
        warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*leaked semaphore")

        from sentence_transformers import SentenceTransformer  # type: ignore

        # Priority: explicit model_path > auto-detected cache > model_id (network)
        model_to_load = self.model_id

        if self.model_path:
            # Explicit path provided - use it
            model_to_load = os.path.expanduser(self.model_path)
            logger.info(f"Loading embedding model from explicit path: {model_to_load}")
        else:
            # Check for cached model (Artifactory download format)
            cached_path = _find_cached_model(self.model_id)
            if cached_path:
                model_to_load = cached_path
                logger.info(f"Found cached model at: {cached_path}")
            else:
                logger.info(f"Loading embedding model: {self.model_id} (will use network if not in standard cache)")

        self._model = SentenceTransformer(model_to_load, device=self.device)
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
