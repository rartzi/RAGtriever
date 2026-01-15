from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import json
import urllib.request

from .base import Embedder

@dataclass
class OllamaEmbedder:
    """Adapter for a local Ollama embeddings endpoint.

    TODO: confirm endpoint shape and add robust error handling/timeouts.
    This adapter is optional; local-only policy still holds because it targets localhost.
    """
    model_id: str
    endpoint: str = "http://127.0.0.1:11434/api/embeddings"
    timeout_s: float = 30.0
    dims: int = 0

    def _call(self, prompt: str) -> list[float]:
        payload = {"model": self.model_id, "prompt": prompt}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        vec = out.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError(f"Unexpected Ollama response: {out}")
        return [float(x) for x in vec]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = [self._call(t) for t in texts]
        arr = np.array(vectors, dtype=np.float32)
        if self.dims == 0:
            self.dims = int(arr.shape[1])
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        v = self._call(query)
        arr = np.array(v, dtype=np.float32)
        if self.dims == 0:
            self.dims = int(arr.shape[0])
        return arr
