"""FAISS approximate nearest neighbor index for large-scale vector search."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pickle

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class FAISSIndex:
    """FAISS index for approximate nearest neighbor search.

    Supports multiple index types:
    - Flat: Exact search (brute-force), best for <10K vectors
    - IVF: Inverted file index (fast), good for 10K-1M vectors
    - HNSW: Hierarchical Navigable Small World (fastest), good for 100K+ vectors
    """

    embedding_dim: int
    index_type: str = "IVF"  # "Flat", "IVF", "HNSW"
    nlist: int = 100  # Number of clusters (IVF)
    nprobe: int = 10  # Number of clusters to search (IVF)
    metric: str = "cosine"  # "cosine" or "l2"

    def __post_init__(self):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss not installed. Install with: pip install faiss-cpu or pip install faiss-gpu"
            )

        self._lock = threading.Lock()

        # Create index based on type
        if self.index_type == "Flat":
            # Brute-force exact search
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for normalized vectors
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)

        elif self.index_type == "IVF":
            # Inverted file index (approximate)
            quantizer = faiss.IndexFlatIP(self.embedding_dim) if self.metric == "cosine" else faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            self.index.nprobe = self.nprobe
            self._trained = False

        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # M=32 (connectivity)

        else:
            raise ValueError(f"Unknown FAISS index type: {self.index_type}")

        # Chunk ID mapping (FAISS only stores vectors, not IDs)
        self.chunk_ids: list[str] = []

    def train(self, vectors: np.ndarray):
        """Train index (required for IVF before adding vectors)."""
        if self.index_type == "IVF" and not self._trained:
            # Normalize for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(vectors)

            self.index.train(vectors)
            self._trained = True

    def add(self, chunk_ids: list[str], vectors: np.ndarray):
        """Add vectors to index.

        Args:
            chunk_ids: List of chunk IDs corresponding to vectors
            vectors: numpy array of shape (n, embedding_dim)
        """
        with self._lock:
            if vectors.shape[0] != len(chunk_ids):
                raise ValueError(f"Mismatch: {len(chunk_ids)} IDs but {vectors.shape[0]} vectors")

            # Normalize for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(vectors)

            # Train if needed (IVF only)
            if self.index_type == "IVF" and not self._trained:
                if vectors.shape[0] < self.nlist:
                    # Not enough vectors to train, fall back to Flat
                    logger.warning(f"Only {vectors.shape[0]} vectors, but {self.nlist} clusters requested. Using Flat index.")
                    if self.metric == "cosine":
                        self.index = faiss.IndexFlatIP(self.embedding_dim)
                    else:
                        self.index = faiss.IndexFlatL2(self.embedding_dim)
                else:
                    self.train(vectors)

            # Add to index
            self.index.add(vectors)
            self.chunk_ids.extend(chunk_ids)

    def search(self, query_vector: np.ndarray, k: int) -> tuple[list[str], list[float]]:
        """Search for k nearest neighbors.

        Args:
            query_vector: Query embedding vector (1D array)
            k: Number of neighbors to return

        Returns:
            Tuple of (chunk_ids, scores)
        """
        with self._lock:
            if len(self.chunk_ids) == 0:
                return [], []

            # Normalize query
            query = query_vector.reshape(1, -1).astype(np.float32)
            if self.metric == "cosine":
                faiss.normalize_L2(query)

            # Search
            k_actual = min(k, len(self.chunk_ids))  # Don't request more than available
            distances, indices = self.index.search(query, k_actual)

            # Map indices to chunk_ids
            chunk_ids_result = []
            scores_result = []

            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunk_ids):
                    # Invalid index (can happen with IVF if not enough results)
                    continue
                chunk_ids_result.append(self.chunk_ids[idx])

                # Convert distance to similarity score
                distance = distances[0][i]
                if self.metric == "cosine":
                    # Inner product is already similarity (higher = better)
                    scores_result.append(float(distance))
                else:
                    # L2 distance: convert to similarity (lower distance = higher similarity)
                    scores_result.append(float(1 / (1 + distance)))

            return chunk_ids_result, scores_result

    def save(self, path: Path):
        """Save index to disk."""
        with self._lock:
            path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, str(path / "faiss.index"))

            # Save chunk ID mapping
            with open(path / "chunk_ids.pkl", "wb") as f:
                pickle.dump(self.chunk_ids, f)

            # Save metadata
            metadata = {
                "index_type": self.index_type,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "metric": self.metric,
                "embedding_dim": self.embedding_dim,
                "num_vectors": len(self.chunk_ids),
            }
            with open(path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

    def load(self, path: Path):
        """Load index from disk."""
        with self._lock:
            if not (path / "faiss.index").exists():
                raise FileNotFoundError(f"FAISS index not found at {path / 'faiss.index'}")

            # Load FAISS index
            self.index = faiss.read_index(str(path / "faiss.index"))

            # Load chunk ID mapping
            with open(path / "chunk_ids.pkl", "rb") as f:
                self.chunk_ids = pickle.load(f)

            # Restore nprobe for IVF
            if self.index_type == "IVF":
                self.index.nprobe = self.nprobe
                self._trained = True

    def size(self) -> int:
        """Return number of vectors in index."""
        return len(self.chunk_ids)

    def clear(self):
        """Clear all vectors from index."""
        with self._lock:
            # Recreate index (no direct clear method)
            self.__post_init__()
