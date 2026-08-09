"""
FAISS index helpers: build, persist, load, and search.
"""
import faiss
import numpy as np
from pathlib import Path


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an IndexFlatL2 over the provided embeddings.

    IndexFlatL2 is an exact nearest-neighbour search — fine for up to ~100k
    vectors. Swap to IndexIVFFlat if the corpus grows significantly larger.
    """
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def search(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) of the k nearest neighbours.

    query_embedding: shape (D,) or (1, D)
    Returns distances and indices both of shape (k,).
    """
    q = query_embedding.astype(np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    distances, indices = index.search(q, k)
    return distances[0], indices[0]
