"""
Embedding abstraction.
Uses sentence-transformers (local, free) for embeddings.
Controlled by environment variables:
  EMBEDDING_MODEL = model name (e.g. "all-MiniLM-L6-v2")
"""
import os
from typing import Union
import numpy as np

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_embedder = None  # module-level singleton


def _build_local_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(texts: Union[str, list[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    return embed


def get_embedder():
    """Return (and cache) the configured embed function.

    The returned callable accepts str | list[str] and returns np.ndarray
    of shape (N, D) where D is the embedding dimension.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    _embedder = _build_local_embedder()
    return _embedder


def embed(texts: Union[str, list[str]]) -> np.ndarray:
    """Convenience wrapper — embeds texts using the configured provider."""
    return get_embedder()(texts)
