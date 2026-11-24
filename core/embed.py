"""Model loading and batch embedding with an on-disk cache keyed by text hash."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
CACHE_DIR = Path(".cache")


@lru_cache(maxsize=2)
def load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(f"sentence-transformers/{name}", device="cpu")


def _key(model_name: str, texts: list[str]) -> str:
    h = hashlib.sha256(model_name.encode())
    for t in texts:
        h.update(b"\x00" + t.encode())
    return h.hexdigest()[:32]


def embed(texts: list[str], model_name: str, use_cache: bool = True) -> np.ndarray:
    """Return L2-normalised embeddings, shape (n, d)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    path = CACHE_DIR / f"{_key(model_name, texts)}.npy"
    if use_cache and path.exists():
        return np.load(path)
    vecs = load_model(model_name).encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)
    if use_cache:
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(path, vecs)
        except OSError:
            pass
    return vecs


def embed_document(windows: list[str], model_name: str) -> np.ndarray:
    """Embed each window and average (F3.7). Returns a normalised (d,) vector."""
    v = embed(windows, model_name).mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)
