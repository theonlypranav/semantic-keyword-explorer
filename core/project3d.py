"""UMAP reduction with a fixed seed."""
from __future__ import annotations

import numpy as np

SEED = 42


def umap_reduce(vecs: np.ndarray, n_components: int = 3, n_neighbors: int = 10) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=min(n_neighbors, len(vecs) - 1),
        min_dist=0.1, metric="cosine", random_state=SEED,
    )
    return reducer.fit_transform(vecs)


def pca_angles(vecs: np.ndarray) -> np.ndarray:
    """Angles (radians) for keywords from a 2D PCA projection (F5.9)."""
    n = len(vecs)
    if n < 3:
        return np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = vecs - vecs.mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    xy = u[:, :2] * s[:2]
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    # Keep PCA ordering around the circle but spread evenly so labels don't collide.
    order = np.argsort(ang)
    out = np.empty(n)
    out[order] = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return out
