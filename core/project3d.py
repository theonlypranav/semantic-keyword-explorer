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


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def pca_directions_3d(vecs: np.ndarray) -> np.ndarray:
    """Unit 3D directions for keywords: evenly spread points on a sphere, assigned so that
    keywords close in a 3D PCA of their embeddings get nearby directions (no overlaps)."""
    n = len(vecs)
    slots = fibonacci_sphere(max(n, 1))[:n]
    if n < 4:
        return slots
    x = vecs - vecs.mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    p = u[:, :3] * s[:3]
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    # Greedy assignment: each keyword takes the free slot closest to its PCA direction.
    out = np.zeros((n, 3))
    free = list(range(n))
    for k in np.argsort(-np.linalg.norm(u[:, :3] * s[:3], axis=1)):
        j = max(free, key=lambda f: float(slots[f] @ p[k]))
        out[k] = slots[j]
        free.remove(j)
    return out


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
