"""Relevance, redundancy, similarity matrices and neighbour lookup."""
from __future__ import annotations

import numpy as np


def normalise(m: np.ndarray) -> np.ndarray:
    m = np.atleast_2d(np.asarray(m, dtype=np.float64))
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    a = normalise(a)
    b = a if b is None else normalise(b)
    return a @ b.T


def relevance(kw_vecs: np.ndarray, doc_vec: np.ndarray) -> np.ndarray:
    """cosine(keyword, paragraph) for each keyword (F3.2)."""
    return cosine_matrix(kw_vecs, doc_vec.reshape(1, -1))[:, 0]


def redundancy(kw_vecs: np.ndarray) -> np.ndarray:
    """Mean cosine of each keyword to the others (F3.3)."""
    n = len(kw_vecs)
    if n < 2:
        return np.zeros(n)
    s = cosine_matrix(kw_vecs)
    return (s.sum(axis=1) - np.diag(s)) / (n - 1)


def mean_overlap(kw_vecs: np.ndarray) -> float:
    """Mean pairwise keyword-to-keyword similarity, one diversity score (F3.4)."""
    n = len(kw_vecs)
    if n < 2:
        return 0.0
    s = cosine_matrix(kw_vecs)
    iu = np.triu_indices(n, k=1)
    return float(s[iu].mean())


def links_above(sim: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    n = sim.shape[0]
    return [(i, j, float(sim[i, j])) for i in range(n) for j in range(i + 1, n) if sim[i, j] >= threshold]


def closest_partner(sim: np.ndarray) -> np.ndarray:
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return s.argmax(axis=1)
