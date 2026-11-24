"""Evaluation: semantic precision at 10, topic accuracy, ARI, failure pairs."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from . import similarity as S

THRESHOLDS = (0.6, 0.7, 0.8)
SEED = 42


def semantic_precision(pred_vecs: np.ndarray, gold_vecs: np.ndarray, threshold: float) -> float:
    """Fraction of predicted keywords whose max cosine to any gold term >= threshold (F4.1)."""
    if len(pred_vecs) == 0 or len(gold_vecs) == 0:
        return 0.0
    best = S.cosine_matrix(pred_vecs, gold_vecs).max(axis=1)
    return float((best >= threshold).mean())


def missed_gold(pred_vecs: np.ndarray, gold_vecs: np.ndarray, gold: list[str], threshold: float) -> list[str]:
    """Gold terms with no predicted keyword at or above threshold (F4.5)."""
    if len(gold_vecs) == 0:
        return []
    if len(pred_vecs) == 0:
        return list(gold)
    best = S.cosine_matrix(gold_vecs, pred_vecs).max(axis=1)
    return [g for g, b in zip(gold, best) if b < threshold]


def precision_table(pred_vecs, gold_vecs, thresholds=THRESHOLDS) -> dict[str, float]:
    return {str(t): round(semantic_precision(pred_vecs, gold_vecs, t), 3) for t in thresholds}


def nn_topic_accuracy(vecs: np.ndarray, labels: list[str]) -> float:
    sim = S.cosine_matrix(vecs)
    nn = S.closest_partner(sim)
    lab = np.asarray(labels)
    return float((lab[nn] == lab).mean())


def cluster_ari(vecs: np.ndarray, labels: list[str]) -> float:
    k = len(set(labels))
    pred = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(S.normalise(vecs))
    return float(adjusted_rand_score(labels, pred))


def failure_pairs(sim: np.ndarray, labels: list[str], k: int = 5):
    """Top cross-topic similar pairs and bottom same-topic pairs (F4.4)."""
    n = len(labels)
    cross, same = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (same if labels[i] == labels[j] else cross).append((i, j, float(sim[i, j])))
    cross.sort(key=lambda x: -x[2])
    same.sort(key=lambda x: x[2])
    return cross[:k], same[:k]
