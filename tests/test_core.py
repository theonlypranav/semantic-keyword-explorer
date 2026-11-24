import numpy as np
import pytest

from core import evaluate as V
from core import preprocess as P
from core import similarity as S
from core import theme as T


def test_relevance_is_cosine():
    kw = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    doc = np.array([1.0, 0.0])
    np.testing.assert_allclose(S.relevance(kw, doc), [1.0, 0.0, 1 / np.sqrt(2)], atol=1e-9)


def test_redundancy_and_mean_overlap():
    kw = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(S.redundancy(kw), [0.5, 0.5, 0.0], atol=1e-9)
    assert S.mean_overlap(kw) == pytest.approx(1 / 3)


def test_links_monotonic_in_threshold():
    rng = np.random.default_rng(0)
    sim = S.cosine_matrix(rng.normal(size=(10, 8)))
    counts = [len(S.links_above(sim, t)) for t in np.linspace(-1, 1, 41)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_semantic_precision():
    pred = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]])
    gold = np.array([[1.0, 0.0]])
    assert V.semantic_precision(pred, gold, 0.7) == pytest.approx(0.25)
    assert V.semantic_precision(pred, gold, 0.6) == pytest.approx(0.5)
    assert V.missed_gold(pred, np.array([[1.0, 0.0], [0.0, -1.0]]), ["a", "b"], 0.7) == ["b"]


def test_nn_accuracy_and_ari():
    vecs = np.array([[1, 0.01], [1, -0.01], [0.01, 1], [-0.01, 1]], dtype=float)
    labels = ["a", "a", "b", "b"]
    assert V.nn_topic_accuracy(vecs, labels) == 1.0
    assert V.cluster_ari(vecs, labels) == pytest.approx(1.0)


def test_input_validation_and_windows():
    assert P.validate("   ")[0] is False
    assert P.validate("x" * (P.MAX_CHARS + 1))[0] is False
    assert P.validate("A fine paragraph.")[0] is True
    text = " ".join(["This sentence has exactly seven words here."] * 60)
    wins = P.windows(text, max_words=50)
    assert len(wins) > 1 and all(len(w.split()) <= 50 for w in wins)


def test_dark_theme_contrast_wcag_aa():
    failing = [r for r in T.contrast_report() if not r["AA"]]
    assert not failing, failing


def test_topic_colours_distinct():
    assert set(T.TOKENS) == {"dark"}
    assert len(set(T.TOPIC_COLOURS)) == 5 and len(set(T.TOPIC_SYMBOLS_2D)) == 5
