"""Keyword extraction: candidate n-grams ranked by embedding similarity, with optional MMR.

Implements the KeyBERT algorithm directly (so the document embedding can be the
window-averaged one, F3.7) and cross-checks scores against KeyBERT itself (F3.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from . import embed as E
from . import preprocess as P
from . import similarity as S

TOP_N = 10


@dataclass
class Extraction:
    keywords: list[str]
    relevance: np.ndarray
    kw_vecs: np.ndarray
    doc_vec: np.ndarray
    n_candidates: int
    windows: list[str]
    notices: list[str] = field(default_factory=list)


def candidates(text: str) -> list[str]:
    try:
        cv = CountVectorizer(ngram_range=(1, 2), stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
        cv.fit([text])
        return list(cv.get_feature_names_out())
    except ValueError:  # empty vocabulary
        return []


def mmr(doc_vec: np.ndarray, cand_vecs: np.ndarray, top_n: int, diversity: float) -> list[int]:
    """Maximal Marginal Relevance selection (F2.4)."""
    rel = S.cosine_matrix(cand_vecs, doc_vec.reshape(1, -1))[:, 0]
    sim = S.cosine_matrix(cand_vecs)
    selected = [int(rel.argmax())]
    remaining = [i for i in range(len(cand_vecs)) if i != selected[0]]
    while remaining and len(selected) < top_n:
        red = sim[np.ix_(remaining, selected)].max(axis=1)
        score = (1 - diversity) * rel[remaining] - diversity * red
        pick = remaining[int(score.argmax())]
        selected.append(pick)
        remaining.remove(pick)
    return selected


def extract(text: str, model_name: str, use_mmr: bool = True, diversity: float = 0.5, top_n: int = TOP_N) -> Extraction:
    text = P.clean(text)
    wins = P.windows(text)
    notices = []
    if len(wins) > 1:
        notices.append(
            f"The paragraph is longer than the model's input limit, so it was split into {len(wins)} sentence "
            "windows that were embedded separately and averaged."
        )
    doc_vec = E.embed_document(wins, model_name)
    cands = candidates(text)
    if not cands:
        return Extraction([], np.zeros(0), np.zeros((0, len(doc_vec))), doc_vec, 0, wins,
                          notices + ["No candidate keywords found (only stop words?)."])
    cand_vecs = E.embed(cands, model_name)
    if use_mmr:
        idx = mmr(doc_vec, cand_vecs, top_n, diversity)
    else:
        rel_all = S.cosine_matrix(cand_vecs, doc_vec.reshape(1, -1))[:, 0]
        idx = list(np.argsort(-rel_all)[:top_n])
    kw_vecs = cand_vecs[idx]
    rel = S.relevance(kw_vecs, doc_vec)
    order = np.argsort(-rel)
    if len(idx) < top_n:
        notices.append(f"Only {len(idx)} candidate keywords were found, fewer than {top_n}.")
    return Extraction(
        keywords=[cands[idx[i]] for i in order],
        relevance=rel[order],
        kw_vecs=kw_vecs[order],
        doc_vec=doc_vec,
        n_candidates=len(cands),
        windows=wins,
        notices=notices,
    )


def keybert_crosscheck(text: str, model_name: str, keywords: list[str], relevance: np.ndarray, tol: float = 0.01):
    """Compare our relevance with KeyBERT's for the same keywords (F3.5). Returns list of (kw, ours, keybert, flag)."""
    from keybert import KeyBERT

    kb = KeyBERT(model=E.load_model(model_name))
    scores = dict(kb.extract_keywords(P.clean(text), candidates=keywords, top_n=len(keywords), stop_words=None))
    out = []
    for k, r in zip(keywords, relevance):
        s = scores.get(k)
        out.append((k, float(r), None if s is None else float(s), s is not None and abs(float(r) - s) > tol))
    return out
