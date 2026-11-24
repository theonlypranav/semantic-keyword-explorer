"""Semantic Keyword and Similarity Explorer — Streamlit UI. Core logic lives in core/."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from core import data as D
from core import embed as E
from core import evaluate as V
from core import export as X
from core import extract as K
from core import preprocess as P
from core import project3d as U
from core import similarity as S
from core import theme as T
from core import viz

st.set_page_config(page_title="Semantic Keyword Explorer", page_icon="✦", layout="wide")

ss = st.session_state
ss.setdefault("text", "")
ss.setdefault("gold", "")
ss.setdefault("history", [])
ss.setdefault("run", None)
ss.setdefault("pending", False)

# ---------- Header + theme ----------
h1, h2 = st.columns([4, 1.4])
with h2:
    choice = st.segmented_control("Theme", ["Light", "Dark", "System"], default="System", key="theme_choice",
                                  label_visibility="collapsed") or "System"
try:
    system_type = st.context.theme.type
except Exception:
    system_type = None
mode = T.resolve(choice, system_type)
st.markdown(T.css(mode), unsafe_allow_html=True)
tok = T.TOKENS[mode]

with h1:
    st.title("Semantic Keyword Explorer")
    st.caption("Paste a paragraph → 10 keywords, scored for relevance and overlap with sentence embeddings. "
               "Runs locally with open models, no paid APIs.")


# ---------- Input ----------
def _load_sample():
    ss.text = D.SAMPLE_PARAGRAPH
    ss.gold = "\n".join(D.SAMPLE_GOLD)


def _request_run():
    ss.pending = True


def _load_history():
    i = ss.get("history_pick")
    if i is not None and 0 <= i < len(ss.history):
        h = ss.history[i]
        ss.text, ss.gold, ss.run = h["text"], h["gold"], h["run"]


st.text_area("Paragraph", key="text", height=180, on_change=_request_run,
             placeholder="Paste a paragraph here. Press Ctrl+Enter (⌘+Enter on Mac) or click Analyse.")
nchar, nword = P.counts(ss.text)
over = nchar > P.MAX_CHARS
st.caption(f"{nchar:,} characters · {nword:,} words" + (f" — over the {P.MAX_CHARS:,} limit" if over else ""))
if over:
    st.warning(f"The paragraph is over {P.MAX_CHARS:,} characters. Please shorten it.")

with st.expander("Gold keywords (optional, one per line) — enables precision at 10", expanded=bool(ss.gold)):
    st.text_area("Gold keywords", key="gold", height=140, label_visibility="collapsed")

c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.6, 1, 1])
model = c1.selectbox("Model", E.MODELS, index=0)
use_mmr = c2.toggle("MMR", value=True, help="Maximal Marginal Relevance reduces near-duplicate keywords.")
diversity = c3.slider("MMR diversity", 0.0, 1.0, 0.5, 0.05, disabled=not use_mmr)
c4.write("")
clicked = c4.button("Analyse", type="primary", use_container_width=True)
c5.write("")
c5.button("Load sample", on_click=_load_sample, use_container_width=True)

if ss.history:
    st.selectbox("Recent runs", options=list(range(len(ss.history))), index=None, key="history_pick",
                 format_func=lambda i: f"{ss.history[i]['run']['run_at'][11:19]} · {ss.history[i]['run']['model']} · "
                                       f"{'MMR' if ss.history[i]['run']['mmr'] else 'no MMR'} · {ss.history[i]['text'][:60]}…",
                 placeholder="Reload one of the last 5 runs", on_change=_load_history)


# ---------- Pipeline ----------
def analyse(text: str, gold_raw: str) -> dict:
    t0 = time.perf_counter()
    ex = K.extract(text, model, use_mmr=use_mmr, diversity=diversity)
    run = {
        "model": model, "mmr": use_mmr, "diversity": diversity if use_mmr else None,
        "keywords": ex.keywords, "relevance": ex.relevance.tolist(), "kw_vecs": ex.kw_vecs,
        "avg_overlap": S.redundancy(ex.kw_vecs).tolist(), "mean_overlap": S.mean_overlap(ex.kw_vecs),
        "notices": ex.notices, "n_candidates": ex.n_candidates, "run_at": X.now_iso(),
        "sentences": P.split_sentences(text), "precision": None, "missed": None,
    }
    if run["sentences"] and ex.keywords:
        sv = E.embed(run["sentences"], model)
        run["support"] = S.cosine_matrix(ex.kw_vecs, sv).argmax(axis=1).tolist()
    gold = [g.strip() for g in gold_raw.splitlines() if g.strip()]
    if gold and ex.keywords:
        gv = E.embed(gold, model)
        run["precision"] = V.precision_table(ex.kw_vecs, gv)
        run["missed"] = {str(th): V.missed_gold(ex.kw_vecs, gv, gold, th) for th in V.THRESHOLDS}
    run["elapsed"] = time.perf_counter() - t0
    return run


if clicked or ss.pending:
    ss.pending = False
    ok, msg = P.validate(ss.text)
    if not ok:
        st.error(msg)
    else:
        with st.spinner("Embedding and ranking… (first run downloads the model)"):
            run = analyse(ss.text, ss.gold)
        ss.run = run
        ss.history = ([{"text": ss.text, "gold": ss.gold, "run": run}] + ss.history)[:5]

# ---------- Results ----------
run = ss.run
if run:
    st.divider()
    st.subheader("Results")
    for n in run["notices"]:
        st.info(n)
    if not run["keywords"]:
        st.stop()
    kws, rel, ovl = run["keywords"], np.array(run["relevance"]), np.array(run["avg_overlap"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Keywords", len(kws))
    m2.metric("Mean keyword overlap", f"{run['mean_overlap']:.3f}", help="Lower = more diverse (F3.4)")
    m3.metric("Candidates scored", run["n_candidates"])
    m4.metric("Time", f"{run['elapsed']:.2f}s")
    st.markdown(" ".join(f'<span class="kw-chip">{k}</span>' for k in kws), unsafe_allow_html=True)
    st.caption(f"Model `{run['model']}` · MMR {'on (diversity ' + str(run['diversity']) + ')' if run['mmr'] else 'off'}")

    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("**Keyword table**")
        sort_col = st.radio("Sort by", ["relevance", "avg_overlap", "keyword"], horizontal=True, key="sort_col")
        asc = st.toggle("Ascending", value=sort_col == "keyword", key="sort_asc")
        df = pd.DataFrame({"keyword": kws, "relevance": rel.round(3), "avg_overlap": ovl.round(3)})
        df = df.sort_values(sort_col, ascending=asc)
        rows = "".join(f"<tr><td>{r.keyword}</td><td>{r.relevance:.3f}</td><td>{r.avg_overlap:.3f}</td></tr>"
                       for r in df.itertuples())
        st.markdown(f'<table class="kw"><thead><tr><th>keyword</th><th>relevance</th><th>avg overlap</th></tr>'
                    f"</thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

        if run["precision"]:
            st.markdown("**Semantic precision at 10**")
            pt = pd.DataFrame({"threshold": list(run["precision"].keys()),
                               "precision@10": list(run["precision"].values())})
            st.dataframe(pt, hide_index=True, use_container_width=True)
            st.caption("A keyword counts as correct if its cosine to any gold term ≥ threshold. "
                       "Headline threshold: **0.7**.")
            missed = run["missed"]["0.7"]
            st.markdown("**Gold terms missed at 0.7:** " + (", ".join(missed) if missed else "none"))

    with right:
        threshold = st.slider("Overlap threshold (keyword ↔ keyword links)", 0.0, 1.0, 0.4, 0.01, key="threshold")
        selected = st.selectbox("Why did this keyword rank here?", ["—"] + kws, key="kw_pick")
        sel = None if selected == "—" else selected
        fig, nlinks = viz.constellation(kws, rel, run["kw_vecs"], threshold, mode, selected=sel)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        top = kws[int(rel.argmax())]
        st.caption(f"Text summary: {len(kws)} keywords; most relevant is “{top}” ({rel.max():.3f}), least is "
                   f"“{kws[int(rel.argmin())]}” ({rel.min():.3f}). {nlinks} keyword pairs overlap at ≥ {threshold:.2f}. "
                   "Closer to the centre and larger = more relevant. Drag to rotate, scroll to zoom.")
        if sel and run.get("support"):
            i = kws.index(sel)
            sent = run["sentences"][run["support"][i]]
            sim = S.cosine_matrix(run["kw_vecs"])
            p = int(S.closest_partner(sim)[i]) if len(kws) > 1 else i
            st.markdown(f"**{sel}** — relevance {rel[i]:.3f}, avg overlap {ovl[i]:.3f}, closest partner "
                        f"**{kws[p]}** ({sim[i, p]:.2f}).<br>Top supporting sentence: "
                        f'<span class="hl">{sent}</span>', unsafe_allow_html=True)

    with st.expander("Cross-check against KeyBERT (F3.5)"):
        if st.button("Run cross-check"):
            rows = K.keybert_crosscheck(ss.text, run["model"], kws, rel)
            cdf = pd.DataFrame(rows, columns=["keyword", "ours", "keybert", "diff > 0.01"])
            st.dataframe(cdf, hide_index=True, use_container_width=True)
            st.caption("KeyBERT embeds the full text in one pass; differences appear when the paragraph was split "
                       "into windows or exceeds the model's token limit.")

    res = X.results_dict(run, mode, threshold)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download results.csv", X.to_csv(res), "results.csv", "text/csv", use_container_width=True)
    d2.download_button("Download results.json", X.to_json(res), "results.json", "application/json",
                       use_container_width=True)
    d3.download_button("Download constellation.html", X.fig_html(fig), "constellation.html", "text/html",
                       use_container_width=True)


# ---------- Sentence explorer ----------
@st.cache_data(show_spinner="Embedding the 50-sentence dataset…")
def sentence_analysis(model_name: str):
    sents = [s for _, s in D.SENTENCES]
    labels = [l for l, _ in D.SENTENCES]
    vecs = E.embed(sents, model_name)
    sim = S.cosine_matrix(vecs)
    cross, same = V.failure_pairs(sim, labels)
    return {
        "sents": sents, "labels": labels, "sim": sim,
        "nn_acc": V.nn_topic_accuracy(vecs, labels), "ari": V.cluster_ari(vecs, labels),
        "c3": U.umap_reduce(vecs, 3), "c2": U.umap_reduce(vecs, 2), "cross": cross, "same": same,
    }


st.divider()
with st.expander("Sentence explorer — 50 sentences, 5 topics (3D view, heatmap, network, failures)"):
    if st.toggle("Load sentence explorer", key="load_explorer"):
        sa = sentence_analysis(model)
        e1, e2 = st.columns(2)
        e1.metric("Nearest-neighbour topic accuracy", f"{sa['nn_acc']:.0%}")
        e2.metric("Adjusted Rand Index (KMeans, k=5, seed 42)", f"{sa['ari']:.3f}")
        tab3d, tabh, tabn, tabf = st.tabs(["3D view", "Heatmap", "Network", "Failure panel"])
        with tab3d:
            pick = st.selectbox("Highlight a sentence", [None] + list(range(len(sa["sents"]))),
                                format_func=lambda i: "—" if i is None else sa["sents"][i])
            f3 = viz.scatter3d(sa["sents"], sa["labels"], sa["c3"], mode, highlight=pick)
            st.plotly_chart(f3, use_container_width=True, theme=None)
            st.caption("Text summary: each point is a sentence, coloured and shaped by topic. Drag to rotate. "
                       "UMAP is exploratory — use the numeric scores for conclusions.")
            st.download_button("Download 3D view (HTML)", X.fig_html(f3), "embedding3d.html", "text/html")
        with tabh:
            fh = viz.heatmap(sa["sents"], sa["labels"], sa["sim"], mode, st.toggle("Sort by topic", True))
            st.plotly_chart(fh, use_container_width=True, theme=None)
            st.caption("Text summary: bright diagonal blocks mean sentences within a topic are similar to each other.")
        with tabn:
            nt = st.slider("Edge threshold", 0.0, 1.0, 0.35, 0.01, key="net_t")
            st.plotly_chart(viz.network(sa["sents"], sa["labels"], sa["sim"], nt, sa["c2"], mode),
                            use_container_width=True, theme=None)
        with tabf:
            st.markdown("**Most similar cross-topic pairs** (the model thinks these are related)")
            st.dataframe(pd.DataFrame([(sa["labels"][i], sa["sents"][i], sa["labels"][j], sa["sents"][j], round(s, 3))
                                       for i, j, s in sa["cross"]],
                                      columns=["topic A", "sentence A", "topic B", "sentence B", "cosine"]),
                         hide_index=True, use_container_width=True)
            st.markdown("**Least similar same-topic pairs** (the model misses the shared topic)")
            st.dataframe(pd.DataFrame([(sa["labels"][i], sa["sents"][i], sa["sents"][j], round(s, 3))
                                       for i, j, s in sa["same"]],
                                      columns=["topic", "sentence A", "sentence B", "cosine"]),
                         hide_index=True, use_container_width=True)

with st.expander("Accessibility: WCAG contrast of theme tokens"):
    st.dataframe(pd.DataFrame(T.contrast_report()), hide_index=True, use_container_width=True)
