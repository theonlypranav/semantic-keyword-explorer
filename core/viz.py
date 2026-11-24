"""Plotly charts: constellation, network, 3D embedding view, heatmap. All colours set per trace."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import similarity as S
from . import theme as T
from .project3d import pca_directions_3d


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def constellation(keywords, relevance, kw_vecs, threshold: float, mode: str, selected: str | None = None):
    """3D constellation: paragraph at the origin, keywords on spherical shells by relevance."""
    t = T.TOKENS[mode]
    rel = np.asarray(relevance, dtype=float)
    n = len(keywords)
    sim = S.cosine_matrix(kw_vecs) if n else np.zeros((0, 0))
    lo, hi = (rel.min(), rel.max()) if n else (0, 1)
    span = (hi - lo) or 1.0
    radius = 0.45 + 0.55 * (hi - rel) / span  # more relevant -> closer to the centre
    dirs = pca_directions_3d(kw_vecs) if n else np.zeros((0, 3))
    xyz = dirs * radius[:, None]
    fig = go.Figure()
    # One faint dotted ring per relevance shell (instead of full wireframes)
    th = np.linspace(0, 2 * np.pi, 120)
    for r in (0.45, 0.725, 1.0):
        fig.add_trace(go.Scatter3d(x=r * np.cos(th), y=r * np.sin(th), z=np.zeros_like(th), mode="lines",
                                   hoverinfo="skip", line=dict(color=_rgba(t["border"], 0.9), width=2, dash="dot"),
                                   showlegend=False))
    # Keyword-to-keyword links above threshold, width by overlap
    links = S.links_above(sim, threshold) if n else []
    for i, j, s in links:
        w = 1.5 + 4.5 * max(0.0, s - threshold) / max(1e-6, 1 - threshold)
        mid = (xyz[i] + xyz[j]) / 2
        fig.add_trace(go.Scatter3d(x=[xyz[i, 0], mid[0], xyz[j, 0]], y=[xyz[i, 1], mid[1], xyz[j, 1]],
                                   z=[xyz[i, 2], mid[2], xyz[j, 2]], mode="lines", hoverinfo="text",
                                   text=f"{keywords[i]} ↔ {keywords[j]}: {s:.2f}",
                                   line=dict(color=_rgba(t["threshold_link"], 0.55), width=w), showlegend=False))
    partner = S.closest_partner(sim) if n > 1 else np.zeros(n, dtype=int)
    hover = [
        f"<b>{k}</b><br>relevance {r:.3f}" + (f"<br>closest: {keywords[p]} ({sim[i, p]:.2f})" if n > 1 else "")
        for i, (k, r, p) in enumerate(zip(keywords, rel, partner))
    ]
    sizes = 6 + 8 * (rel - lo) / span
    colours = [t["beige_accent"] if k == selected else t["beige_fill"] for k in keywords]
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", hovertext=hover, hoverinfo="text",
        marker=dict(size=sizes, color=colours, opacity=1, line=dict(color=t["beige_accent"], width=2)),
        showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", hoverinfo="text",
                               hovertext="Paragraph embedding",
                               marker=dict(size=6, color=t["text"], symbol="diamond"), showlegend=False))
    # Labels as scene annotations: chips with a solid background, so they stay readable over lines.
    top3 = set(np.argsort(-rel)[:3]) if n else set()
    notes = []
    for i, k in enumerate(keywords):
        sel = k == selected
        notes.append(dict(
            x=xyz[i, 0], y=xyz[i, 1], z=xyz[i, 2], text=f"<b>{k}</b>" if i in top3 or sel else k,
            showarrow=False, yshift=16, borderpad=3, borderwidth=1, opacity=1,
            bgcolor=t["beige_fill"] if sel else _rgba(t["background"], 0.85),
            bordercolor=t["beige_accent"] if sel else _rgba(t["border"], 1.0),
            font=dict(color=t["text_on_beige"] if sel else t["text"], size=12),
        ))
    notes.append(dict(x=0, y=0, z=0, text="<i>paragraph</i>", showarrow=False, yshift=-14,
                      font=dict(color=t["text_secondary"], size=11)))
    axis = dict(visible=False, range=[-1.1, 1.1])
    fig.update_layout(**T.layout(
        mode, height=640, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube", bgcolor=t["surface"],
                   annotations=notes, camera=dict(eye=dict(x=1.25, y=1.25, z=0.75))),
        title=dict(text=f"Keyword constellation · {len(links)} links ≥ {threshold:.2f}",
                   font=dict(color=t["text"], size=15)),
    ))
    return fig, len(links)


def network(sentences, labels, sim, threshold: float, coords2d, mode: str):
    t = T.TOKENS[mode]
    fig = go.Figure()
    ex, ey = [], []
    n = len(sentences)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                ex += [coords2d[i, 0], coords2d[j, 0], None]
                ey += [coords2d[i, 1], coords2d[j, 1], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=t["threshold_link"], width=1),
                             opacity=0.5, hoverinfo="skip", showlegend=False))
    for ti, topic in enumerate(T_topics(labels)):
        idx = [i for i, l in enumerate(labels) if l == topic]
        fig.add_trace(go.Scatter(
            x=coords2d[idx, 0], y=coords2d[idx, 1], mode="markers", name=topic,
            marker=dict(size=11, color=T.TOPIC_COLOURS[ti % 5], symbol=T.TOPIC_SYMBOLS_2D[ti % 5],
                        line=dict(color=t["surface"], width=1)),
            hovertext=[sentences[i] for i in idx], hoverinfo="text"))
    ax = dict(visible=False)
    fig.update_layout(**T.layout(mode, height=480, xaxis=ax, yaxis=ax,
                                 title=dict(text=f"Similarity network (edges ≥ {threshold:.2f})")))
    return fig


def T_topics(labels):
    seen = []
    for l in labels:
        if l not in seen:
            seen.append(l)
    return seen


def scatter3d(sentences, labels, coords3d, mode: str, highlight: int | None = None):
    t = T.TOKENS[mode]
    fig = go.Figure()
    for ti, topic in enumerate(T_topics(labels)):
        idx = [i for i, l in enumerate(labels) if l == topic]
        fig.add_trace(go.Scatter3d(
            x=coords3d[idx, 0], y=coords3d[idx, 1], z=coords3d[idx, 2], mode="markers", name=topic,
            marker=dict(size=5, color=T.TOPIC_COLOURS[ti % 5], symbol=T.TOPIC_SYMBOLS_3D[ti % 5]),
            hovertext=[sentences[i] for i in idx], hoverinfo="text"))
    if highlight is not None:
        c = coords3d[highlight]
        fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers", name="Selected",
                                   marker=dict(size=12, color=t["beige_fill"], line=dict(color=t["beige_accent"], width=4)),
                                   hovertext=[sentences[highlight]], hoverinfo="text"))
    axis = dict(backgroundcolor=t["surface"], gridcolor=t["border"], color=t["text_secondary"], showticklabels=False, title="")
    fig.update_layout(**T.layout(mode, height=560, scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, bgcolor=t["surface"]),
                                 title=dict(text="3D embedding view (UMAP, exploratory — distances are approximate)")))
    return fig


def heatmap(sentences, labels, sim, mode: str, sort_by_topic: bool = True):
    t = T.TOKENS[mode]
    order = np.argsort([T_topics(labels).index(l) for l in labels], kind="stable") if sort_by_topic else np.arange(len(labels))
    m = sim[np.ix_(order, order)]
    names = [f"{labels[i][:4]}·{i}" for i in order]
    scale = [[0, t["surface"]], [0.5, t["beige_fill"]], [1, t["threshold_link"]]]
    fig = go.Figure(go.Heatmap(z=m, x=names, y=names, colorscale=scale, zmin=float(np.min(m)), zmax=1,
                               hovertext=[[f"{sentences[order[i]]}<br>↔ {sentences[order[j]]}<br>{m[i, j]:.2f}"
                                           for j in range(len(order))] for i in range(len(order))],
                               hoverinfo="text", colorbar=dict(tickfont=dict(color=t["text"]))))
    fig.update_layout(**T.layout(mode, height=600, title=dict(text="Sentence similarity heatmap"),
                                 xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, autorange="reversed")))
    return fig
