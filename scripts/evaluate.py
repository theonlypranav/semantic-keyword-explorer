"""Offline evaluation: writes RESULTS.md. Run from the repo root: python scripts/evaluate.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data as D  # noqa: E402
from core import embed as E  # noqa: E402
from core import evaluate as V  # noqa: E402
from core import extract as K  # noqa: E402
from core import similarity as S  # noqa: E402
from core import theme as T  # noqa: E402

out = ["# Results\n", f"_Generated {time.strftime('%Y-%m-%d')} by `scripts/evaluate.py`._\n"]
MODELS = sys.argv[1:] or E.MODELS
for model in MODELS:
    out.append(f"\n## {model}\n")
    sents = [s for _, s in D.SENTENCES]
    labels = [l for l, _ in D.SENTENCES]
    vecs = E.embed(sents, model)
    sim = S.cosine_matrix(vecs)
    out.append(f"- Nearest-neighbour topic accuracy (50 sentences): **{V.nn_topic_accuracy(vecs, labels):.0%}**")
    out.append(f"- Adjusted Rand Index (KMeans k=5, seed 42): **{V.cluster_ari(vecs, labels):.3f}**\n")
    out.append("| Paragraph | MMR | P@10 0.6 | P@10 0.7 | P@10 0.8 | Mean overlap | Time (s) |")
    out.append("|---|---|---|---|---|---|---|")
    on_off = {True: [], False: []}
    for pi, g in enumerate(D.GOLD_PARAGRAPHS):
        gv = E.embed(g["gold"], model)
        for mmr in (True, False):
            t0 = time.perf_counter()
            ex = K.extract(g["text"], model, use_mmr=mmr)
            dt = time.perf_counter() - t0
            p = V.precision_table(ex.kw_vecs, gv)
            mo = S.mean_overlap(ex.kw_vecs)
            on_off[mmr].append(mo)
            out.append(f"| {pi + 1} | {'on' if mmr else 'off'} | {p['0.6']:.2f} | {p['0.7']:.2f} | {p['0.8']:.2f} | {mo:.3f} | {dt:.2f} |")
    a, b = sum(on_off[True]) / 3, sum(on_off[False]) / 3
    out.append(f"\nMMR reduces mean keyword overlap from {b:.3f} to {a:.3f} (**{(b - a) / b:.0%}** lower).\n")
    ex = K.extract(D.SAMPLE_PARAGRAPH, model)
    out.append("Sample paragraph keywords (MMR on): " + ", ".join(f"`{k}`" for k in ex.keywords) + "\n")
    cross, same = V.failure_pairs(sim, labels, k=3)
    out.append("**Failure cases: most similar cross-topic pairs**\n")
    for i, j, s in cross:
        out.append(f"- {s:.2f} · [{labels[i]}] {sents[i]} ↔ [{labels[j]}] {sents[j]}")
    out.append("\n**Least similar same-topic pairs**\n")
    for i, j, s in same:
        out.append(f"- {s:.2f} · [{labels[i]}] {sents[i]} ↔ {sents[j]}")

out.append("\n## WCAG contrast (dark theme)\n")
out.append("| Theme | Pair | Ratio | AA |")
out.append("|---|---|---|---|")
for r in T.contrast_report():
    out.append(f"| {r['theme']} | {r['pair']} | {r['ratio']} | {'pass' if r['AA'] else 'FAIL'} |")

Path("RESULTS.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
