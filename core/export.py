"""CSV, JSON and standalone HTML output."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pandas as pd


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def results_dict(run: dict, theme: str, threshold: float) -> dict:
    return {
        "model": run["model"],
        "mmr": run["mmr"],
        "diversity": run["diversity"],
        "theme": theme,
        "overlap_threshold": threshold,
        "keywords": [
            {"term": k, "relevance": round(float(r), 4), "avg_overlap": round(float(o), 4)}
            for k, r, o in zip(run["keywords"], run["relevance"], run["avg_overlap"])
        ],
        "mean_keyword_overlap": round(float(run["mean_overlap"]), 4),
        "precision_at_10": run.get("precision"),
        "precision_threshold_reported": 0.7 if run.get("precision") else None,
        "missed_gold": run.get("missed"),
        "run_at": run["run_at"],
        "exported_at": now_iso(),
    }


def to_json(d: dict) -> str:
    return json.dumps(d, indent=2)


def to_csv(d: dict) -> str:
    df = pd.DataFrame(d["keywords"])
    for k in ("model", "mmr", "diversity", "theme", "overlap_threshold", "mean_keyword_overlap", "run_at"):
        df[k] = d[k]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def fig_html(fig) -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=True)
