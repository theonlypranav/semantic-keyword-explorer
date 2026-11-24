"""Colour tokens (single source of truth), CSS injection and Plotly styling."""
from __future__ import annotations

TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "background": "#FAF8F4",
        "surface": "#FFFFFF",
        "text": "#2B2A27",
        "text_secondary": "#5F5C55",
        "border": "#E4E0D8",
        "beige_fill": "#F3E7CC",
        "text_on_beige": "#4A3B1E",
        "beige_accent": "#B8935A",
        "threshold_link": "#8C6D3F",
    },
    "dark": {
        "background": "#1C1B19",
        "surface": "#262522",
        "text": "#E8E6DF",
        "text_secondary": "#A9A59B",
        "border": "#3A3833",
        "beige_fill": "#6B5A3A",
        "text_on_beige": "#F3E7CC",
        "beige_accent": "#CBA86E",
        "threshold_link": "#D9BE8A",
    },
}

# Same in both themes (F7.6 / AC13). Colour + shape together for accessibility.
TOPIC_COLOURS = ["#378ADD", "#D85A30", "#639922", "#7F77DD", "#D4537E"]
TOPIC_SYMBOLS_2D = ["circle", "square", "diamond", "triangle-up", "star"]
TOPIC_SYMBOLS_3D = ["circle", "square", "diamond", "cross", "x"]

# Text pairs that must meet WCAG AA (4.5:1). Button text sits on the accent.
CONTRAST_PAIRS = [
    ("text", "background"),
    ("text", "surface"),
    ("text_secondary", "background"),
    ("text_secondary", "surface"),
    ("text_on_beige", "beige_fill"),
]
BUTTON_TEXT = {"light": "#1C1B19", "dark": "#1C1B19"}


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def contrast_report() -> list[dict]:
    rows = []
    for mode, t in TOKENS.items():
        for fg, bg in CONTRAST_PAIRS:
            r = contrast_ratio(t[fg], t[bg])
            rows.append({"theme": mode, "pair": f"{fg} on {bg}", "ratio": round(r, 2), "AA": r >= 4.5})
        r = contrast_ratio(BUTTON_TEXT[mode], t["beige_accent"])
        rows.append({"theme": mode, "pair": "button text on beige_accent", "ratio": round(r, 2), "AA": r >= 4.5})
    return rows


def resolve(choice: str, system_type: str | None) -> str:
    """Map Light/Dark/System to 'light' or 'dark'."""
    c = (choice or "System").lower()
    if c in ("light", "dark"):
        return c
    return "dark" if (system_type or "").lower() == "dark" else "light"


def css(mode: str) -> str:
    t = TOKENS[mode]
    btn = BUTTON_TEXT[mode]
    return f"""
<style>
:root {{ color-scheme: {mode}; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
  background-color: {t['background']} !important; color: {t['text']} !important; }}
[data-testid="stSidebar"] {{ background-color: {t['surface']} !important; }}
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMetricValue"] {{ color: {t['text']}; }}
.stApp [data-testid="stCaptionContainer"], .stApp small, .muted {{ color: {t['text_secondary']} !important; }}
.stApp textarea, .stApp input, .stApp [data-baseweb="select"] > div {{
  background-color: {t['surface']} !important; color: {t['text']} !important;
  border-color: {t['border']} !important; }}
.stApp [data-testid="stExpander"] details, .stApp [data-testid="stVerticalBlockBorderWrapper"] {{
  border-color: {t['border']} !important; background-color: {t['surface']}; }}
.stApp button[kind="primary"], .stApp [data-testid="stBaseButton-primary"],
.stApp [data-testid="stDownloadButton"] button {{
  background-color: {t['beige_accent']} !important; color: {btn} !important;
  border: 1px solid {t['beige_accent']} !important; }}
.stApp button[kind="primary"] p, .stApp [data-testid="stBaseButton-primary"] p {{ color: {btn} !important; }}
.stApp button[kind="secondary"], .stApp [data-testid="stBaseButton-secondary"] {{
  background-color: {t['surface']} !important; color: {t['text']} !important; border-color: {t['border']} !important; }}
.stApp [data-testid="stBaseButton-segmented_control"], .stApp [data-testid="stBaseButton-segmented_controlActive"] {{
  background-color: {t['surface']} !important; color: {t['text']} !important; border-color: {t['border']} !important; }}
.stApp [data-testid="stBaseButton-segmented_controlActive"] {{
  background-color: {t['beige_fill']} !important; color: {t['text_on_beige']} !important; }}
.stApp [data-testid="stBaseButton-segmented_controlActive"] p {{ color: {t['text_on_beige']} !important; }}
/* Slider: beige track + thumb */
.stApp [data-baseweb="slider"] [role="slider"] {{ background-color: {t['beige_accent']} !important;
  box-shadow: none !important; }}
.stApp [data-testid="stSliderThumbValue"] {{ color: {t['text']} !important; }}
.stApp [data-baseweb="slider"] > div > div > div {{ background: {t['beige_fill']}; }}
/* Focus rings */
.stApp *:focus-visible {{ outline: 2px solid {t['beige_accent']} !important; outline-offset: 2px; }}
.stApp a {{ color: {t['threshold_link']}; }}
.kw-chip {{ display:inline-block; padding:2px 10px; margin:2px 4px 2px 0; border-radius:999px;
  background:{t['beige_fill']}; color:{t['text_on_beige']}; font-size:0.9rem; }}
.hl {{ background:{t['beige_fill']}; color:{t['text_on_beige']}; padding:0 3px; border-radius:3px; }}
table.kw {{ width:100%; border-collapse:collapse; background:{t['surface']}; }}
table.kw th, table.kw td {{ border-bottom:1px solid {t['border']}; padding:6px 10px; text-align:left; color:{t['text']}; }}
table.kw th {{ color:{t['text_secondary']}; font-weight:600; }}
table.kw tr:hover td {{ background:{t['beige_fill']}; color:{t['text_on_beige']}; }}
</style>
"""


def layout(mode: str, **extra) -> dict:
    """Plotly layout with explicit colours so templates don't override them."""
    t = TOKENS[mode]
    base = dict(
        template="plotly_dark" if mode == "dark" else "plotly_white",
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font=dict(color=t["text"]),
        hoverlabel=dict(bgcolor=t["beige_fill"], font_color=t["text_on_beige"], bordercolor=t["beige_accent"]),
        legend=dict(font=dict(color=t["text"])),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    base.update(extra)
    return base
