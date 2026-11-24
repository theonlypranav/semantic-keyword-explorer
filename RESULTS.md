# Results

_Generated 2026-11-25 by `scripts/evaluate.py`._


## all-MiniLM-L6-v2

- Nearest-neighbour topic accuracy (50 sentences): **90%**
- Adjusted Rand Index (KMeans k=5, seed 42): **0.810**

| Paragraph | MMR | P@10 0.6 | P@10 0.7 | P@10 0.8 | Mean overlap | Time (s) |
|---|---|---|---|---|---|---|
| 1 | on | 0.30 | 0.30 | 0.20 | 0.230 | 0.00 |
| 1 | off | 0.60 | 0.60 | 0.60 | 0.402 | 0.00 |
| 2 | on | 0.40 | 0.40 | 0.20 | 0.145 | 0.00 |
| 2 | off | 0.80 | 0.50 | 0.40 | 0.425 | 0.00 |
| 3 | on | 0.50 | 0.40 | 0.30 | 0.181 | 0.00 |
| 3 | off | 0.90 | 0.80 | 0.50 | 0.378 | 0.00 |

MMR reduces mean keyword overlap from 0.402 to 0.185 (**54%** lower).

Sample paragraph keywords (MMR on): `raised rates`, `recession year`, `sharply borrowing`, `stabilise economists`, `inflation stayed`, `market slow`, `bank shares`, `mortgage costs`, `tightening ease`, `warned housing`

**Failure cases: most similar cross-topic pairs**

- 0.48 · [Sports] The tennis star retired with a knee injury in the third set. ↔ [Health] The patient recovered quickly after knee replacement surgery.
- 0.27 · [Sports] The batting side collapsed and lost eight wickets before lunch. ↔ [Health] The patient recovered quickly after knee replacement surgery.
- 0.25 · [Finance] Diversifying a portfolio reduces exposure to any single asset. ↔ [Health] Regular exercise lowers the risk of heart disease.

**Least similar same-topic pairs**

- -0.07 · [Sports] The club signed a young midfielder on a five-year contract. ↔ The marathon runner collapsed metres from the finish line.
- -0.06 · [Health] High blood pressure often has no obvious symptoms. ↔ The patient recovered quickly after knee replacement surgery.
- -0.05 · [Technology] A software update fixed the security vulnerability. ↔ Electric cars rely on large lithium-ion battery packs.

## WCAG contrast (dark theme)

| Theme | Pair | Ratio | AA |
|---|---|---|---|
| dark | text on background | 13.78 | pass |
| dark | text on surface | 12.27 | pass |
| dark | text_secondary on background | 7.0 | pass |
| dark | text_secondary on surface | 6.23 | pass |
| dark | text_on_beige on beige_fill | 5.43 | pass |
| dark | button text on beige_accent | 7.68 | pass |
