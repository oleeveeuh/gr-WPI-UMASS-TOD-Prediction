# Results directory

| Path | What it is | Authority |
|---|---|---|
| `publication/` | The published paper PDF (`BIOINFORMATICS_2026_398_CR.pdf`, proceedings pp. 704–715) and the WPI REU poster (`poster.pdf`). | **Authoritative** for every published number (headline table = Table 2, PDF p. 11). |
| `sheets/` | 21 historical Excel workbooks ("Overall Model Peformance Results" — the filename typo is original) holding raw per-configuration metrics, written by `research_archive/src/read_train.py`. Metrics are computed on *transformed* (min-max/log) targets, so values are not comparable across normalisation methods and are **not** the paper's headline numbers. | Process artifact |
| `derived/` | Tidy CSV scrapes of `sheets/`: `opt_{1,2,3}_all_models_performance.csv` (historical) and `all_model_performance_tidy.csv` (2026 regeneration via `scripts/verify_results.py`). | Derived |
| `figures/` | `BA11_plot.png` / `BA47_plot.png` — predicted-vs-actual scatter grids produced during the research from the winning configurations. Their StdDev annotations match the paper (BA11 ExtraTrees 0.996; BA47 AdaBoost 1.451; BA11 non-temporal LSTM 3.077). The current script version writes different filenames, so these PNGs are historical outputs (regenerable in spirit via `research_archive/src/visualizations.py`). | Illustrative |
| `leakage_free/` | **NEW (2026, post-publication; NOT paper results).** Nested-CV outputs of `src/tod_pred/` per region, with configuration, data checksum, and an explicit warning label inside each JSON. | New results |

Verify with `python scripts/verify_results.py [--check-paper]`.
