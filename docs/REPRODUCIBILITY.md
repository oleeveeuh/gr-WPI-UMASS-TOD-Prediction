# Reproducibility Guide

## Supported environment

* **Python ≥ 3.10** (CI runs 3.10 and 3.13; developed on 3.10).
* Core stack: numpy, pandas, scikit-learn, openpyxl, PyYAML — **CPU only; no
  GPU, no PyTorch/TensorFlow required** for the maintained workflow.
* The archived research pipeline has its own historical (and no longer cleanly
  installable) pins — see
  [`research_archive/requirements-legacy.txt`](../research_archive/requirements-legacy.txt).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]          # or: pip install -r requirements-core.txt
pytest -m "not slow"           # unit tests (seconds)
pytest                         # includes the end-to-end smoke test
```

## Data acquisition

The tracked datasets (`data/raw/`, `data/processed/`) are sufficient for the
maintained workflow.  To re-fetch upstream public sources and verify local
checksums:

```bash
python scripts/download_data.py                        # SOFT metadata + checksums
python scripts/download_data.py --with-expression      # + GEO supplementary archive (large)
```

See [`docs/DATA.md`](DATA.md) for provenance and checksums.

## Leakage-free reference workflow (maintained)

```bash
python scripts/run_nested_cv.py --region BA11
python scripts/run_nested_cv.py --region BA47
# overrides: --n-iter 4 --n-inner 2 --estimators Ridge RandomForest
```

Writes `results/leakage_free/<region>_nested_cv_results.json` and prints a
summary.  Runs in ~90 s per region on CPU.  Deterministic: fixed seed 42,
single-threaded estimators; re-running reproduces byte-identical metrics.
**These results are new (post-publication), labelled as such, and are not the
paper's numbers** — see [`docs/LIMITATIONS.md`](LIMITATIONS.md) for why they
differ.

## Result verification

```bash
python scripts/verify_results.py --check-paper   # validate transcribed paper constants
python scripts/verify_results.py                 # parse results/sheets/ → tidy CSV
```

`--check-paper` validates the paper's hourly-scale table (Table 2) for internal
consistency (RMSE = √MSE; our method lowest per region).  The default mode
parses all 21 historical performance workbooks into
`results/derived/all_model_performance_tidy.csv` and reports the best row per
option/window/region — on *transformed* targets, as documented.

## The archived (as-published) pipeline

Everything under [`research_archive/`](../research_archive/) is preserved for
provenance.  Roughly, the historical flow was:

```
research_archive/data_combining.R        # GEO → wrangled per-region CSVs (needs untracked all_sample.csv)
research_archive/train_test_splitting.R  # TOD-bin splits + normalisation → data/train_test_split_data/
research_archive/src/option_2/encode_windows.py   # TOD-sorted windows + PyTorch autoencoder → data/window, data/encoded
research_archive/DR_code/*.py            # 2nd-stage DR (PCA/ICA/KPCA/Isomap) → data/reduced_*
research_archive/src/option_{1,2,3}/     # 16 regressors per branch → Excel sheets
research_archive/src/find_best_models.py # scrape sheets → winners
```

Its intermediate outputs (`data/train_test_split_data/`, `data/window/`,
`data/encoded/`, `data/reduced_*/`, `data/*conv*/`, `data/all_dfs.RData`) are
git-ignored; each is regenerable from the archived scripts, and all are also
recoverable from git history (`git log --diff-filter=D --name-only`).  Known
gaps in regenerating everything from scratch:

1. `data_combining.R` needs `all_sample.csv` (never committed) — download the
   GEO supplementary archive and convert, or start from the tracked
   `data/processed/` files (recommended).
2. `data/reduced_data/` (option-1 inputs) has no generating script in the repo.
3. ICA outputs are non-deterministic (no `random_state`) and their 90/95
   filename levels are swapped (`docs/LIMITATIONS.md` §6).
4. `requirements-legacy.txt` only ever installed on Python 3.10 x86_64.

## Deep-learning stack (optional)

The archived experiments used PyTorch (AutoEncoder/CNN/LSTM/MLP via skorch) and
TensorFlow (two option-3 CNN scripts).  These are **not** dependencies of the
maintained workflow.  To explore the archive, install torch/skorch manually on
a supported interpreter (see the header of
[`research_archive/requirements-legacy.txt`](../research_archive/requirements-legacy.txt)
for the compatibility pitfalls); there is no CUDA requirement anywhere — the
archived scripts run on CPU (`device` is forced to CPU in `encode_windows.py`).
