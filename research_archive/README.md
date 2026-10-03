# Research Archive — as-published pipeline

This directory preserves the research pipeline **exactly as it produced the
results in the published paper** (BIOINFORMATICS 2026,
[DOI 10.5220/0014636000004070](https://doi.org/10.5220/0014636000004070)),
including its quirks.  It is kept for transparency and provenance — **not** as
recommended tooling.  Before relying on any number it produces, read
[`docs/LIMITATIONS.md`](../docs/LIMITATIONS.md), which documents confirmed
validation issues (TOD-sorted cross-donor windows, pipeline selection on the
test set, and others).

## Layout

| Path | Role |
|---|---|
| `data_combining.R` | GEO GSE71620 wrangling → one row per donor sample (see caveat below) |
| `train_test_splitting.R` | TOD-bin stratification, train/test splits, normalisation |
| `src/read_train.py` | shared data loading, model training, Excel result writing |
| `src/model_definitions.py` | PyTorch AutoEncoder, CNN, LSTM, MLP definitions |
| `src/find_best_models.py` | scrapes per-config test metrics from the Excel sheets |
| `src/visualizations.py` | final figures (winning configurations hardcoded) |
| `src/best_model_visualizations.R`, `best_model_visuals.Rmd` | R visual analyses |
| `src/option_1/`, `option_2/`, `option_3/` | the three experiment branches (non-windowed, AE-windowed, CNN-windowed) |
| `src/validate.py` | small validation script (moved out of the old `data/reduced_encoded/`) |
| `DR_code/` | second-stage dimensionality reduction (PCA, ICA, KPCA, Isomap) |
| `requirements-legacy.txt` | historical pins (documented; not cleanly installable) |

## Known deviations from a clean checkout (2026 portability fixes)

The only edits made after publication are mechanical path fixes so the scripts
no longer reference a specific laptop (`/Users/olivialiau/Downloads/...`) and
a broken Windows-style glob string:

* `src/visualizations.py` — data path now repo-relative
* `DR_code/KPCA.py` — input/output paths now repo-relative
* `src/option_2/read_output_write_to_excel.py` — glob built with `os.path.join`

No methodological code was changed.  The original files are recoverable from
git history.

## Reproduction caveats

* `data_combining.R` reads `data/raw_data/all_sample.csv` (the full
  ~33,000-gene expression matrix), which was never committed.  The wrangled
  per-region outputs it produces **are** tracked in `data/processed/`, so
  everything downstream of wrangling is reproducible.
* The large intermediate CSV folders (windowed/encoded/reduced/conv) are
  git-ignored and regenerable from these scripts; see `docs/REPRODUCIBILITY.md`.
* `requirements-legacy.txt` is preserved verbatim; see its header for why it
  only ever installed on Python 3.10 x86_64.
