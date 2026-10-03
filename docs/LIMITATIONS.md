# Validation Limitations and Data-Leakage Audit

This document reports the results of a leakage audit of the historical
research pipeline (preserved under [`research_archive/`](../research_archive/)),
performed in 2026 while preparing this repository for public review.  It is
written to be precise about what was found, where, and what it means for the
published results.

**Summary:** the published pipeline contains several validation issues that
plausibly inflate its reported accuracy.  None of this changes *what the paper
reported* — the published numbers are transcribed faithfully elsewhere in this
repository — but the reported MAEs should be treated as **optimistic** until a
clean re-analysis is run.  A clean nested-cross-validated workflow
([`src/tod_pred/`](../src/tod_pred/)) was added alongside this audit; its
results are substantially worse and are reported
[below](#leakage-free-revalidation-2026) and in
[`results/leakage_free/`](../results/leakage_free/).

---

## Confirmed issues in the historical pipeline

### 1. Rows are sorted by the target before features are built

- `research_archive/data_combining.R` line 64 (`arrange(TOD)`) sorts the whole
  dataset by TOD, the prediction target, and
  `research_archive/src/option_2/encode_windows.py` line 46
  (`df.sort_values(by='TOD')`, mirrored in the option-3 CNN preprocessing)
  re-sorts it again before windowing.
- This is a documented design choice in the paper (§2 *Divide Data* and §3.1
  *STEP 1*, which describe sorting samples by TOD to build a
  "pseudo-multivariate time series"), not a hidden bug.  The problem is what
  the sort does downstream.

### 2. Sliding windows span different human subjects

- Each row of the dataset is one donor (one sample from one brain region).
  The "temporal window" around a row is therefore not a time series — it is a
  window over *other, different people* whose only relationship to the centre
  row is that they were **sorted next to it by TOD**.
- Concretely (`research_archive/src/option_2/encode_windows.py` lines 51-62):
  for centre row `i`, every gene's feature becomes the list of that gene's
  values in rows `i-W … i+W`.  Because of the TOD sort, the neighbours' gene
  values encode the neighbours' TODs, which are — by construction — the TODs
  closest to the centre row's TOD.  The target leaks into the features.
- `tests/test_windowing_shapes.py::test_tod_sorted_windows_leak_the_target`
  demonstrates this mechanism quantitatively on synthetic data using
  [`src/tod_pred/windowing.py`](../src/tod_pred/windowing.py), a documented
  reference implementation of the historical construction.
- This is the most serious issue: it plausibly accounts for most of the
  accuracy gap the paper attributes to "sequentiality" (≈2.4–3.3 h MAE without
  windows vs ≈0.8–1.2 h with them).

### 3. Train/test split is positional within TOD bins

- `research_archive/train_test_splitting.R` lines 38-41: within each 2-hour
  TOD bin, the first T% of rows become train and the remainder test.  A
  random-sampling version exists but is commented out.
- The split is deterministic, but train and test samples are not exchangeable:
  within every bin, test donors sit systematically at the high-TOD edge.  In
  the combined "full" dataset a donor's two rows (BA11 and BA47) can land on
  opposite sides of the split.

### 4. Pipeline and model selection used the reported test set

- Every configuration (3 split ratios × 2–3 normalisations × 4 DR methods ×
  2 variance levels × 3 window sizes × 16 models — the paper's §4.3 counts
  730 non-temporal + 2304 CNN-windowed + 2304 autoencoder pipeline-model
  candidates) is scored **once, on the same test set**
  (`research_archive/src/read_train.py` lines 324-343), and the winners are
  then chosen by ranking those test scores (`find_best_models.py`, paper
  Algorithm 1).  There is no nested cross-validation and no untouched holdout.
- Additionally, the selection metric mixes log-scale and min-max-scale MAEs as
  if they were comparable, which they are not.
- The multiple-comparison surface (thousands of candidates ranked on one test
  set) biases the best-reported number upward even with honest per-fit
  evaluation.

### 5. What the historical pipeline got *right*

Verified during the audit — no issue found in these steps:

- Min-max normalisation is fitted on the training split only and applied to
  both splits (`train_test_splitting.R` lines 63-74); the log transform is
  stateless.
- The autoencoder is trained on training-window data only, then encodes both
  splits (`encode_windows.py`).
- PCA/ICA are fitted on the training split only, and the test split is only
  transformed (`research_archive/DR_code/`).

### 6. Additional defects found by the audit

- **ICA variance levels are mislabelled**: `DR_code/ICA.py` writes the
  95%-variance solution to `*_90_*` filenames and vice-versa (four option
  branches), so the "90 vs 95" axis is swapped for every ICA configuration.
- **FastICA has no `random_state`**, so ICA outputs are not reproducible
  run-to-run.
- **The option-1 feature files (`data/reduced_data/`) have no generating
  script in the repository** — the DR scripts read windowed (option-2) inputs,
  not the non-windowed option-1 inputs.
- Results live in hand-formatted Excel files that are scraped back by
  hard-coded cell coordinates (`find_best_models.py`, `read_train.py`
  `write_results_to_excel`), which is fragile; the per-cell search in
  `Isomap.py`/`KPCA.py` contains dead code and heuristic criteria.
- `data_combining.R` reads `data/raw_data/all_sample.csv`, which was never
  committed, so the wrangling step is not reproducible from the repository
  alone (the wrangled outputs are, and they are tracked).

## Consequences for the published numbers

The paper's headline MAEs (0.839 h for BA11, 1.227 h for BA47, hourly scale;
see `results/publication/`) are reproduced faithfully in this repository's
README, but given issues 1-4 they should be read as **lower bounds / optimistic
estimates**, not as expected accuracy on new data.  The paper itself does not
claim external validation, clinical use, or production readiness, and neither
does this repository.

## Leakage-free revalidation (2026)

[`src/tod_pred/`](../src/tod_pred/) implements a clean protocol and
[`scripts/run_nested_cv.py`](../scripts/run_nested_cv.py) runs it:

- one row = one donor; **no windows**; the target is never used in feature
  construction or row ordering;
- all preprocessing (scaling, optional PCA) lives inside the scikit-learn
  pipeline and is fitted within training folds only;
- model + hyperparameter selection happens in an **inner** CV loop
  (`RandomizedSearchCV`, 3-fold, 12 candidates); the outer 5-fold test folds
  are touched exactly once per model;
- deterministic (seed 42; re-running produces bit-identical JSON).

Results (mean ± SD of outer-fold MAE, hours; n = 146 donors per region;
full outputs in `results/leakage_free/`):

| Model | BA11 MAE (h) | BA47 MAE (h) |
|---|---|---|
| Mean-baseline (predict training mean) | 4.881 ± 0.318 | 4.881 ± 0.318 |
| Ridge | **3.683 ± 0.544** | **4.044 ± 0.151** |
| RandomForest | 3.823 ± 0.488 | 4.081 ± 0.498 |
| ExtraTrees | 3.937 ± 0.463 | 4.120 ± 0.311 |
| AdaBoost | 3.928 ± 0.484 | 4.118 ± 0.431 |
| HistGradientBoosting | 4.084 ± 0.516 | 4.237 ± 0.356 |

Reproduce with:

```bash
python scripts/run_nested_cv.py --region BA11
python scripts/run_nested_cv.py --region BA47
```

These numbers are **worse** than the published ones and are labelled as new,
post-publication results everywhere they appear.  They do not tell us what the
paper's models would have scored under a clean protocol; they tell us what a
straightforward model family achieves on this dataset when none of the issues
above are present: roughly 3.7-4.1 h MAE, modestly better than predicting the
mean.  Note also that this workflow drops the window/AE machinery entirely —
by design, because a within-donor window does not exist in this dataset (one
sample per donor).
