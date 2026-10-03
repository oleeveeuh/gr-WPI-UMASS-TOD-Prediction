"""Leakage-free reference workflow for TOD prediction from circadian gene expression.

This package implements a *new* validation workflow that was written after the
published paper (BIOINFORMATICS 2026, DOI 10.5220/0014636000004070).  It exists
to demonstrate an honest, nested-cross-validated estimate of predictive
performance on the same public dataset, with none of the validation issues
documented in ``docs/LIMITATIONS.md``:

* one row = one donor (no windows are built across donors),
* the target is never used to sort, bin, or construct features,
* all preprocessing is fitted inside training folds only,
* model/hyperparameter selection happens in an inner CV loop, never on a
  held-out test score.

Results from this package are **not** the results reported in the paper and are
expected to be worse.  See ``docs/REPRODUCIBILITY.md`` for usage.
"""

__version__ = "0.1.0"
