"""Deterministic, leakage-aware cross-validation splitting.

Design notes (see docs/LIMITATIONS.md for the contrast with the historical
pipeline):

* Every row of a per-region dataset is one donor, so donor-level separation and
  stratification by target bins are both achieved with a seeded
  ``StratifiedKFold`` over coarse TOD bins.
* The combined ("full") dataset contains two rows per donor (BA11 + BA47);
  :func:`donor_groups_for_full` recovers donor identity from the shared TOD
  value so grouped folds can be used there.
* Binning is used *only* to balance folds, never to sort rows or build features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .data import TARGET_COLUMN


def tod_stratification_labels(y: pd.Series, n_bins: int = 4) -> np.ndarray:
    """Assign each sample a coarse TOD-bin label via quantiles (deterministic).

    Rows keep their original order; binning is used only to stratify folds.
    """
    labels = pd.qcut(y.rank(method="first"), q=n_bins, labels=False)
    return np.asarray(labels, dtype=int)


def make_stratified_folds(
    y: pd.Series, n_splits: int = 5, n_bins: int = 4, random_state: int = 42
):
    """Yield deterministic (train_idx, test_idx) pairs stratified by TOD bins."""
    labels = tod_stratification_labels(y, n_bins=n_bins)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(y)), labels))


def donor_groups_for_full(frame: pd.DataFrame) -> np.ndarray:
    """Recover donor identity for the combined BA11+BA47 frame.

    ``data/processed/full_data_6_17_2024.csv`` stores each donor's two rows
    (BA11 and BA47) adjacent to each other with an identical TOD value; TOD is
    therefore a reliable donor key here.  Raises if a TOD value is shared by
    more than two rows, which would make the key ambiguous.
    """
    tod = frame[TARGET_COLUMN]
    counts = tod.value_counts()
    if (counts > 2).any():
        ambiguous = counts[counts > 2].index.tolist()
        raise ValueError(
            "TOD values are not a safe donor key: values shared by >2 rows: "
            f"{ambiguous[:5]}..."
        )
    codes = pd.factorize(tod)[0]
    return np.asarray(codes, dtype=int)


def make_grouped_folds(groups: np.ndarray, n_splits: int = 5):
    """Yield deterministic (train_idx, test_idx) pairs that never split a group."""
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(groups)), groups=groups))
