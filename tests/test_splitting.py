"""Deterministic splitting and stratification behaviour."""

import numpy as np
import pandas as pd

from tod_pred.splitting import (
    make_stratified_folds,
    tod_stratification_labels,
)


def test_folds_are_deterministic(region_frame):
    y = region_frame["TOD"]
    first = make_stratified_folds(y, n_splits=5, random_state=42)
    second = make_stratified_folds(y, n_splits=5, random_state=42)
    assert all(
        np.array_equal(a_train, b_train) and np.array_equal(a_test, b_test)
        for (a_train, a_test), (b_train, b_test) in zip(first, second)
    )


def test_folds_partition_every_row(region_frame):
    y = region_frame["TOD"]
    folds = make_stratified_folds(y, n_splits=5, random_state=42)
    test_union = np.concatenate([test_idx for _, test_idx in folds])
    assert len(test_union) == len(y)
    assert len(np.unique(test_union)) == len(y)  # every row tested exactly once


def test_folds_do_not_overlap(region_frame):
    y = region_frame["TOD"]
    folds = make_stratified_folds(y, n_splits=5, random_state=42)
    for i, (_, test_i) in enumerate(folds):
        for _, test_j in folds[i + 1:]:
            assert not set(test_i) & set(test_j)


def test_stratification_labels_are_balanced(region_frame):
    labels = tod_stratification_labels(region_frame["TOD"], n_bins=4)
    counts = pd.Series(labels).value_counts()
    assert counts.max() - counts.min() <= 1
