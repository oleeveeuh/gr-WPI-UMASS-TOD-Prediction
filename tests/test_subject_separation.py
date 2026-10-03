"""Subject/donor separation: one row = one donor; grouped folds respect that."""

import numpy as np

from tod_pred.splitting import donor_groups_for_full, make_grouped_folds


def test_full_frame_groups_pair_ba11_ba47_rows(full_frame):
    groups = donor_groups_for_full(full_frame)
    assert len(groups) == len(full_frame)
    # Each donor = two rows (BA11 + BA47) sharing the same TOD -> one group.
    for donor in np.unique(groups):
        assert (groups == donor).sum() == 2


def test_grouped_folds_never_split_a_donor(full_frame):
    groups = donor_groups_for_full(full_frame)
    folds = make_grouped_folds(groups, n_splits=5)
    for train_idx, test_idx in folds:
        assert not set(groups[train_idx]) & set(groups[test_idx])
