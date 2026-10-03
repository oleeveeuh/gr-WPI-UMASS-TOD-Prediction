"""Feature construction must never be informed by the target."""

import pandas as pd

from tod_pred.data import TARGET_COLUMN, feature_matrix


def test_target_column_excluded_from_features(region_frame):
    features, _ = feature_matrix(region_frame, expected_gene_count=None)
    assert TARGET_COLUMN not in features.columns


def test_features_independent_of_target_values(region_frame):
    """Permuting the target must leave the design matrix untouched."""
    features, _ = feature_matrix(region_frame, expected_gene_count=None)
    permuted = region_frame[TARGET_COLUMN].sample(frac=1.0, random_state=1)
    features_again, _ = feature_matrix(
        region_frame.assign(TOD=permuted.to_numpy()), expected_gene_count=None
    )
    pd.testing.assert_frame_equal(features, features_again)
