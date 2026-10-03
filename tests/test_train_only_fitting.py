"""Preprocessing must be fitted on training data only."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from tod_pred.data import feature_matrix
from tod_pred.pipeline import build_pipeline


@pytest.fixture
def split_data(region_frame):
    features, target = feature_matrix(region_frame, expected_gene_count=None)
    # Synthetic train/test with clearly different means so a train-fitted
    # scaler is distinguishable from a train+test-fitted one.
    train = features.iloc[:100].copy()
    test = features.iloc[100:].copy()
    train.iloc[:, 2:] += 10.0  # shift gene values in train only
    y_train = target.iloc[:100]
    y_test = target.iloc[100:]
    return train, test, y_train, y_test


def test_scaler_is_fitted_on_train_only(split_data):
    train, _, _, _ = split_data
    pipeline = build_pipeline(Ridge())
    pipeline.fit(train, np.zeros(len(train)))
    mean = pipeline.named_steps["scaler"].mean_
    # Train-only mean of the shifted genes differs strongly from the full mean.
    np.testing.assert_allclose(mean[2:], train.iloc[:, 2:].mean().to_numpy())


def test_pipeline_predicts_without_touching_test_statistics(split_data):
    train, test, y_train, y_test = split_data
    pipeline = build_pipeline(Ridge(alpha=1.0))
    pipeline.fit(train, y_train)
    predictions = pipeline.predict(test)
    assert predictions.shape == y_test.shape


def test_no_refit_after_test_exposure(split_data):
    """The fitted scaler must not change when the pipeline transforms new data."""
    train, test, y_train, _ = split_data
    pipeline = build_pipeline(Ridge(alpha=1.0))
    pipeline.fit(train, y_train)
    mean_before = pipeline.named_steps["scaler"].mean_.copy()
    pipeline.predict(test)
    np.testing.assert_array_equal(pipeline.named_steps["scaler"].mean_, mean_before)
