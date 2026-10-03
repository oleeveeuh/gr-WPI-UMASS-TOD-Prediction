"""Metric calculation and hourly rescaling."""

import numpy as np
import pytest

from tod_pred.metrics import (
    aggregate_fold_metrics,
    minmax_inverse_transform,
    regression_metrics,
)


def test_regression_metrics_exact_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 3.0, 6.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["n"] == 4
    assert metrics["mae"] == pytest.approx(0.75)  # mean(|-1, 0, 0, -2|)
    assert metrics["mse"] == pytest.approx(1.25)  # mean(1, 0, 0, 4)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.25))
    assert metrics["stderr"] == pytest.approx(np.std([-1.0, 0.0, 0.0, -2.0]))


def test_regression_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="non-empty 1-D arrays"):
        regression_metrics(np.zeros(3), np.zeros(4))


def test_aggregate_fold_metrics():
    folds = [
        {"n": 10, "mae": 1.0, "mse": 2.0, "rmse": 1.4, "stderr": 1.0},
        {"n": 10, "mae": 3.0, "mse": 10.0, "rmse": 3.2, "stderr": 3.0},
    ]
    aggregated = aggregate_fold_metrics(folds)
    assert aggregated["mae_mean"] == pytest.approx(2.0)
    assert aggregated["mae_std"] == pytest.approx(1.0)
    assert aggregated["total_n"] == 20


def test_minmax_inverse_round_trip():
    values = np.array([2.0, 4.0, 6.0])
    lo, hi = values.min(), values.max()
    scaled = (values - lo) / (hi - lo)
    np.testing.assert_allclose(
        minmax_inverse_transform(scaled, lo, hi), values
    )


def test_minmax_inverse_rejects_degenerate_range():
    with pytest.raises(ValueError, match="data_max must be greater"):
        minmax_inverse_transform(np.zeros(3), data_min=1.0, data_max=1.0)
