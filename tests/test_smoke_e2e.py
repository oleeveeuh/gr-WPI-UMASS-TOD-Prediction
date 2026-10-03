"""Lightweight end-to-end smoke test of the leakage-free workflow."""

import numpy as np
import pytest
from conftest import make_region_frame

from tod_pred.nested_cv import run_nested_cv
from tod_pred.pipeline import default_estimator_zoo


@pytest.mark.slow
def test_nested_cv_end_to_end_on_synthetic_data():
    frame = make_region_frame(n_rows=90, seed=3)
    features = frame.drop(columns=["TOD"])
    target = frame["TOD"]
    zoo = {
        name: default_estimator_zoo(42)[name]
        for name in ("Ridge", "RandomForest")
    }
    results = run_nested_cv(
        features,
        target,
        estimator_names=list(zoo),
        estimator_zoo=zoo,
        n_outer=2,
        n_inner=2,
        n_iter=2,
        stratification_bins=3,
        random_state=42,
    )
    assert set(results["models"]) == {"Ridge", "RandomForest"}
    for payload in results["models"].values():
        assert len(payload["fold_metrics"]) == 2
        assert np.isfinite(payload["aggregated"]["mae_mean"])
        assert payload["aggregated"]["mae_mean"] > 0
    baseline = results["baseline_mean_predictor"]["aggregated"]
    assert np.isfinite(baseline["mae_mean"])
