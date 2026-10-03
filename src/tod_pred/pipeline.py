"""Pipelines and small search spaces for the leakage-free workflow.

Every estimator is wrapped in a scikit-learn :class:`~sklearn.pipeline.Pipeline`
so that scaling and dimensionality reduction are fitted **inside each training
fold only** — there is no code path that fits a transform on held-out data.
"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_PCA_VARIANCE = 0.90


def build_pipeline(
    estimator,
    use_pca: bool = False,
    pca_variance: float = DEFAULT_PCA_VARIANCE,
    random_state: int = 0,
) -> Pipeline:
    """Return a Pipeline(scale -> [PCA] -> estimator) with fixed random states."""
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, random_state=random_state)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def default_estimator_zoo(random_state: int = 42) -> dict[str, dict]:
    """A small, CPU-friendly zoo: estimator + randomized-search distributions.

    Deliberately far smaller than the historical search space (16 models x
    thousands of pipeline configurations): the goal here is an honest estimate,
    not an optimisation.
    """
    return {
        "Ridge": {
            "estimator": Ridge(random_state=random_state),
            "params": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=random_state, n_jobs=1),
            "params": {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
        "ExtraTrees": {
            "estimator": ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            "params": {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
        "AdaBoost": {
            "estimator": AdaBoostRegressor(random_state=random_state),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.1, 0.5, 1.0],
            },
        },
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingRegressor(random_state=random_state),
            "params": {
                "model__max_iter": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [None, 3],
            },
        },
    }
