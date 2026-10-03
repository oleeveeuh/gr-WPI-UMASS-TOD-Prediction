"""Nested cross-validation: honest model selection and honest test scores.

Outer loop: stratified k-fold over TOD bins (each row is one donor).
Inner loop: ``RandomizedSearchCV`` over the *training portion only*; the outer
test fold is touched exactly once per model, to score the selected pipeline.
A per-fold mean-baseline (predicting the training mean) is always reported so
the numbers have context.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV

from .metrics import aggregate_fold_metrics, regression_metrics
from .pipeline import DEFAULT_PCA_VARIANCE, build_pipeline, default_estimator_zoo
from .splitting import make_stratified_folds


def _search(
    estimator_spec: dict,
    use_pca: bool,
    pca_variance: float,
    n_inner: int,
    n_iter: int,
    random_state: int,
) -> RandomizedSearchCV:
    pipeline = build_pipeline(
        estimator_spec["estimator"],
        use_pca=use_pca,
        pca_variance=pca_variance,
        random_state=random_state,
    )
    param_distributions = dict(estimator_spec["params"])
    if use_pca:
        # Keep PCA a first-class, inner-CV-selected hyperparameter.
        param_distributions["pca"] = [
            PCA(n_components=pca_variance, random_state=random_state),
            "passthrough",
        ]
    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=n_inner,
        random_state=random_state,
        refit=True,
    )


def _jsonable_best_params(best_params: dict) -> dict:
    out = {}
    for key, value in best_params.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def run_nested_cv(
    features,
    target,
    estimator_names: list[str] | None = None,
    estimator_zoo: dict[str, dict] | None = None,
    n_outer: int = 5,
    n_inner: int = 3,
    n_iter: int = 12,
    stratification_bins: int = 4,
    pca_variance: float = DEFAULT_PCA_VARIANCE,
    random_state: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run nested CV and return a JSON-serialisable results dictionary."""
    zoo = estimator_zoo if estimator_zoo is not None else default_estimator_zoo(random_state)
    names = estimator_names if estimator_names is not None else list(zoo)
    unknown = [n for n in names if n not in zoo]
    if unknown:
        raise ValueError(f"Unknown estimators: {unknown}; available: {sorted(zoo)}")

    outer_folds = make_stratified_folds(
        target, n_splits=n_outer, n_bins=stratification_bins, random_state=random_state
    )

    results: dict[str, Any] = {
        "warning": (
            "NEW leakage-free results produced by src/tod_pred (nested CV). "
            "These are NOT the results reported in the published paper and are "
            "expected to be worse."
        ),
        "configuration": {
            "n_outer_folds": n_outer,
            "n_inner_folds": n_inner,
            "n_iter": n_iter,
            "stratification_bins": stratification_bins,
            "pca_variance": pca_variance,
            "random_state": random_state,
            "estimators": names,
        },
        "models": {},
        "baseline_mean_predictor": {},
    }

    started = time.time()
    for name in names:
        fold_metrics = []
        best_params_per_fold = []
        for train_idx, test_idx in outer_folds:
            search = _search(
                zoo[name], use_pca=True, pca_variance=pca_variance,
                n_inner=n_inner, n_iter=n_iter, random_state=random_state,
            )
            search.fit(features.iloc[train_idx], target.iloc[train_idx])
            prediction = search.predict(features.iloc[test_idx])
            fold_metrics.append(
                regression_metrics(target.iloc[test_idx].to_numpy(), prediction)
            )
            best_params_per_fold.append(_jsonable_best_params(search.best_params_))
            if verbose:
                print(f"  {name}: fold MAE = {fold_metrics[-1]['mae']:.3f} h")
        results["models"][name] = {
            "fold_metrics": fold_metrics,
            "aggregated": aggregate_fold_metrics(fold_metrics),
            "best_params_per_fold": best_params_per_fold,
        }

    baseline_folds = []
    for train_idx, test_idx in outer_folds:
        prediction = np.full(len(test_idx), float(np.mean(target.iloc[train_idx])))
        baseline_folds.append(
            regression_metrics(target.iloc[test_idx].to_numpy(), prediction)
        )
    results["baseline_mean_predictor"] = {
        "fold_metrics": baseline_folds,
        "aggregated": aggregate_fold_metrics(baseline_folds),
    }
    results["runtime_seconds"] = round(time.time() - started, 1)
    return results
