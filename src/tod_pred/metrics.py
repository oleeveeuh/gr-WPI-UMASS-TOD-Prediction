"""Regression metrics in hours, plus rescaling helpers.

All metrics are computed in the original units of the target (hours of TOD).
The historical pipeline computed metrics on min-max- or log-transformed targets
and rescaled afterwards (see docs/LIMITATIONS.md); this workflow avoids the
transform entirely, and :func:`minmax_inverse_transform` is provided only so
the historical rescaling step itself stays unit-tested and reproducible.
"""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, MSE, RMSE, and residual StdDev in the original units (hours)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty 1-D arrays of equal length")
    residuals = y_true - y_pred
    mse = float(np.mean(residuals**2))
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(residuals))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "stderr": float(np.std(residuals)),
    }


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-fold metric dicts into mean (and std for MAE) across folds."""
    if not fold_metrics:
        raise ValueError("fold_metrics must be non-empty")
    out: dict[str, float] = {}
    for key in ("mae", "mse", "rmse", "stderr"):
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    out["total_n"] = int(sum(m["n"] for m in fold_metrics))
    return out


def minmax_inverse_transform(
    x_scaled: np.ndarray, data_min: float, data_max: float
) -> np.ndarray:
    """Invert min-max scaling: x = x_scaled * (max - min) + min.

    This mirrors the rescaling used for the paper's hourly-scale metrics
    (research_archive/src/visualizations.py); ``data_min``/``data_max`` must
    come from the *training* split, as they did historically.
    """
    if data_max <= data_min:
        raise ValueError("data_max must be greater than data_min")
    return np.asarray(x_scaled, dtype=float) * (data_max - data_min) + data_min
