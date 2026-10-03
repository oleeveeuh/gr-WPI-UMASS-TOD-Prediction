"""Windowing output shapes — and the leakage the historical construction incurs."""

import numpy as np
import pytest
from conftest import make_region_frame

from tod_pred.data import TARGET_COLUMN
from tod_pred.windowing import build_sliding_windows, window_leakage_demonstration


def test_window_shape_and_metadata(region_frame):
    window_size = 2
    windows = build_sliding_windows(region_frame, window_size)
    expected_rows = len(region_frame) - 2 * window_size
    assert len(windows) == expected_rows
    for column in ("Age", "Sex", TARGET_COLUMN):
        assert column in windows.columns
    # Gene columns are list-valued with 2*W+1 entries.
    gene_column = next(c for c in windows.columns if c.startswith("GENE"))
    assert all(len(values) == 2 * window_size + 1 for values in windows[gene_column])


def test_windows_preserve_centre_metadata(region_frame):
    windows = build_sliding_windows(region_frame, window_size=1)
    sorted_tod = region_frame.sort_values(TARGET_COLUMN)[TARGET_COLUMN].to_numpy()
    np.testing.assert_allclose(windows[TARGET_COLUMN].to_numpy(), sorted_tod[1:-1])


def test_windows_are_invalid_for_zero_window(region_frame):
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        build_sliding_windows(region_frame, window_size=0)


def test_tod_sorted_windows_leak_the_target():
    """The historical construction is target-informed: because rows are sorted
    by TOD, a window's neighbouring rows have the closest TODs to the centre
    row, so neighbours' TODs nearly determine the centre TOD.  The
    order-preserving control (identical construction, no sort) destroys the
    signal.  This test documents the mechanism in docs/LIMITATIONS.md."""
    rng = np.random.default_rng(0)
    n = 200
    tod = rng.uniform(0, 24, n)
    frame = make_region_frame(n_rows=n, seed=0)
    frame[TARGET_COLUMN] = tod
    scores = window_leakage_demonstration(frame, window_size=1)
    assert (
        scores["mae_sorted_windows_hours"]
        < 0.25 * scores["mae_unsorted_control_hours"]
    )
