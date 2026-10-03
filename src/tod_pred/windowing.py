"""Reference implementation of the historical sliding-window construction.

.. warning::
    This module exists for **documentation and testing only**.  It reproduces
    the window construction used by the published pipeline
    (``research_archive/src/option_2/encode_windows.py`` and the option-3 CNN
    preprocessing) so that its leakage property can be demonstrated
    concretely in the test-suite and described precisely in
    ``docs/LIMITATIONS.md``.  It is **not** used by the leakage-free workflow
    and must never be used to build features for a reported result: each
    window spans 2*W+1 *different donors* from a TOD-sorted table, so the
    neighbouring rows encode the target of the centre row.

Each row of the input frame is one donor sample.  Rows are sorted by TOD
(exactly as the published pipeline does), and for every admissible centre row
the values of each column become the list of that column's values across the
centre row and its W neighbours on each side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import METADATA_COLUMNS, TARGET_COLUMN


def build_sliding_windows(frame: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Return one row per admissible centre row with windowed column values.

    Metadata columns (Age/TOD/Sex) keep the centre row's scalar values; every
    other column becomes a list of 2*W+1 consecutive values.  The construction
    mirrors the published pipeline, including the ``sort_values(TOD)`` step.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    df = frame.sort_values(by=TARGET_COLUMN).reset_index(drop=True)
    new_data: dict[str, list] = {col: [] for col in df.columns}
    for centre in range(window_size, len(df) - window_size):
        for col in df.columns:
            if col in METADATA_COLUMNS:
                new_data[col].append(df.loc[centre, col])
            else:
                new_data[col].append(
                    df.iloc[centre - window_size : centre + window_size + 1][col].tolist()
                )
    return pd.DataFrame(new_data)


def window_leakage_demonstration(
    frame: pd.DataFrame, window_size: int = 1
) -> dict[str, float]:
    """Quantify target leakage in TOD-sorted windows.

    Because rows are sorted by the target, the *neighbours' TOD values* alone
    nearly determine the centre row's TOD.  Returns the MAE (in hours) of
    predicting each centre row's TOD from the mean of its neighbours' TODs for
    (a) the historical TOD-sorted construction and (b) an order-preserving
    control in which windows span arbitrary rows (the construction itself is
    identical; only the sort is removed).  The gap between the two numbers is
    the leakage signal; the tests assert it is large.
    """
    from .data import TARGET_COLUMN

    def neighbour_tod_mae(df: pd.DataFrame, sort_by_target: bool) -> float:
        ordered = (
            df.sort_values(by=TARGET_COLUMN).reset_index(drop=True)
            if sort_by_target
            else df.reset_index(drop=True)
        )
        tod = ordered[TARGET_COLUMN]
        offsets = [o for o in range(-window_size, window_size + 1) if o != 0]
        neighbours = np.mean(
            [tod.shift(o).to_numpy()[window_size : len(df) - window_size] for o in offsets],
            axis=0,
        )
        centres = tod.to_numpy()[window_size : len(df) - window_size]
        return float(np.mean(np.abs(centres - neighbours)))

    return {
        "mae_sorted_windows_hours": neighbour_tod_mae(frame, sort_by_target=True),
        "mae_unsorted_control_hours": neighbour_tod_mae(frame, sort_by_target=False),
    }
