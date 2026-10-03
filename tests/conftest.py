"""Shared synthetic fixtures — no real donor data in tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

GENE_NAMES = [f"GENE{i:03d}" for i in range(30)]


def make_region_frame(
    n_rows: int = 146,
    n_genes: int = 30,
    seed: int = 0,
    full: bool = False,
) -> pd.DataFrame:
    """Synthetic frame with the documented schema.

    Gene expression is a sinusoid of TOD plus noise, so a well-behaved model
    can learn it; no real donor values are involved.
    """
    rng = np.random.default_rng(seed)
    tod = rng.uniform(0, 24, size=n_rows)
    age = rng.integers(16, 97, size=n_rows).astype(float)
    sex = rng.integers(0, 2, size=n_rows).astype(float)
    data = {"Age": age, "Sex": sex, "TOD": tod}
    for index, gene in enumerate(GENE_NAMES[:n_genes]):
        phase = index / max(n_genes, 1) * 2 * np.pi
        data[gene] = np.sin((tod / 24) * 2 * np.pi + phase) + rng.normal(0, 0.05, n_rows)
    frame = pd.DataFrame(data)
    if full:
        # Combined frame: two rows (BA11/BA47) per donor share the same TOD.
        frame = pd.concat([frame, frame], ignore_index=True)
    return frame


@pytest.fixture(scope="session")
def region_frame() -> pd.DataFrame:
    return make_region_frame()


@pytest.fixture(scope="session")
def full_frame() -> pd.DataFrame:
    return make_region_frame(full=True)
