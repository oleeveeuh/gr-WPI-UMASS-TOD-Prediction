"""Loading and schema validation for the wrangled TOD datasets.

The tracked processed files (``data/processed/*.csv``) each contain one row per
donor sample with columns ``Age``, ``Sex``, ``TOD`` (hours, the prediction
target) followed by 235 circadian gene expression columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "TOD"
METADATA_COLUMNS = ("Age", "Sex", "TOD")
EXPECTED_GENE_COUNT = 235
EXPECTED_ROW_COUNTS = {"BA11": 146, "BA47": 146, "full": 292}

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REGION_FILES = {
    "BA11": "BA11_data_6_17_2024.csv",
    "BA47": "BA47_data_6_17_2024.csv",
    "full": "full_data_6_17_2024.csv",
}


def load_region(region: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load and validate the wrangled dataset for ``region`` ("BA11", "BA47", "full").

    Raises:
        FileNotFoundError: if the expected CSV is missing.
        ValueError: if the frame does not match the documented schema.
    """
    if region not in REGION_FILES:
        raise ValueError(f"Unknown region {region!r}; expected one of {sorted(REGION_FILES)}")
    data_dir = data_dir or DEFAULT_PROCESSED_DIR
    path = data_dir / REGION_FILES[region]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset for {region}: {path}. "
            "See docs/REPRODUCIBILITY.md for how to restore it."
        )
    frame = pd.read_csv(path)
    validate_region_frame(frame, region=region)
    return frame


def validate_region_frame(
    frame: pd.DataFrame,
    region: str | None = None,
    expected_gene_count: int | None = EXPECTED_GENE_COUNT,
) -> list[str]:
    """Validate the documented schema; return the gene column names.

    Checks column presence and types, domain validity of ``Age``/``Sex``/``TOD``,
    the gene column count (when ``expected_gene_count`` is given), and (when
    ``region`` is given) the expected number of rows.  Synthetic fixtures pass
    ``expected_gene_count=None`` (or their own count) because they use fewer
    than the real 235 genes.  Raises ``ValueError`` with a specific message on
    violation.
    """
    missing = [c for c in METADATA_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Schema error: missing metadata columns {missing}")
    gene_columns = [c for c in frame.columns if c not in METADATA_COLUMNS]
    if not gene_columns:
        raise ValueError("Schema error: no gene columns found")
    if expected_gene_count is not None and len(gene_columns) != expected_gene_count:
        raise ValueError(
            f"Schema error: expected {expected_gene_count} gene columns, found {len(gene_columns)}"
        )
    if frame[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Schema error: NaN values in Age/Sex/TOD")
    tod = frame[TARGET_COLUMN]
    if not pd.api.types.is_numeric_dtype(tod):
        raise ValueError("Schema error: TOD must be numeric (hours)")
    if not ((tod >= 0) & (tod <= 24)).all():
        raise ValueError("Schema error: TOD must be expressed in hours within [0, 24]")
    if not frame["Sex"].isin([0, 1]).all():
        raise ValueError("Schema error: Sex must be coded 0/1")
    if (frame["Age"] <= 0).any():
        raise ValueError("Schema error: Age must be positive")
    if region is not None:
        expected = EXPECTED_ROW_COUNTS[region]
        if len(frame) != expected:
            raise ValueError(
                f"Schema error: {region} should have {expected} rows, found {len(frame)}"
            )
    return gene_columns


def feature_matrix(
    frame: pd.DataFrame, expected_gene_count: int | None = EXPECTED_GENE_COUNT
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated frame into (X, y).

    Features are Age, Sex, and the gene columns; the target is ``TOD`` in
    hours.  The target column is never part of the features and row order is
    preserved.
    """
    gene_columns = validate_region_frame(frame, expected_gene_count=expected_gene_count)
    features = frame[["Age", "Sex"] + gene_columns]
    target = frame[TARGET_COLUMN]
    return features, target
