"""Expected input schema."""

import numpy as np
import pytest
from conftest import GENE_NAMES, make_region_frame

from tod_pred.data import TARGET_COLUMN, feature_matrix, validate_region_frame


def test_valid_frame_passes(region_frame):
    gene_columns = validate_region_frame(region_frame, expected_gene_count=len(GENE_NAMES))
    assert len(gene_columns) == len(GENE_NAMES)


def test_missing_metadata_column_fails():
    frame = make_region_frame().drop(columns=["Age"])
    with pytest.raises(ValueError, match="missing metadata columns"):
        validate_region_frame(frame)


def test_wrong_gene_count_fails():
    frame = make_region_frame(n_genes=10)
    with pytest.raises(ValueError, match="expected 235 gene columns"):
        validate_region_frame(frame, expected_gene_count=235)


def test_tod_outside_hours_fails():
    frame = make_region_frame()
    frame.loc[0, TARGET_COLUMN] = 25.0
    with pytest.raises(ValueError, match=r"TOD must be.*\[0, 24\]"):
        validate_region_frame(frame, expected_gene_count=None)


def test_non_binary_sex_fails():
    frame = make_region_frame()
    frame.loc[0, "Sex"] = 2
    with pytest.raises(ValueError, match="Sex must be coded 0/1"):
        validate_region_frame(frame, expected_gene_count=None)


def test_feature_matrix_excludes_target_and_preserves_order(region_frame):
    features, target = feature_matrix(
        region_frame, expected_gene_count=len(GENE_NAMES)
    )
    assert TARGET_COLUMN not in features.columns
    assert list(features.index) == list(region_frame.index)
    assert len(target) == len(features)
    np.testing.assert_allclose(target.to_numpy(), region_frame[TARGET_COLUMN].to_numpy())
