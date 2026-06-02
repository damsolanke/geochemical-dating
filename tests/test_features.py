"""Tests for geochemical feature engineering."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import add_geochemical_features, get_feature_cols


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "SiO2": [50.0, 70.0],
            "Al2O3": [15.0, 13.0],
            "Fe2O3tot": [10.0, 2.0],
            "CaO": [8.0, 2.0],
            "Total": [100.0, 99.5],
            "Ti": [5000.0, 1000.0],
            "Rb": [50.0, 100.0],
            "Nb": [20.0, 5.0],
            "La": [30.0, 10.0],
            "Ce": [60.0, 20.0],
            "Sm": [5.0, 3.0],
            "Yb": [2.0, 3.0],
            "LaN": [100.0, 33.0],
            "CeN": [75.0, 25.0],
            "SmN": [25.0, 15.0],
            "YbN": [10.0, 15.0],
            "La/Nb": [1.5, 2.0],
            "Rb/Sm": [10.0, 33.0],
            "Nb/La": [0.67, 0.5],
            "Nb/Yb": [10.0, 1.67],
            "Nb/Rb": [0.4, 0.05],
            "La/Sm": [6.0, 3.3],
            "La/Yb": [15.0, 3.3],
            "(La/Yb)N": [10.0, 2.2],
            "Sm/Yb": [2.5, 1.0],
            "(Sm/Yb)N": [2.5, 1.0],
            "SiO2(LOI free)": [52.0, 72.0],
            "TiO2(LOI free)": [1.0, 0.2],
            "Al2O3(LOI free)": [16.0, 14.0],
            "Fe2O3(LOI free)": [11.0, 2.5],
            "MgO(LOI free)": [5.0, 0.5],
            "CaO(LOI free)": [9.0, 2.5],
            "Na2O(LOI free)": [3.0, 5.0],
        }
    )


def test_add_features_adds_columns(sample_df):
    result = add_geochemical_features(sample_df)
    assert "Mg_number" in result.columns
    assert "ASI_proxy" in result.columns
    assert "LREE_HREE_ratio" in result.columns
    assert "Nb_anomaly" in result.columns
    assert "log_Ti" in result.columns
    assert "REE_slope" in result.columns
    assert "Ce_anomaly" in result.columns
    assert len(result.columns) > len(sample_df.columns)


def test_add_features_no_nans(sample_df):
    result = add_geochemical_features(sample_df)
    feature_cols = get_feature_cols(result)
    assert not result[feature_cols].isnull().any().any()


def test_mg_number_range(sample_df):
    result = add_geochemical_features(sample_df)
    assert (result["Mg_number"] >= 0).all()
    assert (result["Mg_number"] <= 1).all()


def test_get_feature_cols_excludes_id_and_label(sample_df):
    sample_df["Label"] = [0, 1]
    result = add_geochemical_features(sample_df)
    feature_cols = get_feature_cols(result)
    assert "id" not in feature_cols
    assert "Label" not in feature_cols
    assert "Id" not in feature_cols


def test_sio2_bin_values(sample_df):
    result = add_geochemical_features(sample_df)
    # SiO2=50 → basic (bin=1), SiO2=70 → acid (bin=3)
    assert result["SiO2_bin"].iloc[0] == 1.0
    assert result["SiO2_bin"].iloc[1] == 3.0
