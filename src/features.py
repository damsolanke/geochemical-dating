"""Geochemical feature engineering.

Domain-informed features for igneous rock classification based on
major oxide geochemistry, trace element ratios, and REE patterns.
"""

import numpy as np
import pandas as pd


def add_geochemical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific geochemical features."""
    df = df.copy()

    # Mg# (magnesium number) — proxy for magma evolution
    # MgO / (MgO + FeO) in molar proportions; higher = more primitive
    mgo = df["MgO(LOI free)"]
    feo = df["Fe2O3(LOI free)"] * 0.8998  # approx Fe2O3 → FeO conversion
    df["Mg_number"] = mgo / (mgo + feo + 1e-8)

    # Alumina saturation index (ASI) proxy — A/CNK
    # Al2O3 / (CaO + Na2O) in molecular proportions (K2O not available)
    df["ASI_proxy"] = df["Al2O3(LOI free)"] / (
        df["CaO(LOI free)"] + df["Na2O(LOI free)"] + 1e-8
    )

    # Total alkali (Na2O only, K2O not in dataset)
    df["Na2O_SiO2_ratio"] = df["Na2O(LOI free)"] / (df["SiO2(LOI free)"] + 1e-8)

    # LREE/HREE enrichment — key discriminator for tectonic setting
    df["LREE_sum"] = df["La"] + df["Ce"]
    df["HREE_sum"] = df["Sm"] + df["Yb"]
    df["LREE_HREE_ratio"] = df["LREE_sum"] / (df["HREE_sum"] + 1e-8)

    # Normalized LREE/HREE
    df["LaN_CeN_sum"] = df["LaN"] + df["CeN"]
    df["SmN_YbN_sum"] = df["SmN"] + df["YbN"]
    df["LREE_HREE_N_ratio"] = df["LaN_CeN_sum"] / (df["SmN_YbN_sum"] + 1e-8)

    # Nb anomaly — indicator of subduction influence
    df["Nb_anomaly"] = df["Nb"] / (np.sqrt(df["La"] * df["Ce"]) + 1e-8)

    # Ti/Nb ratio — discriminates OIB vs arc magmas
    df["Ti_Nb_ratio"] = df["Ti"] / (df["Nb"] + 1e-8)

    # Log transforms of trace elements (typically log-normal distributed)
    for col in ["Ti", "Rb", "Nb", "La", "Ce", "Sm", "Yb"]:
        df[f"log_{col}"] = np.log1p(df[col])

    # Silica classification bins (ultrabasic/basic/intermediate/acid)
    df["SiO2_bin"] = pd.cut(
        df["SiO2"],
        bins=[0, 45, 52, 63, 100],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Fe/Mg ratio — tholeiitic vs calc-alkaline
    df["Fe_Mg_ratio"] = df["Fe2O3(LOI free)"] / (df["MgO(LOI free)"] + 1e-8)

    # LOI = Total - sum of LOI-free oxides (if derivable)
    loi_free_sum = (
        df["SiO2(LOI free)"]
        + df["TiO2(LOI free)"]
        + df["Al2O3(LOI free)"]
        + df["Fe2O3(LOI free)"]
        + df["MgO(LOI free)"]
        + df["CaO(LOI free)"]
        + df["Na2O(LOI free)"]
    )
    df["LOI_free_total"] = loi_free_sum
    df["missing_oxide_pct"] = 100.0 - loi_free_sum  # K2O + P2O5 + MnO + LOI etc.

    # Interaction features between top discriminators
    df["Ti_x_Nb"] = df["Ti"] * df["Nb"]
    df["La_x_Ce"] = df["La"] * df["Ce"]
    df["SiO2_x_MgO"] = df["SiO2"] * df["MgO(LOI free)"]

    # REE slope — steepness of REE pattern
    df["REE_slope"] = (df["LaN"] - df["YbN"]) / (df["YbN"] + 1e-8)

    # Ce anomaly — redox indicator
    df["Ce_anomaly"] = df["CeN"] / (np.sqrt(df["LaN"] * df["SmN"]) + 1e-8)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all feature columns (excluding id and Label)."""
    return [c for c in df.columns if c not in ("Label", "id", "Id")]
