"""End-to-end smoke test for src/pipeline.py on a small synthetic dataset.

The dataset uses the competition's exact column names (33 raw features plus
``Id``/``Label``), ~600 train / ~200 test rows, a few exact train/test
duplicates, and labels that follow contiguous ``Id`` runs so both branches of
the blend have signal. The pipeline is driven through ``main(argv)`` with fast
settings (2 folds, 1 seed, 30 boosting rounds); no network, no Kaggle data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline  # noqa: E402

RAW_COLS = [
    "SiO2", "Al2O3", "Fe2O3tot", "CaO", "Total",
    "Ti", "Rb", "Nb", "La", "Ce", "Sm", "Yb",
    "LaN", "CeN", "SmN", "YbN",
    "La/Nb", "Rb/Sm", "Nb/La", "Nb/Yb", "Nb/Rb", "La/Sm", "La/Yb", "(La/Yb)N",
    "Sm/Yb", "(Sm/Yb)N",
    "SiO2(LOI free)", "TiO2(LOI free)", "Al2O3(LOI free)", "Fe2O3(LOI free)",
    "MgO(LOI free)", "CaO(LOI free)", "Na2O(LOI free)",
]
N_TRAIN, N_TEST, N_DUPS = 600, 200, 5
FAST_ARGS = ["--n-folds", "2", "--seeds", "1", "--knn-folds", "2", "--n-rounds", "30"]
METRIC_KEYS = {
    "idknn_oof_macro_f1", "ensemble_oof_macro_f1", "blended_oof_macro_f1",
    "id_in_ensemble", "n_features", "n_duplicate_overrides",
}


def make_synthetic(out_dir, n_train=N_TRAIN, n_test=N_TEST, n_dups=N_DUPS, seed=0):
    """Write train.csv / test.csv / sample_submission.csv with the competition schema."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    ids = np.arange(1, n + 1)
    labels = (ids // 120) % 3  # contiguous Id runs share a class (the Id-KNN signal)

    sio2 = np.clip(44 + 9 * labels + rng.normal(0, 3, n), 38, 78)
    cols = {
        "SiO2": sio2,
        "Al2O3": np.clip(15 + rng.normal(0, 1.5, n), 8, 22),
        "Fe2O3tot": np.clip(11 - 0.12 * (sio2 - 44) + rng.normal(0, 1, n), 1, 16),
        "CaO": np.clip(10 - 0.15 * (sio2 - 44) + rng.normal(0, 1, n), 0.5, 14),
        "Total": np.clip(99.5 + rng.normal(0, 0.4, n), 97, 101),
    }
    for el, base in (("Ti", 6000), ("Rb", 40), ("Nb", 12), ("La", 25), ("Ce", 50), ("Sm", 5), ("Yb", 2.2)):
        cols[el] = base * np.exp(rng.normal(0.15 * labels, 0.4, n))
    for el, chondrite in (("La", 0.237), ("Ce", 0.612), ("Sm", 0.153), ("Yb", 0.170)):
        cols[f"{el}N"] = cols[el] / chondrite
    for ratio in ("La/Nb", "Rb/Sm", "Nb/La", "Nb/Yb", "Nb/Rb", "La/Sm", "La/Yb", "Sm/Yb"):
        num, den = ratio.split("/")
        cols[ratio] = cols[num] / cols[den]
    cols["(La/Yb)N"] = cols["LaN"] / cols["YbN"]
    cols["(Sm/Yb)N"] = cols["SmN"] / cols["YbN"]
    cols["SiO2(LOI free)"] = sio2 + 1.0
    cols["TiO2(LOI free)"] = cols["Ti"] / 6000 * 1.2
    cols["Al2O3(LOI free)"] = cols["Al2O3"] + 0.5
    cols["Fe2O3(LOI free)"] = cols["Fe2O3tot"] + 0.3
    cols["MgO(LOI free)"] = np.clip(9 - 0.2 * (sio2 - 44) + rng.normal(0, 1, n), 0.3, 20)
    cols["CaO(LOI free)"] = cols["CaO"] + 0.3
    cols["Na2O(LOI free)"] = np.clip(3 + 0.05 * (sio2 - 44) + rng.normal(0, 0.4, n), 0.5, 8)

    df = pd.DataFrame(cols)[RAW_COLS].round(4)
    df.insert(0, "Id", ids)
    perm = rng.permutation(n)
    tr_idx, te_idx = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    train = df.iloc[tr_idx].copy()
    train["Label"] = labels[tr_idx]
    test = df.iloc[te_idx].copy()
    # Exact copies of train rows (under test Ids) exercise the duplicate override.
    test.iloc[:n_dups, 1:] = train.iloc[:n_dups][RAW_COLS].values
    dup_labels = dict(zip(test["Id"].iloc[:n_dups], train["Label"].iloc[:n_dups]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    pd.DataFrame({"Id": test["Id"], "Label": 0}).to_csv(out_dir / "sample_submission.csv", index=False)
    return {"train": train, "test": test, "dup_labels": dup_labels}


def run_pipeline(data_dir, out_dir, tag, extra_args=()):
    """Run main() in-process, capturing the feature list handed to the ensemble."""
    seen = {}
    original = pipeline.run_full_cv

    def spy(train_df, test_df, all_features, **kwargs):
        seen["features"] = list(all_features)
        return original(train_df, test_df, all_features, **kwargs)

    pipeline.run_full_cv = spy
    try:
        argv = ["--tag", tag, "--data-dir", str(data_dir), "--out-dir", str(out_dir), *FAST_ARGS, *extra_args]
        metrics = pipeline.main(argv)
    finally:
        pipeline.run_full_cv = original
    return {
        "metrics": metrics,
        "features": seen["features"],
        "submission": Path(out_dir) / "submissions" / f"submission_{tag}.csv",
        "metrics_path": Path(out_dir) / "logs" / f"metrics_{tag}.json",
    }


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    info = make_synthetic(data_dir)
    info["data_dir"] = data_dir
    return info


@pytest.fixture(scope="module")
def leakfree_run(synthetic, tmp_path_factory):
    return run_pipeline(synthetic["data_dir"], tmp_path_factory.mktemp("out_leakfree"), "leakfree")


@pytest.fixture(scope="module")
def competition_run(synthetic, tmp_path_factory):
    return run_pipeline(
        synthetic["data_dir"], tmp_path_factory.mktemp("out_final"), "final", ["--id-in-ensemble"]
    )


def test_submission_uses_competition_columns_and_duplicate_overrides(synthetic, leakfree_run):
    sub = pd.read_csv(leakfree_run["submission"])
    assert list(sub.columns) == ["Id", "Label"]
    assert len(sub) == N_TEST
    np.testing.assert_array_equal(sub["Id"].values, synthetic["test"]["Id"].values)
    assert set(sub["Label"].unique()) <= {0, 1, 2}
    # Exact train/test duplicates must carry their known train label.
    by_id = dict(zip(sub["Id"], sub["Label"]))
    assert all(by_id[i] == lab for i, lab in synthetic["dup_labels"].items())


def test_metrics_json_records_scores_and_override_count(leakfree_run):
    assert leakfree_run["metrics_path"].exists()
    metrics = json.loads(leakfree_run["metrics_path"].read_text())
    assert METRIC_KEYS <= set(metrics)
    assert metrics == leakfree_run["metrics"]
    assert metrics["id_in_ensemble"] is False
    assert metrics["n_duplicate_overrides"] == N_DUPS
    assert metrics["n_features"] == len(leakfree_run["features"])
    for key in ("idknn_oof_macro_f1", "ensemble_oof_macro_f1", "blended_oof_macro_f1"):
        assert 0.0 <= metrics[key] <= 1.0


def test_id_reaches_the_ensemble_only_with_the_flag(leakfree_run, competition_run):
    assert "id" not in leakfree_run["features"]
    assert "Id" not in leakfree_run["features"]
    assert "id" in competition_run["features"]
    assert competition_run["metrics"]["id_in_ensemble"] is True
    assert competition_run["metrics"]["n_features"] == leakfree_run["metrics"]["n_features"] + 1
    assert list(pd.read_csv(competition_run["submission"]).columns) == ["Id", "Label"]
