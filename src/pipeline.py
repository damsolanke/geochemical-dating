"""Geochemical Dating — final solution pipeline.

End-to-end reproduction of the 2nd-place submission (private Macro-F1 0.96988).

The prediction is a 50/50 arithmetic blend of two components:

  1. Id-KNN (50%) -- exploits that the sample ``Id`` tracks the source
     database's row order, so contiguous ``Id`` runs share an age class
     (see ``src/idknn.py``).
  2. Model ensemble (50%) -- a weighted blend of three classifiers trained on
     domain geochemical features plus KMeans cluster one-hot encodings, with
     cluster-frequency sample weighting, 15-fold CV averaged over 5 seeds:

         0.52 * XGBoost  +  0.09 * LightGBM  +  0.39 * SVM(RBF)

Exact train/test duplicate rows are then overridden with their known labels.

Usage:
    python src/pipeline.py --tag final

Requires the competition CSVs in ``data/`` (not distributed with this repo --
see the README for how to obtain them from Kaggle).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import add_geochemical_features, get_feature_cols  # noqa: E402
from idknn import id_knn_proba  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
LOGS = ROOT / "logs"

N_CLASSES = 3

# Final model-ensemble weights (XGBoost / LightGBM / SVM).
MODEL_WEIGHTS = {"xgb": 0.52, "lgbm": 0.09, "svm": 0.39}
# Final blend weight on the Id-KNN component (model ensemble gets 1 - this).
KNN_WEIGHT = 0.50

LGBM_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
    "boosting_type": "gbdt", "num_leaves": 127, "learning_rate": 0.05,
    "min_child_samples": 8, "max_depth": 8, "subsample": 0.8,
    "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "verbosity": -1,
}

XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES, "eval_metric": "mlogloss",
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "min_child_weight": 8, "tree_method": "hist", "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Data + features
# ---------------------------------------------------------------------------

def load_data():
    train = pd.read_csv(DATA / "train.csv").rename(columns={"Id": "id"})
    test = pd.read_csv(DATA / "test.csv").rename(columns={"Id": "id"})
    train = add_geochemical_features(train)
    test = add_geochemical_features(test)
    return train, test, get_feature_cols(train)


def add_cluster_onehot(train, test, feature_cols):
    """Append KMeans cluster-membership one-hot columns at several granularities."""
    scaler = StandardScaler()
    all_feats = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
    scaled = scaler.fit_transform(all_feats)
    new_cols = []
    for k in [8, 16, 32, 64, 128]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(scaled)
        onehot = pd.get_dummies(clusters, prefix=f"clust_{k}").astype(np.float32)
        new_cols.append(onehot)
    cluster_df = pd.concat(new_cols, axis=1)
    n_train = len(train)
    train_new = pd.concat(
        [train.reset_index(drop=True), cluster_df.iloc[:n_train].reset_index(drop=True)],
        axis=1,
    )
    test_new = pd.concat(
        [test.reset_index(drop=True), cluster_df.iloc[n_train:].reset_index(drop=True)],
        axis=1,
    )
    return train_new, test_new, list(cluster_df.columns)


def compute_cluster_weights(train, test, feature_cols, n_clusters=80):
    """Weight train rows by how common their cluster is in the test set."""
    scaler = StandardScaler()
    all_feats = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
    scaled = scaler.fit_transform(all_feats)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(scaled)
    train_clusters = clusters[: len(train)]
    test_clusters = clusters[len(train):]
    test_counts = Counter(test_clusters)
    return np.array([test_counts.get(c, 0) + 0.1 for c in train_clusters])


def find_duplicate_labels(train, test):
    """Test rows that exactly match a train row on raw features -> free labels."""
    raw_cols = [c for c in pd.read_csv(DATA / "train.csv").columns if c not in ("Label", "Id")]
    avail_cols = [c for c in raw_cols if c in train.columns and c in test.columns]
    merged = test.merge(train[avail_cols + ["Label"]], on=avail_cols, how="inner")
    return dict(zip(merged["id"], merged["Label"]))


# ---------------------------------------------------------------------------
# Model ensemble (XGBoost + LightGBM + SVM), cluster-weighted, multi-seed CV
# ---------------------------------------------------------------------------

def run_full_cv(train_df, test_df, all_features, n_folds=15, seeds=(42, 2024, 7, 99, 123)):
    """Return seed-averaged OOF and test probabilities for each base model."""
    X = train_df[all_features].values
    y = train_df["Label"].values
    X_test = test_df[all_features].values
    n_train, n_test = len(X), len(X_test)
    n_seeds = len(seeds)

    oof = {m: np.zeros((n_train, N_CLASSES)) for m in ("xgb", "lgbm", "svm")}
    test_p = {m: np.zeros((n_test, N_CLASSES)) for m in ("xgb", "lgbm", "svm")}

    weights = compute_cluster_weights(train_df, test_df, all_features)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_test_sc = scaler.transform(X_test)

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, vl_idx in skf.split(X, y):
            X_tr, X_vl = X[tr_idx], X[vl_idx]
            y_tr, y_vl = y[tr_idx], y[vl_idx]
            w_tr = weights[tr_idx]

            # XGBoost
            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
            dval = xgb.DMatrix(X_vl, label=y_vl)
            m = xgb.train(
                {**XGB_PARAMS, "random_state": seed}, dtrain, num_boost_round=2000,
                evals=[(dval, "val")], early_stopping_rounds=200, verbose_eval=False,
            )
            oof["xgb"][vl_idx] += m.predict(dval) / n_seeds
            test_p["xgb"] += m.predict(xgb.DMatrix(X_test)) / (n_folds * n_seeds)

            # LightGBM
            dt = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=list(all_features))
            dv = lgb.Dataset(X_vl, label=y_vl, feature_name=list(all_features), reference=dt)
            m_l = lgb.train(
                {**LGBM_PARAMS, "random_state": seed}, dt, num_boost_round=2000,
                valid_sets=[dv],
                callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
            )
            oof["lgbm"][vl_idx] += m_l.predict(X_vl) / n_seeds
            test_p["lgbm"] += m_l.predict(X_test) / (n_folds * n_seeds)

            # SVM (RBF) on standardised features
            svm = SVC(kernel="rbf", C=50, gamma="scale", probability=True, random_state=seed)
            svm.fit(X_sc[tr_idx], y_tr, sample_weight=w_tr)
            oof["svm"][vl_idx] += svm.predict_proba(X_sc[vl_idx]) / n_seeds
            test_p["svm"] += svm.predict_proba(X_test_sc) / (n_folds * n_seeds)

        print(f"  seed {seed} done", flush=True)

    return oof, test_p


def blend(parts, weights):
    """Weighted sum of probability matrices keyed by model name."""
    return sum(weights[m] * parts[m] for m in weights)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-folds", type=int, default=15, help="CV folds for the model ensemble")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 7, 99, 123])
    ap.add_argument("--knn-folds", type=int, default=10, help="CV folds for Id-KNN OOF")
    ap.add_argument("--tag", default="final", help="submission filename suffix")
    args = ap.parse_args()

    print("Loading data and engineering features...", flush=True)
    train_df, test_df, feature_cols = load_data()
    train_df, test_df, cluster_cols = add_cluster_onehot(train_df, test_df, feature_cols)
    all_features = feature_cols + cluster_cols + ["id"]
    print(f"  train={len(train_df)} test={len(test_df)} features={len(all_features)}", flush=True)

    ids_train = train_df["id"].values
    ids_test = test_df["id"].values
    y = train_df["Label"].values

    dup_map = find_duplicate_labels(train_df, test_df)
    print(f"  exact train/test duplicates: {len(dup_map)}", flush=True)

    # --- Component 1: Id-KNN (OOF via CV, test on full train) ---
    print("Id-KNN...", flush=True)
    id_knn_oof = np.zeros((len(ids_train), N_CLASSES))
    skf = StratifiedKFold(n_splits=args.knn_folds, shuffle=True, random_state=42)
    for tr_idx, vl_idx in skf.split(ids_train, y):
        id_knn_oof[vl_idx] = id_knn_proba(ids_train[tr_idx], y[tr_idx], ids_train[vl_idx])
    id_knn_test = id_knn_proba(ids_train, y, ids_test)
    print(f"  Id-KNN OOF Macro-F1: {f1_score(y, id_knn_oof.argmax(1), average='macro'):.4f}", flush=True)

    # --- Component 2: model ensemble (XGB + LGBM + SVM) ---
    print("Model ensemble (XGBoost + LightGBM + SVM)...", flush=True)
    oof, test_p = run_full_cv(train_df, test_df, all_features, n_folds=args.n_folds, seeds=args.seeds)
    model_oof = blend(oof, MODEL_WEIGHTS)
    model_test = blend(test_p, MODEL_WEIGHTS)
    print(f"  model OOF Macro-F1: {f1_score(y, model_oof.argmax(1), average='macro'):.4f}", flush=True)

    # --- Final 50/50 blend ---
    blend_oof = (1 - KNN_WEIGHT) * model_oof + KNN_WEIGHT * id_knn_oof
    blend_test = (1 - KNN_WEIGHT) * model_test + KNN_WEIGHT * id_knn_test
    oof_f1 = f1_score(y, blend_oof.argmax(1), average="macro")
    print(f"  blended OOF Macro-F1: {oof_f1:.4f}", flush=True)

    labels = blend_test.argmax(1).copy()

    # Override exact duplicates with their known labels.
    pos_by_id = {tid: i for i, tid in enumerate(ids_test)}
    overrides = 0
    for tid, lab in dup_map.items():
        if tid in pos_by_id:
            labels[pos_by_id[tid]] = lab
            overrides += 1
    print(f"  applied {overrides} duplicate overrides", flush=True)

    SUBMISSIONS.mkdir(exist_ok=True)
    sub_path = SUBMISSIONS / f"submission_{args.tag}.csv"
    pd.DataFrame({"id": ids_test, "Label": labels}).to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}", flush=True)

    LOGS.mkdir(exist_ok=True)
    metrics = {
        "tag": args.tag,
        "n_folds": args.n_folds,
        "seeds": list(args.seeds),
        "model_weights": MODEL_WEIGHTS,
        "knn_weight": KNN_WEIGHT,
        "blended_oof_macro_f1": float(oof_f1),
        "n_duplicate_overrides": overrides,
    }
    with open(LOGS / f"metrics_{args.tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
