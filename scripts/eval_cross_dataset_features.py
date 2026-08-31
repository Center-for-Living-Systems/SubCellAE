#!/usr/bin/env python3
"""
eval_cross_dataset_features.py

Cross-dataset generalization: for each DS1 label-efficiency job (fold × budget × repeat),
train LightGBM on DS1 features with DS1 labels, then evaluate on all labeled patches
from DS2 and DS3.

The feature space is shared (same 50 CP or 80 ilastik features), so the trained
model can be applied directly across datasets.

Usage
-----
  python scripts/eval_cross_dataset_features.py --feature cp
  python scripts/eval_cross_dataset_features.py --feature ilastik

Output
------
  ae_results/features/eval_results/{feature_type}_cross_ds.csv
  Columns: name, ds_train, ds_test, fold, budget, repeat, n_train, n_test, bal_acc
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO    = Path(__file__).resolve().parents[1]
ANN_DIR = DATA / "labelling" / "le_b2_supcon"
OUT_DIR = DATA / "ae_results" / "features" / "eval_results"

FEAT_DIRS = {
    "cp":      DATA / "ae_results" / "features" / "cellprofiler",
    "ilastik": DATA / "ae_results" / "features" / "ilastik",
}

JOB_RE = re.compile(r"le_b2_(ds\d)_fv(\d)_nb(\w+)_r(\d)")

LGBM_PARAMS = dict(
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=1,
    class_weight="balanced",
    n_jobs=4,
    random_state=42,
    verbose=-1,
)

TEST_DATASETS = ["ds2", "ds3"]


def _hyph_to_under(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def load_features(feat_dir: Path, ds: str) -> pd.DataFrame:
    df = pd.read_csv(feat_dir / f"{ds}.csv")
    return df.set_index("filename")


def eval_job(name: str, fold: int, budget: str, repeat: int,
             train_feat_df: pd.DataFrame,
             test_feat_dfs: dict[str, pd.DataFrame],
             test_all: dict[str, pd.DataFrame]) -> list[dict]:
    ann_path = ANN_DIR / f"{name}.csv"
    if not ann_path.exists():
        return []

    train_ann = pd.read_csv(ann_path)
    train_ann["fn"] = train_ann["unique_ID"].apply(_hyph_to_under)

    train_feats = train_feat_df.reindex(train_ann["fn"])
    train_mask  = train_feats.notna().all(axis=1)
    X_train = train_feats[train_mask].values
    y_train = train_ann.loc[train_mask.values, "label"].values

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return []

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_train, y_train_enc)

    results = []
    for ds_test in TEST_DATASETS:
        test_ann = test_all[ds_test].copy()
        test_ann["fn"] = test_ann["unique_ID"].apply(_hyph_to_under)

        test_feats = test_feat_dfs[ds_test].reindex(test_ann["fn"])
        test_mask  = test_feats.notna().all(axis=1)
        X_test = test_feats[test_mask].values
        y_test = test_ann.loc[test_mask.values, "label"].values

        if len(X_test) == 0 or len(np.unique(y_test)) < 2:
            continue

        try:
            y_test_enc = le.transform(y_test)
        except ValueError:
            continue

        y_pred  = clf.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test_enc, y_pred)

        results.append(dict(
            name=name, ds_train="ds1", ds_test=ds_test,
            fold=fold, budget=budget, repeat=repeat,
            n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
            bal_acc=round(bal_acc, 6),
        ))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["cp", "ilastik"], required=True)
    args = ap.parse_args()

    feat_dir = FEAT_DIRS[args.feature]

    # DS1 job list
    job_list_path = REPO / "config" / "le_b2_supcon" / "job_list_ds1.txt"
    if not job_list_path.exists():
        main_list = REPO / "config" / "le_b2_supcon" / "job_list.txt"
        all_jobs  = main_list.read_text().splitlines()
        jobs = [j for j in all_jobs if "_b2_ds1_" in j]
    else:
        jobs = job_list_path.read_text().splitlines()

    print(f"Cross-dataset eval | feature={args.feature} | {len(jobs)} DS1 jobs", flush=True)

    print("Loading features ...", flush=True)
    train_feat_df = load_features(feat_dir, "ds1")
    test_feat_dfs = {ds: load_features(feat_dir, ds) for ds in TEST_DATASETS}
    print(f"  DS1 features: {train_feat_df.shape}")
    for ds, df in test_feat_dfs.items():
        print(f"  {ds} features:  {df.shape}")

    # Load all labeled patches for each test dataset (all folds combined)
    test_all = {}
    for ds in TEST_DATASETS:
        splits = pd.read_csv(ANN_DIR / f"fold_splits_{ds}.csv")
        test_all[ds] = splits
        adh  = (splits["label"] == "adhesion").sum()
        noad = (splits["label"] == "No adhesion").sum()
        print(f"  {ds} test set: {len(splits)} patches  (adh={adh} noad={noad})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, job_path in enumerate(jobs):
        name = Path(job_path).stem
        m    = JOB_RE.search(name)
        if not m:
            continue
        _, fold, budget, repeat = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))

        if i % 50 == 0:
            print(f"  {i}/{len(jobs)} ...", flush=True)

        rows = eval_job(name, fold, budget, repeat,
                        train_feat_df, test_feat_dfs, test_all)
        all_results.extend(rows)

    df = pd.DataFrame(all_results)
    if len(df):
        out = OUT_DIR / f"{args.feature}_cross_ds.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows → {out}")

        numeric = df[df["budget"] != "all"].copy()
        numeric["budget_int"] = numeric["budget"].astype(int)
        summary = (numeric.groupby(["ds_test", "budget_int"])["bal_acc"]
                   .agg(["mean", "std"]).round(4))
        print(f"\nBudget summary (mean balanced acc):\n{summary}")
    else:
        print("No results.")

    print("\nDone.")


if __name__ == "__main__":
    main()
