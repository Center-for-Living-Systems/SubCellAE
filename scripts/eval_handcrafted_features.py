#!/usr/bin/env python3
"""
eval_handcrafted_features.py

Evaluate CellProfiler or Ilastik features using the exact same 5-fold CV
splits and label subsamples as the le_b2_supcon SupCon-AE benchmark.

For each job (fold × budget × repeat):
  - Training labels  : labelling/le_b2_supcon/{name}.csv  (unique_ID, label)
  - Test labels      : labelling/le_b2_supcon/fold_splits_{ds}.csv  (fold == current)
  - Features         : ae_results/features/{cp|ilastik}/{ds}.csv
  - Classifier       : LightGBM (binary, balanced)
  - Metric           : balanced accuracy

Output
------
  ae_results/features/eval_results/{feature_type}_{ds}.csv
  Columns: name, ds, fold, budget, repeat, n_train, n_test, bal_acc

Usage
-----
  python scripts/eval_handcrafted_features.py --feature cp --dataset ds1
  python scripts/eval_handcrafted_features.py --feature cp --dataset ds1 --feature-subset intensity
  python scripts/eval_handcrafted_features.py --feature cp --dataset ds1 --feature-subset texture
  python scripts/eval_handcrafted_features.py --feature ilastik --dataset ds2
  python scripts/eval_handcrafted_features.py --feature cp --all
  python scripts/eval_handcrafted_features.py --feature ilastik --all
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO     = Path(__file__).resolve().parents[1]
OUT_DIR  = DATA / "ae_results" / "features" / "eval_results"

FEAT_DIRS = {
    "cp":      DATA / "ae_results" / "features" / "cellprofiler",
    "ilastik": DATA / "ae_results" / "features" / "ilastik",
}

# CP feature subsets (intensity profile vs Haralick texture)
FEAT_SUBSETS = {
    "all":       None,          # use all columns
    "intensity": lambda cols: [c for c in cols if c.startswith("intensity_")],
    "texture":   lambda cols: [c for c in cols if not c.startswith("intensity_")],
}

LABEL_SET_CFG = {
    "b2":  {"ann_dir": DATA / "labelling" / "le_b2_supcon",
            "job_re":  re.compile(r"le_b2_(ds\d)_fv(\d)_nb(\w+)_r(\d)"),
            "prefix":  "le_b2"},
    "b12": {"ann_dir": DATA / "labelling" / "le_b12_supcon",
            "job_re":  re.compile(r"le_b12_(ds\d)_fv(\d)_nb(\w+)_r(\d)"),
            "prefix":  "le_b12"},
}

# defaults (overridden by --label-set in main)
ANN_DIR = LABEL_SET_CFG["b2"]["ann_dir"]
JOB_RE  = LABEL_SET_CFG["b2"]["job_re"]

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


def _hyph_to_under(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def load_features(feat_dir: Path, ds: str, subset: str = "all") -> pd.DataFrame:
    df = pd.read_csv(feat_dir / f"{ds}.csv")
    df["filename"] = df["filename"].apply(lambda x: x if "_f" in x else x)
    df = df.set_index("filename")
    fn = FEAT_SUBSETS.get(subset)
    if fn is not None:
        keep = fn(df.columns.tolist())
        df = df[keep]
    return df


def eval_job(name: str, ds: str, fold: int, budget: str, repeat: int,
             feat_df: pd.DataFrame, fold_splits: pd.DataFrame,
             ann_dir: Path = None, classifier: str = "lgbm") -> dict | None:
    ann_path = (ann_dir or ANN_DIR) / f"{name}.csv"
    if not ann_path.exists():
        return None

    # Training labels (hyphen → underscore for feature lookup)
    train_ann = pd.read_csv(ann_path)
    train_ann["fn"] = train_ann["unique_ID"].apply(_hyph_to_under)

    # Test labels: patches in this fold, not in training set
    test_ann = fold_splits[fold_splits["fold"] == fold].copy()
    test_ann["fn"] = test_ann["unique_ID"].apply(_hyph_to_under)
    train_fns = set(train_ann["fn"])
    test_ann  = test_ann[~test_ann["fn"].isin(train_fns)]

    # Look up features
    train_feats = feat_df.reindex(train_ann["fn"])
    test_feats  = feat_df.reindex(test_ann["fn"])

    train_mask = train_feats.notna().all(axis=1)
    test_mask  = test_feats.notna().all(axis=1)

    X_train = train_feats[train_mask].values
    y_train = train_ann.loc[train_mask.values, "label"].values
    X_test  = test_feats[test_mask].values
    y_test  = test_ann.loc[test_mask.values, "label"].values

    if len(X_train) == 0 or len(X_test) == 0:
        return None
    if len(np.unique(y_train)) < 2:
        return None

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    if classifier == "logreg":
        sc = StandardScaler().fit(X_train)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(sc.transform(X_train), y_train_enc)
        y_pred = clf.predict(sc.transform(X_test))
    else:
        clf = LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_train, y_train_enc)
        y_pred = clf.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test_enc, y_pred)

    return dict(
        name=name, ds=ds, fold=fold, budget=budget, repeat=repeat,
        n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        bal_acc=round(bal_acc, 6),
    )


def process_dataset(ds: str, feat_type: str,
                    ann_dir: Path, job_re: re.Pattern,
                    label_set: str, classifier: str = "lgbm",
                    feat_subset: str = "all") -> pd.DataFrame:
    feat_dir  = FEAT_DIRS[feat_type]
    cfg_dir   = REPO / "config" / f"le_{label_set}_supcon"
    job_list  = cfg_dir / f"job_list_{ds}.txt"
    if not job_list.exists():
        all_jobs = (cfg_dir / "job_list.txt").read_text().splitlines()
        jobs = [j for j in all_jobs if f"_{ds}_" in j]
    else:
        jobs = job_list.read_text().splitlines()

    fold_splits = pd.read_csv(ann_dir / f"fold_splits_{ds}.csv")

    print(f"\n{ds} | {feat_type} | label_set={label_set} | subset={feat_subset}: loading features ...", flush=True)
    feat_df = load_features(feat_dir, ds, feat_subset)
    print(f"  features: {feat_df.shape}  ({feat_df.isna().any(axis=1).sum()} rows with NaN)")

    results = []
    for i, job_path in enumerate(jobs):
        name = Path(job_path).stem
        m    = job_re.search(name)
        if not m:
            continue
        _, fold, budget, repeat = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))

        if i % 50 == 0:
            print(f"  {i}/{len(jobs)} ...", flush=True)

        row = eval_job(name, ds, fold, budget, repeat, feat_df, fold_splits, ann_dir, classifier)
        if row:
            results.append(row)

    df = pd.DataFrame(results)
    if len(df):
        clf_tag    = f"_{classifier}" if classifier != "lgbm" else ""
        subset_tag = f"_{feat_subset}" if feat_subset != "all" else ""
        out = OUT_DIR / f"{feat_type}_{label_set}{clf_tag}{subset_tag}_{ds}.csv"
        df.to_csv(out, index=False)
        print(f"  Saved {len(df)} rows → {out}")
        numeric_budgets = df[df["budget"] != "all"].copy()
        numeric_budgets["budget_int"] = numeric_budgets["budget"].astype(int)
        summary = (numeric_budgets.groupby("budget_int")["bal_acc"]
                   .agg(["mean", "std"]).round(4))
        print(f"\n  Budget summary (mean balanced acc across folds/repeats):\n{summary}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature",        choices=["cp", "ilastik"], required=True)
    ap.add_argument("--label-set",      choices=["b2", "b12"], default="b2",
                    help="b2=Annabel DS1 only, b12=B1+B2 combined")
    ap.add_argument("--classifier",     choices=["lgbm", "logreg"], default="lgbm")
    ap.add_argument("--feature-subset", choices=["all", "intensity", "texture"],
                    default="all", help="CP feature subset: all / intensity / texture")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset", choices=["ds1", "ds2", "ds3"])
    grp.add_argument("--all", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ls_cfg   = LABEL_SET_CFG[args.label_set]
    ann_dir  = ls_cfg["ann_dir"]
    job_re   = ls_cfg["job_re"]
    label_set = args.label_set

    datasets = ["ds1"] if args.label_set == "b12" else (["ds1", "ds2", "ds3"] if args.all else [args.dataset])
    if not args.all and args.dataset:
        datasets = [args.dataset]

    for ds in datasets:
        process_dataset(ds, args.feature, ann_dir, job_re, label_set,
                        args.classifier, args.feature_subset)

    print("\nDone.")


if __name__ == "__main__":
    main()
