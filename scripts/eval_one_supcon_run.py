#!/usr/bin/env python3
"""
eval_one_supcon_run.py

Inline LGBM eval for a single SupCon-AE run.  Called at the end of each
array-job task so results are available immediately after training.

Usage:
  python scripts/eval_one_supcon_run.py \\
      --run-dir  <path/to/run_dir> \\
      --ann-csv  <path/to/annotation.csv> \\
      --fold-splits <path/to/fold_splits.csv> \\
      --fold     <int> \\
      --budget   <int|all> \\
      --repeat   <int>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

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


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",     required=True, type=Path)
    ap.add_argument("--ann-csv",     required=True, type=Path)
    ap.add_argument("--fold-splits", required=True, type=Path)
    ap.add_argument("--fold",        required=True, type=int)
    ap.add_argument("--budget",      required=True)
    ap.add_argument("--repeat",      required=True, type=int)
    args = ap.parse_args()

    latent_path = args.run_dir / "latents.csv"
    out_path    = args.run_dir / "lgbm_result.json"

    if not latent_path.exists():
        print(f"latents.csv not found in {args.run_dir}, skipping eval.")
        return

    # Detect latent dim from columns
    lat_df   = pd.read_csv(latent_path, nrows=1)
    z_cols   = [c for c in lat_df.columns if c.startswith("z_") and "_proj" not in c]
    latents  = pd.read_csv(latent_path, usecols=["filename"] + z_cols).set_index("filename")

    train_ann = pd.read_csv(args.ann_csv)
    train_ann["fn"] = train_ann["unique_ID"].apply(_h2u)

    fold_splits = pd.read_csv(args.fold_splits)
    test_ann    = fold_splits[fold_splits["fold"] == args.fold].copy()
    test_ann["fn"] = test_ann["unique_ID"].apply(_h2u)
    test_ann = test_ann[~test_ann["fn"].isin(set(train_ann["fn"]))]

    X_train = latents.reindex(train_ann["fn"]).values
    X_test  = latents.reindex(test_ann["fn"]).values

    train_mask = ~np.isnan(X_train).any(axis=1)
    test_mask  = ~np.isnan(X_test).any(axis=1)

    X_tr = X_train[train_mask]
    y_tr = train_ann["label"].values[train_mask]
    X_te = X_test[test_mask]
    y_te = test_ann["label"].values[test_mask]

    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        print("Insufficient data for eval.")
        return

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)

    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr_enc)
    y_pred   = clf.predict(X_te)
    bal_acc  = balanced_accuracy_score(y_te_enc, y_pred)

    result = dict(
        fold=args.fold, budget=args.budget, repeat=args.repeat,
        n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        bal_acc=round(bal_acc, 6),
    )
    out_path.write_text(json.dumps(result))
    print(f"  LGBM eval: fold={args.fold} budget={args.budget} repeat={args.repeat} "
          f"bal_acc={bal_acc:.4f}  ({int(train_mask.sum())} train / {int(test_mask.sum())} test)")


if __name__ == "__main__":
    main()
