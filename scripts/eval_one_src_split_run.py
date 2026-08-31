#!/usr/bin/env python3
"""
eval_one_src_split_run.py

Per-condition LGBM evaluation for a single le_b2_src_split run.
Reports balanced accuracy separately for:
  - combined (ctrl + ycomp) test set
  - ctrl-only test set
  - ycomp-only test set

The LGBM is always trained on the variant training labels (ctrl/mix/ycomp).
Test labels come from the fold_splits_ds1.csv (same as other LE benchmarks).

Usage:
  python scripts/eval_one_src_split_run.py \\
      --run-dir   .../contrastive_run/le_b2_src_split/le_b2_src_ctrl_fv0_r0 \\
      --ann-csv   .../labelling/le_b2_src_split/le_b2_src_ctrl_fv0_r0.csv \\
      --fold-splits .../labelling/le_b2_src_split/fold_splits_ds1.csv \\
      --fold 0 --variant ctrl --repeat 0
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


def run_lgbm(X_tr, y_tr, X_te, y_te, le):
    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return None
    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return round(balanced_accuracy_score(y_te, y_pred), 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",     required=True)
    ap.add_argument("--ann-csv",     required=True, help="training annotation CSV")
    ap.add_argument("--fold-splits", required=True, help="fold_splits_ds1.csv")
    ap.add_argument("--fold",        type=int, required=True)
    ap.add_argument("--variant",     required=True, choices=["ctrl", "mix", "ycomp"])
    ap.add_argument("--repeat",      type=int, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    latent_path = run_dir / "latents.csv"
    if not latent_path.exists():
        print(f"ERROR: latents.csv not found at {latent_path}")
        return

    # ---- load latents ----
    lat = pd.read_csv(latent_path)
    lat["fn"] = lat["filename"].apply(_h2u)
    z_cols = [c for c in lat.columns if c.startswith("z_")]
    lat_idx = lat.set_index("fn")

    # ---- training labels ----
    train_ann = pd.read_csv(args.ann_csv)
    train_ann["fn"] = train_ann["unique_ID"].apply(_h2u)

    # ---- test labels from fold splits ----
    fold_splits = pd.read_csv(args.fold_splits)
    fold_splits["fn"] = fold_splits["unique_ID"].apply(_h2u)
    test_all = fold_splits[fold_splits["fold"] == args.fold].copy()
    train_fns = set(train_ann["fn"])
    test_all  = test_all[~test_all["fn"].isin(train_fns)]

    # source label from unique_ID
    test_all["src"] = test_all["unique_ID"].apply(
        lambda x: "ctrl" if x.startswith("control") else "ycomp"
    )
    test_ctrl  = test_all[test_all["src"] == "ctrl"]
    test_ycomp = test_all[test_all["src"] == "ycomp"]

    # ---- build feature matrices ----
    def _feats(ann_df):
        feats = lat_idx[z_cols].reindex(ann_df["fn"])
        mask  = feats.notna().all(axis=1)
        X = feats[mask].values
        y = ann_df.loc[mask.values, "label"].values
        return X, y

    X_tr, y_tr_raw = _feats(train_ann)
    X_te_all,   y_te_all   = _feats(test_all)
    X_te_ctrl,  y_te_ctrl  = _feats(test_ctrl)
    X_te_ycomp, y_te_ycomp = _feats(test_ycomp)

    le = LabelEncoder()
    le.fit(["No adhesion", "adhesion"])
    y_tr = le.transform(y_tr_raw)

    def _score(X_te, y_te_raw):
        if len(X_te) == 0 or len(np.unique(y_te_raw)) < 2:
            return None
        return run_lgbm(X_tr, y_tr, X_te, le.transform(y_te_raw), le)

    bal_acc_all   = _score(X_te_all,   y_te_all)
    bal_acc_ctrl  = _score(X_te_ctrl,  y_te_ctrl)
    bal_acc_ycomp = _score(X_te_ycomp, y_te_ycomp)

    result = dict(
        variant=args.variant,
        fold=args.fold,
        repeat=args.repeat,
        n_train=len(X_tr),
        n_test_all=len(X_te_all),
        n_test_ctrl=len(X_te_ctrl),
        n_test_ycomp=len(X_te_ycomp),
        bal_acc_all=bal_acc_all,
        bal_acc_ctrl=bal_acc_ctrl,
        bal_acc_ycomp=bal_acc_ycomp,
    )

    out_path = run_dir / "lgbm_result_src.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
