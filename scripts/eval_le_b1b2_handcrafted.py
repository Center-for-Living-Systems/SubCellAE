#!/usr/bin/env python3
"""
eval_le_b1b2_handcrafted.py

Evaluate CellProfiler and ilastik features on the B1/B2 matched
label-efficiency benchmark (both batches, all budgets, 5-fold CV).

Reads job lists from config/le_b1b2_{batch}_lat12p8/job_list.txt,
annotation CSVs from labelling/le_b1b2_le/,
fold splits from labelling/le_b1b2_matched/fold_splits_{batch}.csv.

Outputs (EVAL_DIR):
  cp_b1b2_b1_ds1.csv,  cp_b1b2_b2_ds1.csv
  il_b1b2_b1_ds1.csv,  il_b1b2_b2_ds1.csv
  (logreg variants with _logreg_ infix if --classifier logreg)

Columns: name, batch, fold, budget, repeat, n_train, n_test, bal_acc

Usage
-----
  python scripts/eval_le_b1b2_handcrafted.py
  python scripts/eval_le_b1b2_handcrafted.py --classifier logreg
"""
from __future__ import annotations

import argparse
import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO     = Path(__file__).resolve().parents[1]
EVAL_DIR = DATA / "ae_results" / "features" / "eval_results"
ANN_DIR  = DATA / "labelling" / "le_b1b2_le"
FS_DIR   = DATA / "labelling" / "le_b1b2_matched"

CP_CSV  = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
IL_CSV  = DATA / "ae_results" / "features" / "ilastik" / "ds1.csv"

JOB_RE = re.compile(r"le_b1b2_(b\d)_lat\w+_fv(\d)_nb(\w+)_r(\d)")

LGBM_PARAMS = dict(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    min_child_samples=1, class_weight="balanced",
    n_jobs=4, random_state=42, verbose=-1,
)


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _eval_one(name, batch, fold, budget, repeat, feat_df, fold_splits, classifier):
    ann_name = f"le_b1b2_{batch}_fv{fold}_nb{budget}_r{repeat}"
    ann_path = ANN_DIR / f"{ann_name}.csv"
    if not ann_path.exists():
        return None

    train_ann = pd.read_csv(ann_path)
    train_ann["fn"] = train_ann["unique_ID"].apply(_h2u)

    test_fs = fold_splits[fold_splits["fold"] == fold].copy()
    test_fs["fn"] = test_fs["unique_ID"].apply(_h2u)
    train_fns = set(train_ann["fn"])
    test_fs = test_fs[~test_fs["fn"].isin(train_fns)]

    X_tr_raw = feat_df.reindex(train_ann["fn"]).values
    X_te_raw = feat_df.reindex(test_fs["fn"]).values

    tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
    te_ok = ~np.isnan(X_te_raw).any(axis=1)

    X_tr = X_tr_raw[tr_ok]
    X_te = X_te_raw[te_ok]
    y_tr = train_ann["label"].values[tr_ok]
    y_te = test_fs["label"].values[te_ok]

    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return None

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)

    if classifier == "logreg":
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(sc.transform(X_tr), y_tr_enc)
        y_pred = clf.predict(sc.transform(X_te))
    else:
        clf = LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_tr, y_tr_enc)
        y_pred = clf.predict(X_te)

    return dict(
        name=name, batch=batch, fold=fold, budget=budget, repeat=repeat,
        n_train=int(tr_ok.sum()), n_test=int(te_ok.sum()),
        bal_acc=round(balanced_accuracy_score(y_te_enc, y_pred), 6),
    )


def _run_batch(batch, feat_df, classifier):
    cfg_dir = REPO / "config" / f"le_b1b2_{batch}_lat12p8"
    jobs    = (cfg_dir / "job_list.txt").read_text().splitlines()
    fs      = pd.read_csv(FS_DIR / f"fold_splits_{batch}.csv")

    results = []
    n_skip  = 0
    print(f"  {batch}: {len(jobs)} jobs", flush=True)
    for i, job_path in enumerate(jobs):
        if i % 50 == 0:
            print(f"    {i}/{len(jobs)} ...", flush=True)
        name = Path(job_path).stem
        m = JOB_RE.search(name)
        if not m:
            continue
        b, fold, budget, repeat = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        row = _eval_one(name, b, fold, budget, repeat, feat_df, fs, classifier)
        if row:
            results.append(row)
        else:
            n_skip += 1

    print(f"    Done: {len(results)} rows, {n_skip} skipped")
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["lgbm", "logreg"], default="lgbm")
    args = ap.parse_args()
    clf = args.classifier
    tag = f"_logreg" if clf == "logreg" else ""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    cp_feat = pd.read_csv(CP_CSV).set_index("filename")
    il_feat = pd.read_csv(IL_CSV).set_index("filename")

    for feat_name, feat_df in [("cp", cp_feat), ("il", il_feat)]:
        print(f"\n=== {feat_name.upper()} ({clf}) ===")
        for batch in ["b1", "b2"]:
            df = _run_batch(batch, feat_df, clf)
            out = EVAL_DIR / f"{feat_name}_b1b2_{batch}{tag}_ds1.csv"
            df.to_csv(out, index=False)
            print(f"  Saved → {out}")

            if not df.empty:
                num = df[df["budget"] != "all"].copy()
                num["b"] = num["budget"].astype(int)
                s = num.groupby("b")["bal_acc"].mean()
                print(f"  Mean bal_acc by budget:\n{s.round(3).to_string()}")


if __name__ == "__main__":
    main()
