#!/usr/bin/env python3
"""
eval_le_b2_ctrl_vs_ycomp.py

For every completed DS1 B2 LE run (le_b2_lat12p8), evaluate how well
the latent features classify ctrl vs ycomp — overall, within adhesion
patches, and within no-adhesion patches.

Also evaluates CP and ilastik features at the same budget-limited training sets.

Outputs (EVAL_DIR):
  le_b2_ctrlvy_supcon_ds1.csv
  le_b2_ctrlvy_cp_ds1.csv
  le_b2_ctrlvy_il_ds1.csv

Columns: fold, budget, repeat, subset (all/ad/noad), n_train, n_test, bal_acc

Usage
-----
  python scripts/eval_le_b2_ctrl_vs_ycomp.py [--classifier lgbm|logreg]
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

DATA      = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR  = DATA / "ae_results" / "features" / "eval_results"
RUN_ROOT  = DATA / "ae_results" / "contrastive_run" / "le_b2_lat12p8"
ANN_DIR   = DATA / "labelling" / "le_b2_supcon"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "le_b2_lat12p8"

CP_CSV = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
IL_CSV = DATA / "ae_results" / "features" / "ilastik" / "ds1.csv"

LGBM_PARAMS = dict(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    min_child_samples=1, class_weight="balanced",
    n_jobs=4, random_state=42, verbose=-1,
)


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _condition(uid: str) -> str:
    """Extract condition from filename: 'control-f...' → 'control', 'ycomp-f...' → 'ycomp'"""
    return uid.split("-f")[0].split("_f")[0]


def _load_fold_splits() -> pd.DataFrame:
    fs = pd.read_csv(ANN_DIR / "fold_splits_ds1.csv")
    fs["fn"]        = fs["unique_ID"].apply(_h2u)
    fs["condition"] = fs["unique_ID"].apply(_condition)
    return fs


def _load_job_list() -> list[tuple]:
    job_re = re.compile(r"le_b2_lat12p8_ds1_fv(\d)_nb(\w+)_r(\d)")
    names  = []
    jl = CONFIG_DIR / "job_list.txt"
    if not jl.exists():
        return names
    for line in jl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name = Path(line).stem
        m = job_re.search(name)
        if m:
            names.append((name, int(m.group(1)), m.group(2), int(m.group(3))))
    return names


def _eval_subset(X_tr, y_tr, X_te, y_te, classifier):
    if len(X_tr) == 0 or len(X_te) == 0:
        return None
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
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
    return balanced_accuracy_score(y_te_enc, y_pred)


def _run_eval(fold_splits, feat_df, job_names, classifier,
              latent_loader=False) -> pd.DataFrame:
    fs = fold_splits.copy()
    results  = []
    n_skipped = 0

    for i, (name, fold, budget, repeat) in enumerate(job_names):
        if i % 50 == 0:
            print(f"  {i}/{len(job_names)} ...", flush=True)

        if latent_loader:
            lat_path = RUN_ROOT / name / "latents.csv"
            if not lat_path.exists():
                n_skipped += 1
                continue
            lat_raw = pd.read_csv(lat_path)
            z_cols  = [c for c in lat_raw.columns if c.startswith("z_") and "_proj" not in c]
            feats   = lat_raw[["filename"] + z_cols].set_index("filename")
        else:
            feats = feat_df

        ann_name = name.replace("_lat12p8", "")
        ann_path = ANN_DIR / f"{ann_name}.csv"
        if not ann_path.exists():
            n_skipped += 1
            continue

        ann = pd.read_csv(ann_path)
        train_fns = set(ann["unique_ID"].apply(_h2u))

        test_fs  = fs[fs["fold"] == fold].copy()
        test_fs  = test_fs[~test_fs["fn"].isin(train_fns)]
        train_fs = fs[fs["fn"].isin(train_fns)].copy()

        for subset, mask_fn in [
            ("all",  lambda df: pd.Series([True] * len(df), index=df.index)),
            ("ad",   lambda df: df["label"] == "adhesion"),
            ("noad", lambda df: df["label"] == "No adhesion"),
        ]:
            tr_mask = mask_fn(train_fs).values
            te_mask = mask_fn(test_fs).values

            tr_df = train_fs[tr_mask]
            te_df = test_fs[te_mask]

            X_tr_raw = feats.reindex(tr_df["fn"]).values
            X_te_raw = feats.reindex(te_df["fn"]).values

            tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
            te_ok = ~np.isnan(X_te_raw).any(axis=1)

            X_tr = X_tr_raw[tr_ok]
            X_te = X_te_raw[te_ok]
            y_tr = tr_df["condition"].values[tr_ok]
            y_te = te_df["condition"].values[te_ok]

            acc = _eval_subset(X_tr, y_tr, X_te, y_te, classifier)
            if acc is None:
                continue

            results.append(dict(
                fold=fold, budget=budget, repeat=repeat,
                subset=subset,
                n_train=int(tr_ok.sum()), n_test=int(te_ok.sum()),
                bal_acc=round(acc, 6),
            ))

    print(f"  Done: {len(results)} rows, {n_skipped} skipped")
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["lgbm", "logreg"], default="lgbm")
    args = ap.parse_args()
    clf = args.classifier

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    fold_splits = _load_fold_splits()
    job_names   = _load_job_list()
    tag         = f"_{clf}" if clf != "lgbm" else ""
    print(f"Total jobs: {len(job_names)}")

    cp_feat = pd.read_csv(CP_CSV).set_index("filename")
    il_feat = pd.read_csv(IL_CSV).set_index("filename")

    print("\nEvaluating SupCon latents …")
    df_sc = _run_eval(fold_splits, None, job_names, clf, latent_loader=True)
    sc_out = EVAL_DIR / f"le_b2_ctrlvy_supcon{tag}_ds1.csv"
    df_sc.to_csv(sc_out, index=False)
    print(f"  Saved → {sc_out}")

    print("\nEvaluating CP features …")
    df_cp = _run_eval(fold_splits, cp_feat, job_names, clf)
    cp_out = EVAL_DIR / f"le_b2_ctrlvy_cp{tag}_ds1.csv"
    df_cp.to_csv(cp_out, index=False)
    print(f"  Saved → {cp_out}")

    print("\nEvaluating ilastik features …")
    df_il = _run_eval(fold_splits, il_feat, job_names, clf)
    il_out = EVAL_DIR / f"le_b2_ctrlvy_il{tag}_ds1.csv"
    df_il.to_csv(il_out, index=False)
    print(f"  Saved → {il_out}")

    for label, df in [("SupCon", df_sc), ("CP", df_cp), ("ilastik", df_il)]:
        if df.empty:
            continue
        num = df[df["budget"] != "all"].copy()
        num["b"] = num["budget"].astype(int)
        s = num.groupby(["subset", "b"])["bal_acc"].mean().unstack(0)
        print(f"\n{label} mean bal_acc (ctrl vs ycomp):\n{s.round(3)}")


if __name__ == "__main__":
    main()
