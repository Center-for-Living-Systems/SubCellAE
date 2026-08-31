#!/usr/bin/env python3
"""
compare_pd8_vs_pd16_ds2.py

Compare le_b2_supcon (proj_dim=8) vs le_b2_supcon_pd16 (proj_dim=16)
on DS2, using whatever jobs are already complete.

Classifier: logistic regression (consistent with SupCon latent evaluation).
Metric: balanced accuracy.

Output: ae_results/features/eval_results/pd8_vs_pd16_ds2.csv
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO    = Path(__file__).resolve().parents[1]
ANN_DIR = DATA / "labelling" / "le_b2_supcon"
OUT_DIR = DATA / "ae_results" / "features" / "eval_results"

RUN_ROOTS = {
    "pd8":  DATA / "ae_results" / "contrastive_run" / "le_b2_supcon",
    "pd16": DATA / "ae_results" / "contrastive_run" / "le_b2_supcon_pd16",
}

JOB_RE = re.compile(r"le_b2_(ds\d)_fv(\d)_nb(\w+)_r(\d)")
Z_COLS = [f"z_{i}" for i in range(32)]

LR_PARAMS = dict(max_iter=2000, class_weight="balanced", random_state=42)


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def eval_job(name: str, fold: int, budget: str, repeat: int,
             run_dir: Path, fold_splits: pd.DataFrame) -> dict | None:
    lat_path = run_dir / name / "latents.csv"
    ann_path = ANN_DIR / f"{name}.csv"
    if not lat_path.exists() or not ann_path.exists():
        return None

    lat = pd.read_csv(lat_path, usecols=["filename"] + Z_COLS).set_index("filename")

    train_ann = pd.read_csv(ann_path)
    train_ann["fn"] = train_ann["unique_ID"].apply(_h2u)

    test_ann = fold_splits[fold_splits["fold"] == fold].copy()
    test_ann["fn"] = test_ann["unique_ID"].apply(_h2u)
    test_ann = test_ann[~test_ann["fn"].isin(set(train_ann["fn"]))]

    X_tr = lat.reindex(train_ann["fn"])
    X_te = lat.reindex(test_ann["fn"])
    mtr  = X_tr.notna().all(axis=1)
    mte  = X_te.notna().all(axis=1)

    X_tr = X_tr[mtr].values;  y_tr = train_ann.loc[mtr.values, "label"].values
    X_te = X_te[mte].values;  y_te = test_ann.loc[mte.values,  "label"].values

    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return None

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)

    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(sc.transform(X_tr), y_tr_enc)
    bal_acc = balanced_accuracy_score(y_te_enc, lr.predict(sc.transform(X_te)))

    return dict(fold=fold, budget=budget, repeat=repeat,
                n_train=int(mtr.sum()), n_test=int(mte.sum()),
                bal_acc=round(bal_acc, 6))


def eval_config(tag: str, run_dir: Path, fold_splits: pd.DataFrame,
                jobs: list[str]) -> pd.DataFrame:
    results = []
    n_skip  = 0
    for job_path in jobs:
        name = Path(job_path).stem
        m    = JOB_RE.search(name)
        if not m:
            continue
        _, fold, budget, repeat = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        row = eval_job(name, fold, budget, repeat, run_dir, fold_splits)
        if row:
            results.append(row)
        else:
            n_skip += 1
    df = pd.DataFrame(results)
    print(f"  {tag}: {len(results)} evaluated, {n_skip} skipped (not done)")
    return df


def summarize(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    num = df[df["budget"] != "all"].copy()
    num["bi"] = num["budget"].astype(int)
    s = num.groupby("bi")["bal_acc"].agg(["mean", "std", "count"]).round(4)
    s.columns = [f"{tag}_mean", f"{tag}_std", f"{tag}_n"]
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_splits = pd.read_csv(ANN_DIR / "fold_splits_ds2.csv")
    fold_splits["fn"] = fold_splits["unique_ID"].apply(_h2u)

    # DS2 jobs from main job list (pd8) and pd16 job list
    all_jobs_main = Path(REPO / "config/le_b2_supcon/job_list.txt").read_text().splitlines()
    ds2_jobs_pd8  = [j for j in all_jobs_main if "_b2_ds2_" in j]
    ds2_jobs_pd16 = Path(REPO / "config/le_b2_supcon_pd16/job_list_ds2.txt").read_text().splitlines()

    print(f"DS2 jobs — pd8: {len(ds2_jobs_pd8)}  pd16: {len(ds2_jobs_pd16)}")

    print("\nEvaluating pd8 ...")
    df8  = eval_config("pd8",  RUN_ROOTS["pd8"],  fold_splits, ds2_jobs_pd8)
    print("Evaluating pd16 ...")
    df16 = eval_config("pd16", RUN_ROOTS["pd16"], fold_splits, ds2_jobs_pd16)

    s8  = summarize(df8,  "pd8")
    s16 = summarize(df16, "pd16")

    cmp = s8.join(s16, how="outer")
    cmp["delta_mean"] = (cmp["pd16_mean"] - cmp["pd8_mean"]).round(4)
    print(f"\n{'='*65}")
    print("DS2 logreg balanced accuracy: pd8 vs pd16  (mean ± std)")
    print(f"{'='*65}")
    print(cmp.to_string())

    # Save full results
    df8["config"]  = "pd8"
    df16["config"] = "pd16"
    out = pd.concat([df8, df16], ignore_index=True)
    out_path = OUT_DIR / "pd8_vs_pd16_ds2.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
