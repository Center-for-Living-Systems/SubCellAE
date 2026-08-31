#!/usr/bin/env python3
"""
eval_b12_combined_labels.py

Evaluate SupCon-AE, CellProfiler, and ilastik using combined B1+B2 DS1 labels
for the downstream classifier — WITHOUT retraining any model.

For each existing le_b2_supcon DS1 job (fold × budget × repeat):
  - Test set  : B2 fold-k patches (identical to standard eval)
  - Train set : B2 budget-n subsample (from annotation CSV)
                + ALL B1-only patches (non-overlapping with B2 by construction)
  - Classifier: logistic regression
  - Metric    : balanced accuracy

B1-only = labels_vinc_20260521.csv excluding Uncertain + B2-overlapping patches.
These patches were already encoded by every DS1 SupCon-AE model (all patches
are encoded regardless of label status) and their features are in the same
CP/ilastik CSVs.

Output
------
  ae_results/features/eval_results/b12_combined_{method}_ds1.csv
  Columns: method, fold, budget, repeat, n_b2_train, n_b1_train, n_test, bal_acc
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

BASE_SC = DATA / "ae_results" / "contrastive_run" / "le_b2_supcon"
FEAT_DIRS = {
    "cp":      DATA / "ae_results" / "features" / "cellprofiler",
    "ilastik": DATA / "ae_results" / "features" / "ilastik",
}
Z_COLS = [f"z_{i}" for i in range(32)]

B1_FILE = DATA / "labelling" / "labels_vinc_20260521.csv"
B2_FILE = DATA / "labelling" / "vinc_combined_label_Annabel_20260816.csv"

JOB_RE  = re.compile(r"le_b2_(ds\d)_fv(\d)_nb(\w+)_r(\d)")
LR_PARAMS = dict(max_iter=2000, class_weight="balanced", random_state=42)


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _binarize(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def build_b1_only(b2_fns: set[str]) -> pd.DataFrame:
    b1 = pd.read_csv(B1_FILE)
    b1 = b1[b1["classification"] != "Uncertain"].copy()
    b1["fn"]    = b1["unique_ID"].apply(_h2u)
    b1["label"] = b1["classification"].apply(_binarize)
    return b1[~b1["fn"].isin(b2_fns)][["fn", "label"]].reset_index(drop=True)


def lr_eval(X_tr, y_tr, X_te, y_te):
    le = LabelEncoder()
    yt = le.fit_transform(y_tr)
    yv = le.transform(y_te)
    if len(np.unique(yt)) < 2:
        return np.nan
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(sc.transform(X_tr), yt)
    return balanced_accuracy_score(yv, lr.predict(sc.transform(X_te)))


def main():
    # Load B2 fold splits
    b2_splits = pd.read_csv(ANN_DIR / "fold_splits_ds1.csv")
    b2_splits["fn"] = b2_splits["unique_ID"].apply(_h2u)
    b2_fns = set(b2_splits["fn"])

    # Build B1-only training pool
    b1 = build_b1_only(b2_fns)
    print(f"B1-only pool: {len(b1)}  "
          f"(adh={( b1['label']=='adhesion').sum()} "
          f"noad={(b1['label']=='No adhesion').sum()})")

    # DS1 job list
    jl = REPO / "config" / "le_b2_supcon" / "job_list.txt"
    jobs = [j for j in jl.read_text().splitlines() if "_b2_ds1_" in j]

    # Feature matrices (loaded once)
    feat_dfs = {k: pd.read_csv(v / "ds1.csv").set_index("filename")
                for k, v in FEAT_DIRS.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {m: [] for m in ["supcon", "cp", "ilastik"]}

    for i, job_path in enumerate(jobs):
        name = Path(job_path).stem
        m = JOB_RE.search(name)
        if not m:
            continue
        _, fold, bud_s, repeat = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if bud_s == "all":
            continue

        if i % 50 == 0:
            print(f"  {i}/{len(jobs)} ...", flush=True)

        ann = pd.read_csv(ANN_DIR / f"{name}.csv")
        ann["fn"] = ann["unique_ID"].apply(_h2u)

        # B2 test set: fold-k patches not in training
        test = b2_splits[(b2_splits["fold"] == fold) &
                         (~b2_splits["fn"].isin(set(ann["fn"])))].copy()

        # Combined training: B2 budget subsample + all B1
        train = pd.concat([
            ann[["fn", "label"]],
            b1[["fn", "label"]],
        ], ignore_index=True).drop_duplicates(subset="fn")

        n_b2 = len(ann)
        n_b1 = len(train) - n_b2

        row_base = dict(fold=fold, budget=bud_s, repeat=repeat,
                        n_b2_train=n_b2, n_b1_train=n_b1,
                        n_test=len(test))

        # ── SupCon latents ───────────────────────────────────────────────────
        lat_path = BASE_SC / name / "latents.csv"
        if lat_path.exists():
            lat = pd.read_csv(lat_path, usecols=["filename"] + Z_COLS).set_index("filename")
            tf  = lat.reindex(train["fn"]);  mtr = tf.notna().all(axis=1)
            ev  = lat.reindex(test["fn"]);   mte = ev.notna().all(axis=1)
            X_tr = tf[mtr].values;  y_tr = train.loc[mtr.values, "label"].values
            X_te = ev[mte].values;  y_te = test.loc[mte.values, "label"].values
            acc  = lr_eval(X_tr, y_tr, X_te, y_te)
            results["supcon"].append({**row_base, "bal_acc": acc})

        # ── Handcrafted features ─────────────────────────────────────────────
        for feat, fdf in feat_dfs.items():
            tf  = fdf.reindex(train["fn"]); mtr = tf.notna().all(axis=1)
            ev  = fdf.reindex(test["fn"]);  mte = ev.notna().all(axis=1)
            X_tr = tf[mtr].values;  y_tr = train.loc[mtr.values, "label"].values
            X_te = ev[mte].values;  y_te = test.loc[mte.values, "label"].values
            acc  = lr_eval(X_tr, y_tr, X_te, y_te)
            results[feat].append({**row_base, "bal_acc": acc})

    for method, rows in results.items():
        df = pd.DataFrame(rows)
        out = OUT_DIR / f"b12_combined_{method}_ds1.csv"
        df.to_csv(out, index=False)
        num = df[df["budget"] != "all"].copy()
        num["bi"] = num["budget"].astype(int)
        summary = num.groupby("bi")["bal_acc"].agg(["mean", "std"]).round(4)
        print(f"\n{method} (B1+B2 training):\n{summary}")

    print("\nDone.")


if __name__ == "__main__":
    main()
