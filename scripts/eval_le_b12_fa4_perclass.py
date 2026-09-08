#!/usr/bin/env python3
"""
eval_le_b12_fa4_perclass.py

For every completed DS1 B12 LE run (le_b12_ds1_lat12p8), train a 5-class
(FA subtype) LightGBM classifier using training patches that have B2 FA4
labels, then compute per-class F1 on the B2-labeled test patches.

FA classes: No adhesion, focal adhesion, Nascent Adhesion,
            focal complex, fibrillar adhesion

Outputs (EVAL_DIR):
  le_b12_fa4_perclass_supcon_ds1.csv
  le_b12_fa4_perclass_cp_ds1.csv
  le_b12_fa4_perclass_il_ds1.csv

Columns: fold, budget, repeat, n_train, n_test, bal_acc, macro_f1,
         f1_No adhesion, f1_focal adhesion, f1_Nascent Adhesion,
         f1_focal complex, f1_fibrillar adhesion

Usage
-----
  python scripts/eval_le_b12_fa4_perclass.py [--classifier lgbm|logreg]
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA      = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO      = Path(__file__).resolve().parents[1]
EVAL_DIR  = DATA / "ae_results" / "features" / "eval_results"
RUN_ROOT  = DATA / "ae_results" / "contrastive_run" / "le_b12_ds1_lat12p8"
ANN_DIR   = DATA / "labelling" / "le_b12_supcon"

CP_CSV  = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
IL_CSV  = DATA / "ae_results" / "features" / "ilastik" / "ds1.csv"
B2_CSV  = DATA / "labelling" / "vinc_combined_label_Annabel_20260816.csv"
CONFIG_DIR = REPO / "config" / "le_b12_ds1_lat12p8"

FA4_CLASSES = [
    "No adhesion", "focal adhesion", "Nascent Adhesion",
    "focal complex", "fibrillar adhesion",
]

LGBM_PARAMS = dict(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    min_child_samples=1, class_weight="balanced",
    n_jobs=4, random_state=42, verbose=-1,
)


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _load_b2_fa4() -> dict[str, str]:
    """Return dict: underscore filename → FA4 label."""
    b2 = pd.read_csv(B2_CSV)
    b2["fn"] = b2["filename"].apply(lambda x: x.replace("-f", "_f", 1))
    return dict(zip(b2["fn"], b2["label"]))


def _load_fold_splits() -> pd.DataFrame:
    fs = pd.read_csv(ANN_DIR / "fold_splits_ds1.csv")
    fs["fn"] = fs["unique_ID"].apply(_h2u)
    return fs


def _load_job_list():
    import re
    job_re = re.compile(r"le_b12_ds1_lat12p8_fv(\d)_nb(\w+)_r(\d)")
    names = []
    for jl in [CONFIG_DIR / "job_list.txt", CONFIG_DIR / "job_list_nb1000_1500.txt"]:
        if not jl.exists():
            continue
        for line in jl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            name = Path(line).stem
            m = job_re.search(name)
            if m:
                names.append((name, int(m.group(1)), m.group(2), int(m.group(3))))
    return names


def _run_eval(fold_splits, b2_fa4, feat_df, job_names, classifier,
              latent_loader=False):
    results = []
    n_skip  = 0

    for i, (name, fold, budget, repeat) in enumerate(job_names):
        if i % 50 == 0:
            print(f"  {i}/{len(job_names)} …", flush=True)

        # Load features
        if latent_loader:
            lat_path = RUN_ROOT / name / "latents.csv"
            if not lat_path.exists():
                n_skip += 1
                continue
            lat_raw = pd.read_csv(lat_path)
            z_cols  = [c for c in lat_raw.columns if c.startswith("z_") and "_proj" not in c]
            feats   = lat_raw[["filename"] + z_cols].set_index("filename")
        else:
            feats = feat_df

        # Training annotation CSV (binary labels; we remap to FA4 below)
        ann_name = name.replace("_lat12p8", "")
        ann_path = ANN_DIR / f"{ann_name}.csv"
        if not ann_path.exists():
            n_skip += 1
            continue

        ann = pd.read_csv(ann_path)
        train_fns = set(ann["unique_ID"].apply(_h2u))

        # Keep only B2 patches with FA4 labels
        train_fns_b2 = [fn for fn in train_fns if fn in b2_fa4]
        test_fs = fold_splits[fold_splits["fold"] == fold].copy()
        test_fs = test_fs[~test_fs["fn"].isin(train_fns)]
        test_fs_b2 = test_fs[test_fs["fn"].isin(b2_fa4)]

        if len(train_fns_b2) == 0 or len(test_fs_b2) == 0:
            n_skip += 1
            continue

        X_tr_raw = feats.reindex(train_fns_b2).values
        X_te_raw = feats.reindex(test_fs_b2["fn"]).values

        tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
        te_ok = ~np.isnan(X_te_raw).any(axis=1)

        X_tr = X_tr_raw[tr_ok]
        X_te = X_te_raw[te_ok]
        y_tr = np.array([b2_fa4[fn] for fn in train_fns_b2])[tr_ok]
        y_te = test_fs_b2["fn"].map(b2_fa4).values[te_ok]

        if len(np.unique(y_tr)) < 2:
            n_skip += 1
            continue

        le = LabelEncoder().fit(FA4_CLASSES)
        y_tr_enc = le.transform(y_tr)
        y_te_enc = le.transform(y_te)

        if classifier == "logreg":
            sc = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     multi_class="multinomial", random_state=42)
            clf.fit(sc.transform(X_tr), y_tr_enc)
            y_pred = clf.predict(sc.transform(X_te))
        else:
            clf = LGBMClassifier(**LGBM_PARAMS)
            clf.fit(X_tr, y_tr_enc)
            y_pred = clf.predict(X_te)

        bal_acc  = balanced_accuracy_score(y_te_enc, y_pred)
        f1_per   = f1_score(y_te_enc, y_pred, labels=range(len(FA4_CLASSES)),
                            average=None, zero_division=0)
        macro_f1 = f1_score(y_te_enc, y_pred, average="macro", zero_division=0)

        row = dict(fold=fold, budget=budget, repeat=repeat,
                   n_train=int(tr_ok.sum()), n_test=int(te_ok.sum()),
                   bal_acc=round(bal_acc, 6), macro_f1=round(macro_f1, 6))
        for cls, f1 in zip(FA4_CLASSES, f1_per):
            row[f"f1_{cls}"] = round(float(f1), 6)
        results.append(row)

    print(f"  Done: {len(results)} rows, {n_skip} skipped")
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["lgbm", "logreg"], default="lgbm")
    args = ap.parse_args()
    clf  = args.classifier

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    b2_fa4      = _load_b2_fa4()
    fold_splits = _load_fold_splits()
    job_names   = _load_job_list()
    tag         = f"_{clf}" if clf != "lgbm" else ""

    print(f"Jobs: {len(job_names)}  B2 FA4 patches: {len(b2_fa4)}")

    cp_feat = pd.read_csv(CP_CSV).set_index("filename")
    il_feat = pd.read_csv(IL_CSV).set_index("filename")

    print("\nEvaluating SupCon latents …")
    df_sc = _run_eval(fold_splits, b2_fa4, None, job_names, clf, latent_loader=True)
    out = EVAL_DIR / f"le_b12_fa4_perclass_supcon{tag}_ds1.csv"
    df_sc.to_csv(out, index=False)
    print(f"  Saved → {out}")

    print("\nEvaluating CP features …")
    df_cp = _run_eval(fold_splits, b2_fa4, cp_feat, job_names, clf)
    out = EVAL_DIR / f"le_b12_fa4_perclass_cp{tag}_ds1.csv"
    df_cp.to_csv(out, index=False)
    print(f"  Saved → {out}")

    print("\nEvaluating ilastik features …")
    df_il = _run_eval(fold_splits, b2_fa4, il_feat, job_names, clf)
    out = EVAL_DIR / f"le_b12_fa4_perclass_il{tag}_ds1.csv"
    df_il.to_csv(out, index=False)
    print(f"  Saved → {out}")

    # Quick summary
    for label, df in [("SupCon", df_sc), ("CP", df_cp), ("ilastik", df_il)]:
        if df.empty:
            continue
        num = df[df["budget"] != "all"].copy()
        num["b"] = num["budget"].astype(int)
        f1_cols = [f"f1_{c}" for c in FA4_CLASSES if f"f1_{c}" in num.columns]
        s = num.groupby("b")[f1_cols].mean()
        s.columns = [c.replace("f1_", "") for c in s.columns]
        print(f"\n{label} mean per-class F1:\n{s.round(3).to_string()}")


if __name__ == "__main__":
    main()
