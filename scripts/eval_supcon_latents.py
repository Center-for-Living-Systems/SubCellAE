#!/usr/bin/env python3
"""
eval_supcon_latents.py

Evaluate SupCon-AE latents using the same 5-fold CV splits as the
le_b2_supcon benchmark. For each completed run, loads its latents.csv,
trains LightGBM on the training-fold latents, tests on the held-out fold.

Usage
-----
  python scripts/eval_supcon_latents.py --dataset ds1
  python scripts/eval_supcon_latents.py --dataset ds2
  python scripts/eval_supcon_latents.py --all

Output
------
  ae_results/features/eval_results/supcon_{run_tag}_{ds}.csv
  Columns: name, ds, fold, budget, repeat, n_train, n_test, bal_acc
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO    = Path(__file__).resolve().parents[1]
OUT_DIR = DATA / "ae_results" / "features" / "eval_results"

RUN_TAG_CFG = {
    "le_b2_supcon": {
        "run_root": DATA / "ae_results" / "contrastive_run" / "le_b2_supcon",
        "ann_dir":  DATA / "labelling" / "le_b2_supcon",
        "job_re":   re.compile(r"le_b2_(ds\d)_fv(\d)_nb(\w+)_r(\d)"),
    },
    "le_b2_supcon_pd16": {
        "run_root": DATA / "ae_results" / "contrastive_run" / "le_b2_supcon_pd16",
        "ann_dir":  DATA / "labelling" / "le_b2_supcon",
        "job_re":   re.compile(r"le_b2_(ds\d)_fv(\d)_nb(\w+)_r(\d)"),
    },
    "le_b12_supcon": {
        "run_root": DATA / "ae_results" / "contrastive_run" / "le_b12_supcon",
        "ann_dir":  DATA / "labelling" / "le_b12_supcon",
        "job_re":   re.compile(r"le_b12_(ds\d)_fv(\d)_nb(\w+)_r(\d)"),
    },
    "le_b2_vinc_ctrl": {
        "run_root":         DATA / "ae_results" / "contrastive_run" / "le_b2_vinc_ctrl",
        "ann_dir":          DATA / "labelling" / "le_b2_vinc_ctrl",
        "job_re":           re.compile(r"le_b2_vinc_ctrl_fv(\d)_nb(\w+)_r(\d)"),
        "fixed_ds":         "vc",
        "fold_splits_file": "fold_splits.csv",
        "datasets":         ["vc"],
    },
    "le_b2_lat12p8": {
        "run_root":    DATA / "ae_results" / "contrastive_run" / "le_b2_lat12p8",
        "ann_dir":     DATA / "labelling" / "le_b2_supcon",
        "job_re":      re.compile(r"le_b2_lat12p8_(ds\d)_fv(\d)_nb(\w+)_r(\d)"),
        "ann_name_fn": lambda name: name.replace("le_b2_lat12p8_", "le_b2_"),
        "datasets":    ["ds1"],
    },
    "le_b2_ds1c": {
        "run_root":          DATA / "ae_results" / "contrastive_run" / "le_b2_ds1c",
        "ann_dir":           DATA / "labelling" / "le_b2_ds1c",
        "job_re":            re.compile(r"le_b2_ds1c_fv(\d)_nb(\w+)_r(\d)"),
        "fixed_ds":          "ds1c",
        "fold_splits_file":  "fold_splits_ds1c.csv",
        "datasets":          ["ds1c"],
    },
    "le_b12_ds1_lat12p8": {
        "run_root":    DATA / "ae_results" / "contrastive_run" / "le_b12_ds1_lat12p8",
        "ann_dir":     DATA / "labelling" / "le_b12_supcon",
        "job_re":      re.compile(r"le_b12_ds1_lat12p8_fv(\d)_nb(\w+)_r(\d)"),
        "ann_name_fn": lambda name: name.replace("_lat12p8", ""),
        "fixed_ds":    "ds1",
        "fold_splits_file": "fold_splits_ds1.csv",
        "datasets":    ["ds1"],
    },
    "le_b12_ds2_lat64p32": {
        "run_root":    DATA / "ae_results" / "contrastive_run" / "le_b12_ds2_lat64p32",
        "ann_dir":     DATA / "labelling" / "le_b12_supcon",
        "job_re":      re.compile(r"le_b12_ds2_lat64p32_fv(\d)_nb(\w+)_r(\d)"),
        "ann_name_fn": lambda name: name.replace("le_b12_ds2_lat64p32_", "le_b12_ds2_"),
        "fixed_ds":    "ds2",
        "fold_splits_file": "fold_splits_ds2.csv",
        "datasets":    ["ds2"],
    },
}

Z_COLS = [f"z_{i}" for i in range(32)]

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


def eval_job(name: str, ds: str, fold: int, budget: str, repeat: int,
             run_dir: Path, fold_splits: pd.DataFrame,
             ann_dir: Path = None, classifier: str = "lgbm",
             ann_name_fn=None) -> dict | None:
    latent_path = run_dir / name / "latents.csv"
    if not latent_path.exists():
        return None

    ann_name = ann_name_fn(name) if ann_name_fn else name
    ann_path = (ann_dir or ANN_DIR) / f"{ann_name}.csv"
    if not ann_path.exists():
        return None

    all_cols = pd.read_csv(latent_path, nrows=0).columns.tolist()
    z_cols   = [c for c in all_cols if c.startswith("z_")]
    latents  = pd.read_csv(latent_path, usecols=["filename"] + z_cols)
    latents  = latents.set_index("filename")

    train_ann = pd.read_csv(ann_path)
    train_ann["fn"] = train_ann["unique_ID"].apply(_hyph_to_under)

    test_ann = fold_splits[fold_splits["fold"] == fold].copy()
    test_ann["fn"] = test_ann["unique_ID"].apply(_hyph_to_under)
    train_fns = set(train_ann["fn"])
    test_ann  = test_ann[~test_ann["fn"].isin(train_fns)]

    train_feats = latents.reindex(train_ann["fn"])
    test_feats  = latents.reindex(test_ann["fn"])

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
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_train, y_train_enc)
        y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test_enc, y_pred)

    return dict(
        name=name, ds=ds, fold=fold, budget=budget, repeat=repeat,
        n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        bal_acc=round(bal_acc, 6),
    )


def process_dataset(ds: str, run_tag: str, classifier: str = "lgbm") -> pd.DataFrame:
    cfg      = RUN_TAG_CFG[run_tag]
    run_dir  = cfg["run_root"]
    ann_dir  = cfg["ann_dir"]
    job_re   = cfg["job_re"]
    fixed_ds    = cfg.get("fixed_ds")
    ann_name_fn = cfg.get("ann_name_fn")
    actual_ds   = fixed_ds or ds

    job_list_path = REPO / "config" / run_tag / f"job_list_{ds}.txt"
    if not job_list_path.exists():
        main_list = REPO / "config" / run_tag / "job_list.txt"
        all_jobs  = main_list.read_text().splitlines()
        jobs = all_jobs if fixed_ds else [j for j in all_jobs if f"_{ds}_" in j]
    else:
        jobs = job_list_path.read_text().splitlines()

    fs_file = cfg.get("fold_splits_file", f"fold_splits_{ds}.csv")
    fold_splits = pd.read_csv(ann_dir / fs_file)

    print(f"\n{actual_ds} | {run_tag} | {classifier}: {len(jobs)} jobs", flush=True)

    results = []
    n_missing = 0
    for i, job_path in enumerate(jobs):
        name = Path(job_path).stem
        m    = job_re.search(name)
        if not m:
            continue
        if fixed_ds:
            fold, budget, repeat = int(m.group(1)), m.group(2), int(m.group(3))
        else:
            _, fold, budget, repeat = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))

        if i % 50 == 0:
            print(f"  {i}/{len(jobs)} ...", flush=True)

        row = eval_job(name, actual_ds, fold, budget, repeat, run_dir, fold_splits, ann_dir, classifier, ann_name_fn)
        if row:
            results.append(row)
        else:
            n_missing += 1

    print(f"  Evaluated: {len(results)}  Skipped (not done): {n_missing}")

    df = pd.DataFrame(results)
    if len(df):
        clf_tag = f"_{classifier}" if classifier != "lgbm" else ""
        out = OUT_DIR / f"supcon_{run_tag}{clf_tag}_{ds}.csv"
        df.to_csv(out, index=False)
        print(f"  Saved {len(df)} rows → {out}")
        numeric = df[df["budget"] != "all"].copy()
        numeric["budget_int"] = numeric["budget"].astype(int)
        summary = (numeric.groupby("budget_int")["bal_acc"]
                   .agg(["mean", "std"]).round(4))
        print(f"\n  Budget summary:\n{summary}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="le_b2_supcon",
                    choices=list(RUN_TAG_CFG.keys()))
    ap.add_argument("--classifier", choices=["lgbm", "logreg"], default="logreg")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset")
    grp.add_argument("--all", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = RUN_TAG_CFG[args.run_tag]
    default_datasets = cfg.get("datasets", ["ds1", "ds2", "ds3"])
    datasets = default_datasets if args.all else [args.dataset]
    for ds in datasets:
        process_dataset(ds, args.run_tag, args.classifier)

    print("\nDone.")


if __name__ == "__main__":
    main()
