#!/usr/bin/env python3
"""
eval_fa4_features.py

5-fold stratified CV for FA 4-class subtype classification using:
  - SupCon-AE z latents  (32-d, averaged across 5 nball models)
  - CellProfiler features (50-d)
  - ilastik features      (80-d)

Labels: B1 (Margaret) vinc patches with FA subtypes, Uncertain and No-adhesion dropped.
  Nascent Adhesion | focal complex | focal adhesion | fibrillar adhesion

No model retraining — features are taken as-is.
Classifier: logistic regression and LightGBM (both reported).
Metrics: balanced accuracy, macro F1, per-class F1.

Output
------
  ae_results/features/eval_results/fa4_cv_{method}.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO    = Path(__file__).resolve().parents[1]
ANN_DIR = DATA / "labelling"
OUT_DIR = DATA / "ae_results" / "features" / "eval_results"
BASE_SC = DATA / "ae_results" / "contrastive_run" / "le_b2_supcon"
BASE_B12 = DATA / "ae_results" / "contrastive_run"

FA4 = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]

Z_COLS   = [f"z_{i}" for i in range(32)]
Z32_COLS = [f"z_{i}" for i in range(32)]   # B12 lat32 model
N_FOLDS  = 5
CV_SEED  = 42

LGBM_PARAMS = dict(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=1,
    class_weight="balanced",
    n_jobs=4,
    random_state=42,
    verbose=-1,
)
LR_PARAMS = dict(max_iter=2000, class_weight="balanced", random_state=42,
                 multi_class="multinomial", solver="lbfgs")


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def load_labels(combined_b12: bool = False) -> pd.DataFrame:
    FA4_set = set(FA4)
    b1 = pd.read_csv(ANN_DIR / "labels_vinc_20260521.csv")
    b1["fn"]    = b1["unique_ID"].apply(_h2u)
    b1["label"] = b1["classification"]
    b1 = b1[b1["label"].isin(FA4_set)].copy()
    b1_out = b1[["fn", "label"]].reset_index(drop=True)
    if not combined_b12:
        return b1_out

    # Add B2 (Annabel) labels, B2 takes priority for duplicates
    b2 = pd.read_csv(ANN_DIR / "vinc_combined_label_Annabel_20260816.csv")
    b2 = b2[b2["label"].isin(FA4_set)].copy()
    b2["fn"] = b2["filename"]   # already underscore format
    b2_out = b2[["fn", "label"]].reset_index(drop=True)
    b2_ids = set(b2_out["fn"])
    combined = pd.concat([b2_out, b1_out[~b1_out["fn"].isin(b2_ids)]], ignore_index=True)
    return combined.reset_index(drop=True)


def load_supcon_latents(patches: pd.DataFrame) -> pd.DataFrame:
    """Average latents across 5 nball models for stable representation."""
    all_z = []
    nball_runs = [f"le_b2_ds1_fv{f}_nball_r0" for f in range(N_FOLDS)]
    found_runs = 0
    for run in nball_runs:
        lat_path = BASE_SC / run / "latents.csv"
        if not lat_path.exists():
            continue
        lat = pd.read_csv(lat_path, usecols=["filename"] + Z_COLS).set_index("filename")
        z   = lat.reindex(patches["fn"])
        all_z.append(z.values)
        found_runs += 1
    print(f"  SupCon: averaged {found_runs} nball models", flush=True)
    avg_z = np.nanmean(np.stack(all_z, axis=0), axis=0)  # [N, 32]
    return pd.DataFrame(avg_z, columns=Z_COLS, index=patches.index)


def load_b12_latents(patches: pd.DataFrame) -> pd.DataFrame:
    """Average latents across 3 B12 lat32p16 models."""
    all_z = []
    found = 0
    for v in range(3):
        lat_path = BASE_B12 / f"annabel_vinc_supcon2_stage2_b12_lat32p16_v{v}" / "latents.csv"
        if not lat_path.exists():
            continue
        lat = pd.read_csv(lat_path, usecols=["filename"] + Z32_COLS).set_index("filename")
        z   = lat.reindex(patches["fn"])
        all_z.append(z.values)
        found += 1
    if found == 0:
        print("  B12-SupCon: no models found yet — skipping", flush=True)
        return pd.DataFrame()
    print(f"  B12-SupCon lat32: averaged {found}/3 models", flush=True)
    avg_z = np.nanmean(np.stack(all_z, axis=0), axis=0)
    return pd.DataFrame(avg_z, columns=Z32_COLS, index=patches.index)


def run_cv(X: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    le    = LabelEncoder().fit(labels)
    y_enc = le.transform(y)

    results = {clf: {"bal_acc": [], "macro_f1": [],
                     **{f"f1_{c}": [] for c in labels}}
               for clf in ["logreg", "lgbm"]}

    for fold, (tr, te) in enumerate(skf.split(X, y_enc)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_enc[tr], y_enc[te]

        if len(np.unique(y_tr)) < 2:
            continue

        sc = StandardScaler().fit(X_tr)

        # Logistic regression
        lr = LogisticRegression(**LR_PARAMS)
        lr.fit(sc.transform(X_tr), y_tr)
        yp = lr.predict(sc.transform(X_te))
        results["logreg"]["bal_acc"].append(balanced_accuracy_score(y_te, yp))
        results["logreg"]["macro_f1"].append(f1_score(y_te, yp, average="macro", zero_division=0))
        for i, c in enumerate(le.classes_):
            results["logreg"][f"f1_{c}"].append(f1_score(y_te == i, yp == i, zero_division=0))

        # LightGBM
        clf = LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_tr, y_tr)
        yp2 = clf.predict(X_te)
        results["lgbm"]["bal_acc"].append(balanced_accuracy_score(y_te, yp2))
        results["lgbm"]["macro_f1"].append(f1_score(y_te, yp2, average="macro", zero_division=0))
        for i, c in enumerate(le.classes_):
            results["lgbm"][f"f1_{c}"].append(f1_score(y_te == i, yp2 == i, zero_division=0))

    return {clf: {k: np.mean(v) for k, v in m.items()} for clf, m in results.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # B1-only patches (baseline)
    patches_b1 = load_labels(combined_b12=False)
    print(f"FA4 patches B1-only: {len(patches_b1)}")
    print(patches_b1["label"].value_counts())

    # B1+B2 combined patches (for B12 SupCon eval)
    patches_b12 = load_labels(combined_b12=True)
    print(f"\nFA4 patches B1+B2: {len(patches_b12)}")
    print(patches_b12["label"].value_counts())

    # Load feature matrices (indexed by underscore filename)
    feat_mats = {
        "cp":      pd.read_csv(DATA / "ae_results/features/cellprofiler/ds1.csv").set_index("filename"),
        "ilastik": pd.read_csv(DATA / "ae_results/features/ilastik/ds1.csv").set_index("filename"),
    }

    print("\nLoading binary SupCon latents (B1 eval)...")
    sc_z_b1 = load_supcon_latents(patches_b1)

    print("\nLoading B12 lat32 SupCon latents...")
    sc_z_b12 = load_b12_latents(patches_b12)

    all_rows = []

    # ── B1-only methods ──────────────────────────────────────────────────
    for method, feat_df, patches in [
        ("supcon_bin32_b1",  sc_z_b1,          patches_b1),
        ("cp_b1",            feat_mats["cp"],   patches_b1),
        ("ilastik_b1",       feat_mats["ilastik"], patches_b1),
    ]:
        if isinstance(feat_df, pd.DataFrame) and feat_df.index.name == "filename":
            f = feat_df.reindex(patches["fn"])
        else:
            f = feat_df  # already aligned

        mask = f.notna().all(axis=1)
        X    = f[mask].values
        y    = patches.loc[mask.values, "label"].values
        print(f"\n{method}: {X.shape[0]} patches, {X.shape[1]} features", flush=True)

        cv_res = run_cv(X, y, FA4)
        for clf, metrics in cv_res.items():
            row = {"method": method, "classifier": clf, **{k: round(v, 4) for k, v in metrics.items()}}
            all_rows.append(row)
            print(f"  {clf:8s}  bal_acc={metrics['bal_acc']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
                  f"NA={metrics['f1_Nascent Adhesion']:.3f}  FC={metrics['f1_focal complex']:.3f}  "
                  f"FA={metrics['f1_focal adhesion']:.3f}  Fib={metrics['f1_fibrillar adhesion']:.3f}")

    # ── B1+B2 methods ────────────────────────────────────────────────────
    for method, feat_df, patches in [
        ("cp_b12",      feat_mats["cp"],      patches_b12),
        ("ilastik_b12", feat_mats["ilastik"], patches_b12),
    ]:
        if isinstance(feat_df, pd.DataFrame) and feat_df.index.name == "filename":
            f = feat_df.reindex(patches["fn"])
        else:
            f = feat_df

        mask = f.notna().all(axis=1)
        X    = f[mask].values
        y    = patches.loc[mask.values, "label"].values
        print(f"\n{method}: {X.shape[0]} patches, {X.shape[1]} features", flush=True)

        cv_res = run_cv(X, y, FA4)
        for clf, metrics in cv_res.items():
            row = {"method": method, "classifier": clf, **{k: round(v, 4) for k, v in metrics.items()}}
            all_rows.append(row)
            print(f"  {clf:8s}  bal_acc={metrics['bal_acc']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
                  f"NA={metrics['f1_Nascent Adhesion']:.3f}  FC={metrics['f1_focal complex']:.3f}  "
                  f"FA={metrics['f1_focal adhesion']:.3f}  Fib={metrics['f1_fibrillar adhesion']:.3f}")

    # ── B12 FA4 SupCon lat32 (runs only when models are ready) ───────────
    if isinstance(sc_z_b12, pd.DataFrame) and len(sc_z_b12) > 0:
        f    = sc_z_b12
        mask = f.notna().all(axis=1)
        X    = f[mask].values
        y    = patches_b12.loc[mask.values, "label"].values
        print(f"\nsupcon_fa4_lat32_b12: {X.shape[0]} patches, {X.shape[1]} features", flush=True)
        cv_res = run_cv(X, y, FA4)
        for clf, metrics in cv_res.items():
            row = {"method": "supcon_fa4_lat32_b12", "classifier": clf,
                   **{k: round(v, 4) for k, v in metrics.items()}}
            all_rows.append(row)
            print(f"  {clf:8s}  bal_acc={metrics['bal_acc']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
                  f"NA={metrics['f1_Nascent Adhesion']:.3f}  FC={metrics['f1_focal complex']:.3f}  "
                  f"FA={metrics['f1_focal adhesion']:.3f}  Fib={metrics['f1_fibrillar adhesion']:.3f}")

    df = pd.DataFrame(all_rows)
    out = OUT_DIR / "fa4_cv_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
