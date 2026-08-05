#!/usr/bin/env python3
"""
run_lgbm_crossds_labels.py

Retrain the 2-class LightGBM classifier by adding ppax + pfak labeled latents
to the vinc training set (without re-training the AE encoder).

For each split (s1v3, s2v2, s3v1):
  1. Load z_recon latents for labeled vinc patches (train split only)
  2. Load z_recon latents for labeled ppax/pfak patches from blind_test CSVs
  3. Retrain LightGBM on combined labeled set
  4. Evaluate on all blind-test evaluation sets
  5. Save results to {result_dir}/fa_cls_zrecon_crossds_labels/

This compares to the original vinc-only LightGBM to measure cross-dataset
label impact without encoder fine-tuning.

Usage:
  python scripts/run_lgbm_crossds_labels.py [--split s2v2]
  python scripts/run_lgbm_crossds_labels.py  # runs all 3 splits
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR = DATA_ROOT / "labelling"
RUN_DIR   = DATA_ROOT / "ae_results" / "contrastive_run"

LABEL_ORDER = ["No adhesion", "adhesion"]
ADHESION_CLASSES = {"focal adhesion", "Nascent Adhesion", "fibrillar adhesion", "focal complex"}

EVAL_SETS = [
    ("vinc",  "control", "vinc_control_latents.csv"),
    ("vinc",  "ycomp",   "vinc_ycomp_latents.csv"),
    ("ppax",  "control", "ppax_control_latents.csv"),
    ("pfak",  "control", "pfak_control_latents.csv"),
]

# Fine-grained eval label files for ppax/pfak
FINE_LABELS = {
    "ppax_control": LABEL_DIR / "labels_ppax_20260521.csv",
    "pfak_control": LABEL_DIR / "labels_pfak_20260521.csv",
}


def _load_vinc_train_latents(result_dir: Path, split: str) -> pd.DataFrame:
    """Load z_recon latents for vinc labeled patches in the train split."""
    # Latent features live in the blind_test CSV; split/annotation info in predictions_all
    lat  = pd.read_csv(result_dir / "blind_test" / "vinc_control_latents.csv")
    pred = pd.read_csv(result_dir / "fa_cls_zrecon" / "predictions_all.csv")
    # Both share the underscore filename — join directly
    merged = lat.merge(pred[["filename", "split", "annotation_label_name"]],
                       on="filename", how="inner")
    labeled = merged[
        merged["annotation_label_name"].notna() & (merged["split"] == "train")
    ].copy()
    labeled["label_2cls"] = labeled["annotation_label_name"].apply(
        lambda x: "adhesion" if x in ADHESION_CLASSES else "No adhesion"
    )
    z_cols = [c for c in lat.columns if c.startswith("z_")]
    return labeled[z_cols + ["label_2cls"]].copy()


def _load_crossds_latents(result_dir: Path, ds: str, cond: str,
                          label_csv: Path) -> pd.DataFrame:
    """Load blind-test latents for ds/cond and merge with 2-class labels."""
    lats  = pd.read_csv(result_dir / "blind_test" / f"{ds}_{cond}_latents.csv")
    labels = pd.read_csv(label_csv)
    # Convert multi-class to 2-class
    labels = labels[labels["classification"] != "Uncertain"].copy()
    labels["label_2cls"] = labels["classification"].apply(
        lambda c: "adhesion" if c in ADHESION_CLASSES else "No adhesion"
    )
    merged = lats.merge(labels[["unique_ID", "label_2cls"]], on="unique_ID", how="inner")
    z_cols = [c for c in merged.columns if c.startswith("z_")]
    return merged[z_cols + ["label_2cls"]].copy()


def _train_lgbm(X_train: np.ndarray, y_train: np.ndarray):
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=3, class_weight="balanced",
        random_state=42, verbose=-1, n_jobs=1,
    )
    clf.fit(X_train, y_train, feature_name=[f"z_{i}" for i in range(X_train.shape[1])])
    return clf


def _metrics_row(split, eval_key, y_true, y_pred, n):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = rep.get("accuracy", 0.0)
    mf1 = rep.get("macro avg", {}).get("f1-score", 0.0)
    wf1 = rep.get("weighted avg", {}).get("f1-score", 0.0)
    return {"split": split, "eval": eval_key, "n": n,
            "accuracy": round(acc, 4), "macro_f1": round(mf1, 4), "weighted_f1": round(wf1, 4)}


def run_split(split: str):
    result_dir = RUN_DIR / f"annabel_vinc_supcon2_{split}"
    out_dir    = result_dir / "fa_cls_zrecon_crossds_labels"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Split: {split}")
    print(f"{'='*60}")

    # ── Build combined training set ──────────────────────────────────────
    vinc_df = _load_vinc_train_latents(result_dir, split)
    z_cols  = [c for c in vinc_df.columns if c.startswith("z_")]

    # ppax
    ppax_df = _load_crossds_latents(
        result_dir, "ppax", "control",
        LABEL_DIR / "labels_ppax_20260521.csv"
    )
    # pfak
    pfak_df = _load_crossds_latents(
        result_dir, "pfak", "control",
        LABEL_DIR / "labels_pfak_20260521.csv"
    )

    combined = pd.concat([vinc_df, ppax_df, pfak_df], ignore_index=True)
    label_map = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}

    X_train = combined[z_cols].values.astype(np.float32)
    y_train = combined["label_2cls"].map(label_map).values

    n_vinc = len(vinc_df)
    n_ppax = len(ppax_df)
    n_pfak = len(pfak_df)
    print(f"  Train: vinc={n_vinc}  ppax={n_ppax}  pfak={n_pfak}  total={len(combined)}")
    print(f"  Label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    clf = _train_lgbm(X_train, y_train)

    import joblib
    joblib.dump(clf, str(out_dir / "model.pkl"))
    print(f"  Model saved → {out_dir.name}/model.pkl")

    # ── Evaluate ─────────────────────────────────────────────────────────
    rows = []
    for ds, cond, latent_fname in EVAL_SETS:
        lat_path = result_dir / "blind_test" / latent_fname
        if not lat_path.exists():
            continue
        lat = pd.read_csv(lat_path)

        # Load fine-grained labels
        fine_key = f"{ds}_{cond}"
        if fine_key in FINE_LABELS:
            fine_df = pd.read_csv(FINE_LABELS[fine_key])
            fine_df = fine_df[fine_df["classification"] != "Uncertain"].copy()
            fine_df["label_2cls"] = fine_df["classification"].apply(
                lambda c: "adhesion" if c in ADHESION_CLASSES else "No adhesion"
            )
            eval_df = lat.merge(fine_df[["unique_ID", "label_2cls"]], on="unique_ID", how="inner")
        else:
            # For vinc: use existing predictions_all with annotation
            pred_all = result_dir / "fa_cls_zrecon" / "predictions_all.csv"
            pred_df = pd.read_csv(pred_all)
            # Build unique_ID from filename
            pred_df["unique_ID"] = pred_df["filename"].apply(
                lambda f: Path(f).stem.replace("_", "-", 1) + ".tif"
                if "_" in Path(f).stem else f
            )
            # annotation_label_name is already 2-class ("No adhesion" / "adhesion")
            ann = pred_df[pred_df["annotation_label_name"].notna()][["unique_ID","annotation_label_name"]].copy()
            ann["label_2cls"] = ann["annotation_label_name"]
            eval_df = lat.merge(ann[["unique_ID","label_2cls"]], on="unique_ID", how="inner")

        if len(eval_df) == 0:
            continue

        Xe = eval_df[z_cols].values.astype(np.float32)
        ye = eval_df["label_2cls"].map(label_map).values
        yp = clf.predict(Xe)
        n = len(Xe)

        eval_key = f"{ds}_{cond}"
        rep_str = classification_report(ye, yp, target_names=LABEL_ORDER, zero_division=0)
        print(f"\n  [{eval_key}]  n={n}")
        print(rep_str)

        rows.append(_metrics_row(split, eval_key, ye, yp, n))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, choices=["s1v3", "s2v2", "s3v1"],
                    help="Single split; omit to run all 3.")
    args = ap.parse_args()

    splits = [args.split] if args.split else ["s1v3", "s2v2", "s3v1"]

    all_rows = []
    for sp in splits:
        all_rows.extend(run_split(sp))

    summary = pd.DataFrame(all_rows)
    print("\n" + "="*60)
    print("SUMMARY — LightGBM retrained with vinc + ppax + pfak labels")
    print("="*60)
    print(summary.to_string(index=False))

    out_csv = RUN_DIR / "crossds_lgbm_summary.csv"
    summary.to_csv(str(out_csv), index=False)
    print(f"\nSummary → {out_csv.name}")


if __name__ == "__main__":
    main()
