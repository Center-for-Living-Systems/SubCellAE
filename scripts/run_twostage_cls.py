#!/usr/bin/env python3
"""
run_twostage_cls.py

Two-stage classifier pipeline:
  Stage 1: SupCon-2cls LightGBM (No adhesion vs adhesion) — already trained
  Stage 2: 4-class FA subtype LightGBM (Nascent / focal complex / focal adhesion / fibrillar)
            trained on adhesion-only Annabel vinc patches

Runs for all 3 train/val splits (s1v3, s2v2, s3v1), each using z_recon features.
Evaluates on:
  - In-domain val (Annabel vinc control)
  - Blind test: vinc/control, vinc/ycomp, ppax/control, pfak/control  (Margaret labels)
  - Ernest ppax (FA-only labels)

Saves confusion matrices and metrics to:
  <result_dir>/twostage/<eval_set>/

Usage:
  python scripts/run_twostage_cls.py
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR   = DATA_ROOT / "ae_results" / "contrastive_run"
LABEL_DIR = DATA_ROOT / "labelling"

SPLITS = ["s1v3", "s2v2", "s3v1"]

LABEL_ORDER_5 = [
    "No adhesion",
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]
LABEL_ORDER_4 = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]
ADHESION_TYPES = set(LABEL_ORDER_4)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fn_norm(s: str) -> str:
    return Path(s).name


def _patch_to_uid(fname: str) -> str:
    return re.sub(r"_(?=f\d{4})", "-", Path(fname).name, count=1)


def _plot_cm(y_true, y_pred, label_order, title, out_path: Path,
             normalize: bool = True):
    present = [c for c in label_order if c in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(denom > 0, cm.astype(float) / np.where(denom > 0, denom, 1), 0.0)
        fmt = ".2f"
    else:
        cm_plot, fmt = cm, "d"

    n = len(present)
    fig, ax = plt.subplots(figsize=(max(4.5, n * 0.9), max(3.5, n * 0.75)),
                           facecolor="white")
    disp = ConfusionMatrixDisplay(cm_plot, display_labels=present)
    disp.plot(ax=ax, colorbar=False, values_format=fmt)
    ax.set_title(title, fontsize=9, pad=6)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor="white")
    plt.close(fig)


def _metrics_row(y_true, y_pred, label_order) -> dict:
    present = [c for c in label_order if c in set(y_true) | set(y_pred)]
    rep = classification_report(y_true, y_pred, labels=present,
                                target_names=present, zero_division=0,
                                output_dict=True)
    return {
        "accuracy":   rep.get("accuracy", 0.0),
        "macro_f1":   rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
    }


def _apply_pipeline(lat_df: pd.DataFrame, stage1, stage2,
                    label_order_4: list[str]) -> pd.Series:
    """
    Apply two-stage pipeline to a latent DataFrame.
    Returns a Series of predicted labels (5-class: No adhesion + 4 FA subtypes).
    """
    z_cols = [c for c in lat_df.columns if c.startswith("z_")]
    X = lat_df[z_cols].values.astype(np.float32)

    # Stage 1
    s1_pred = stage1.predict(X)          # integer: 0=No adhesion, 1=adhesion
    # Map integer → label
    s1_labels = np.array(["No adhesion" if p == 0 else "adhesion" for p in s1_pred])

    # Stage 2 on adhesion subset
    final = s1_labels.copy().astype(object)
    ad_mask = s1_labels == "adhesion"
    if ad_mask.any():
        X_ad = X[ad_mask]
        s2_pred = stage2.predict(X_ad)  # integer indices into label_order_4
        s2_labels = np.array([label_order_4[int(p)] if int(p) < len(label_order_4)
                               else str(p) for p in s2_pred])
        final[ad_mask] = s2_labels

    return pd.Series(final, index=lat_df.index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load Annabel's original 5-class labels
    ann = pd.read_csv(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv")
    ann["_fn"] = ann["filename"].apply(_fn_norm)

    # Load Margaret's blind-test label CSVs
    marg_vinc = pd.read_csv(LABEL_DIR / "labels_vinc_20260521.csv")
    marg_ppax = pd.read_csv(LABEL_DIR / "labels_ppax_20260521.csv")
    marg_pfak = pd.read_csv(LABEL_DIR / "labels_pfak_20260521.csv")

    # Ernest's ppax labels
    ern = pd.read_csv(LABEL_DIR / "ppax_control_label_Ernest_20260727_1142.csv")
    ern["unique_ID"] = ern["filename"].apply(_patch_to_uid)

    summary_rows = []

    for split in SPLITS:
        result_dir = RUN_DIR / f"annabel_vinc_supcon2_{split}"
        out_base   = result_dir / "twostage"
        print(f"\n{'='*65}")
        print(f"Split: {split}   result_dir: {result_dir.name}")
        print(f"{'='*65}")

        # ── Stage 1: load saved supcon2 LightGBM (2-class) ─────────────
        stage1 = joblib.load(str(result_dir / "fa_cls_zrecon" / "model.pkl"))

        # ── Stage 2: train on Annabel adhesion train patches ────────────
        lat = pd.read_csv(result_dir / "latents.csv")
        lat["_fn"] = lat["filename"].apply(_fn_norm)
        lat = lat.merge(ann[["_fn", "label"]], on="_fn", how="left")

        train_ad = lat[(lat["split"] == "train") &
                       (lat["label"].isin(ADHESION_TYPES))].copy()
        print(f"  Stage-2 training: {len(train_ad)} adhesion patches")
        print("  Class counts:", train_ad["label"].value_counts().to_dict())

        z_cols = [c for c in lat.columns if c.startswith("z_")]
        X2_tr  = train_ad[z_cols].values.astype(np.float32)
        y2_tr  = train_ad["label"].values

        # Build integer encoding consistent with LABEL_ORDER_4
        lo4_present = [c for c in LABEL_ORDER_4 if c in set(y2_tr)]
        lo4_to_int  = {c: i for i, c in enumerate(lo4_present)}
        y2_tr_int   = np.array([lo4_to_int[l] for l in y2_tr])

        stage2 = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=3, class_weight="balanced",
            random_state=42, verbose=-1, n_jobs=1,
        )
        stage2.fit(X2_tr, y2_tr_int)
        # Save it
        out_base.mkdir(exist_ok=True)
        joblib.dump(stage2, str(out_base / "stage2_model.pkl"))
        print(f"  Stage-2 label order: {lo4_present}")

        # ── Helper: evaluate two-stage on a labeled DataFrame ───────────
        def _eval(lat_eval, gt_df, uid_col, label_col, eval_name,
                  exclude_labels=("Uncertain",), id_is_uid=False):
            """
            lat_eval: DataFrame with z_ cols and unique_ID column
            gt_df: ground truth DataFrame
            Returns summary dict
            """
            pred = _apply_pipeline(lat_eval, stage1, stage2, lo4_present)
            pred_df = lat_eval[["unique_ID"]].copy()
            pred_df["pred"] = pred.values

            merged = gt_df.merge(pred_df, on="unique_ID", how="inner")
            if exclude_labels:
                merged = merged[~merged[label_col].isin(exclude_labels)].copy()
            if len(merged) == 0:
                print(f"  [{eval_name}] no matched patches after filtering")
                return None

            y_true = merged[label_col].tolist()
            y_pred = merged["pred"].tolist()
            n_matched = len(merged)

            m = _metrics_row(y_true, y_pred, LABEL_ORDER_5)
            print(f"  [{eval_name}] n={n_matched}  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")
            # print per-class report
            present = [c for c in LABEL_ORDER_5 if c in set(y_true)]
            print(classification_report(y_true, y_pred, labels=present,
                                        target_names=present, zero_division=0))

            # Confusion matrix (normalised)
            out_dir = out_base / eval_name.replace("/", "_").replace(" ", "_")
            _plot_cm(y_true, y_pred, LABEL_ORDER_5,
                     f"{split} | {eval_name} (row-normalised)",
                     out_dir / "confusion_matrix_norm.png", normalize=True)
            _plot_cm(y_true, y_pred, LABEL_ORDER_5,
                     f"{split} | {eval_name} (counts)",
                     out_dir / "confusion_matrix_counts.png", normalize=False)
            pd.DataFrame([{"split": split, "eval": eval_name, **m, "n": n_matched}]).to_csv(
                out_dir / "metrics.csv", index=False)
            return {"split": split, "eval": eval_name, **m, "n": n_matched}

        # ── In-domain val (Annabel vinc ctrl) ───────────────────────────
        val_lat = lat[lat["split"] == "val"].copy()
        val_lat["unique_ID"] = val_lat["filename"].apply(_patch_to_uid)
        # ground truth from Annabel's 5-class labels
        ann_uid = ann.copy()
        ann_uid["unique_ID"] = ann_uid["filename"].apply(_patch_to_uid)
        ann_uid = ann_uid.rename(columns={"label": "label5"})
        r = _eval(val_lat, ann_uid[["unique_ID", "label5"]], "unique_ID", "label5",
                  "indomain_val", exclude_labels=())
        if r: summary_rows.append(r)

        # ── Blind test: vinc/control (Margaret) ─────────────────────────
        blind_lat_vc = pd.read_csv(result_dir / "blind_test" / "vinc_control_latents.csv")
        gt_vc = marg_vinc[marg_vinc["condition"] == "control"].copy()
        r = _eval(blind_lat_vc, gt_vc.rename(columns={"classification": "cls"}),
                  "unique_ID", "cls", "vinc_control_margaret")
        if r: summary_rows.append(r)

        # ── Blind test: vinc/ycomp (Margaret) ───────────────────────────
        blind_lat_vy = pd.read_csv(result_dir / "blind_test" / "vinc_ycomp_latents.csv")
        gt_vy = marg_vinc[marg_vinc["condition"] == "ycomp"].copy()
        r = _eval(blind_lat_vy, gt_vy.rename(columns={"classification": "cls"}),
                  "unique_ID", "cls", "vinc_ycomp_margaret")
        if r: summary_rows.append(r)

        # ── Blind test: ppax/control (Margaret) ─────────────────────────
        blind_lat_pp = pd.read_csv(result_dir / "blind_test" / "ppax_control_latents.csv")
        r = _eval(blind_lat_pp, marg_ppax.rename(columns={"classification": "cls"}),
                  "unique_ID", "cls", "ppax_control_margaret")
        if r: summary_rows.append(r)

        # ── Blind test: pfak/control (Margaret) ─────────────────────────
        blind_lat_pf = pd.read_csv(result_dir / "blind_test" / "pfak_control_latents.csv")
        r = _eval(blind_lat_pf, marg_pfak.rename(columns={"classification": "cls"}),
                  "unique_ID", "cls", "pfak_control_margaret")
        if r: summary_rows.append(r)

        # ── Ernest ppax (FA-only, no No adhesion) ───────────────────────
        pred_pp = _apply_pipeline(blind_lat_pp, stage1, stage2, lo4_present)
        pred_pp_df = blind_lat_pp[["unique_ID"]].copy()
        pred_pp_df["pred"] = pred_pp.values
        ern_merged = ern.merge(pred_pp_df, on="unique_ID", how="inner")
        if len(ern_merged) > 0:
            y_true_e = ern_merged["label"].tolist()
            y_pred_e = ern_merged["pred"].tolist()
            m = _metrics_row(y_true_e, y_pred_e, LABEL_ORDER_4)
            print(f"  [ppax_ernest] n={len(ern_merged)}  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")
            present = [c for c in LABEL_ORDER_4 if c in set(y_true_e)]
            print(classification_report(y_true_e, y_pred_e, labels=present,
                                        target_names=present, zero_division=0))
            out_dir = out_base / "ppax_ernest"
            _plot_cm(y_true_e, y_pred_e, LABEL_ORDER_4,
                     f"{split} | ppax Ernest — stage-2 subtype (row-norm)",
                     out_dir / "confusion_matrix_norm.png", normalize=True)
            _plot_cm(y_true_e, y_pred_e, LABEL_ORDER_4,
                     f"{split} | ppax Ernest — stage-2 subtype (counts)",
                     out_dir / "confusion_matrix_counts.png", normalize=False)
            pd.DataFrame([{"split": split, "eval": "ppax_ernest", **m,
                           "n": len(ern_merged)}]).to_csv(out_dir / "metrics.csv", index=False)
            summary_rows.append({"split": split, "eval": "ppax_ernest", **m,
                                  "n": len(ern_merged)})

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY — Two-stage (SupCon-2cls + 4-class FA subtype)")
    print(f"{'='*65}")
    df = pd.DataFrame(summary_rows)
    cols = ["split", "eval", "n", "accuracy", "macro_f1", "weighted_f1"]
    print(df[cols].to_string(index=False, float_format="{:.3f}".format))

    # Save combined summary
    summary_path = RUN_DIR / "twostage_summary.csv"
    df.to_csv(str(summary_path), index=False)
    print(f"\nSummary → {summary_path.name}")


if __name__ == "__main__":
    main()
