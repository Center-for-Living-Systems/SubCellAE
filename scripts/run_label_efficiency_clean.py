#!/usr/bin/env python3
"""
run_label_efficiency_clean.py

Evaluate the 48 clean label-efficiency SupCon AE runs.

For each run (le_cX_npiY_rZ):
  - Train LGBM on the K annotation labels (same labels that shaped the SupCon loss)
  - Test on ALL annotated patches from held-out frames (full Annabel 539-label set)
  - Record balanced accuracy

Output:
  results/le_clean_results.csv
  results/le_clean_curve.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
REPO   = Path(__file__).resolve().parents[1]
DATA   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LE_DIR = DATA / "ae_results" / "contrastive_run" / "le_clean"
LAB_DIR = DATA / "labelling"

# Full Annabel ground-truth labels (frames 0–3, 539 patches)
FULL_ANN_FILE = LAB_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"

# Latent columns used as features
Z_COLS = [f"z_{i}" for i in range(12)]

CFG_FRAMES = {
    0: {"train": [0],       "test": [1, 2, 3]},
    1: {"train": [0, 1],    "test": [2, 3]},
    2: {"train": [0, 1, 2], "test": [3]},
}

NPI_ORDER = ["10", "25", "50", "75", "100", "all"]

# ---------------------------------------------------------------------------

def parse_run_name(name: str) -> dict:
    """le_c1_npi75_r2  →  {cfg:1, npi:'75', repeat:2}"""
    m = re.match(r"le_c(\d+)_npi(\w+)_r(\d+)$", name)
    if not m:
        raise ValueError(f"Unexpected run name: {name}")
    return {"cfg": int(m.group(1)), "npi": m.group(2), "repeat": int(m.group(3))}


def extract_frame(filename: str) -> int:
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else -1


def run_one(run_dir: Path, full_ann: pd.DataFrame,
            ann_dir: Path | None = None) -> dict | None:
    meta    = parse_run_name(run_dir.name)
    cfg     = meta["cfg"]
    frames  = CFG_FRAMES[cfg]
    _ann_dir = ann_dir if ann_dir is not None else (LAB_DIR / "le_clean")
    ann_csv = _ann_dir / f"{run_dir.name}.csv"
    lat_csv = run_dir / "latents.csv"

    if not lat_csv.exists():
        print(f"  [skip] no latents: {run_dir.name}")
        return None
    if not ann_csv.exists():
        print(f"  [skip] no annotation CSV: {ann_csv}")
        return None

    latents = pd.read_csv(lat_csv)
    latents["frame"] = latents["filename"].apply(extract_frame)
    ann_train = pd.read_csv(ann_csv)

    # ── train LGBM ──────────────────────────────────────────────────────────
    train_latents = latents[latents["frame"].isin(frames["train"])].copy()
    train_labeled = train_latents.merge(ann_train[["filename", "label"]],
                                        on="filename", how="inner")
    if len(train_labeled) == 0:
        print(f"  [skip] no train labels: {run_dir.name}")
        return None

    le = LabelEncoder()
    y_train = le.fit_transform(train_labeled["label"])
    X_train = train_labeled[Z_COLS].values

    w_train = compute_sample_weight("balanced", y_train)
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=w_train)

    # ── evaluate on held-out frames ─────────────────────────────────────────
    test_latents = latents[latents["frame"].isin(frames["test"])].copy()
    test_labeled = test_latents.merge(full_ann[["filename", "label"]],
                                      on="filename", how="inner")
    if len(test_labeled) == 0:
        print(f"  [skip] no test labels: {run_dir.name}")
        return None

    y_test = le.transform(test_labeled["label"])
    X_test = test_labeled[Z_COLS].values
    y_pred = clf.predict(X_test)
    bacc   = balanced_accuracy_score(y_test, y_pred)

    # Per-class metrics: LabelEncoder sorts alphabetically so
    # class 0 = "No adhesion", class 1 = "adhesion"
    recalls    = recall_score(   y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
    precisions = precision_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
    noad_recall    = float(recalls[0])
    adh_recall     = float(recalls[1])
    noad_precision = float(precisions[0])
    adh_precision  = float(precisions[1])

    # ── training performance (on the selected npi labels) ───────────────────
    y_pred_train = clf.predict(X_train)
    bacc_train   = float(balanced_accuracy_score(y_train, y_pred_train))

    # ── validation: remaining labels from training frames not used for training
    all_train_labeled = train_latents.merge(
        full_ann[["filename", "label"]], on="filename", how="inner"
    )
    val_labeled = all_train_labeled[
        ~all_train_labeled["filename"].isin(ann_train["filename"])
    ]
    if len(val_labeled) > 0:
        y_val      = le.transform(val_labeled["label"])
        y_pred_val = clf.predict(val_labeled[Z_COLS].values)
        bacc_val   = float(balanced_accuracy_score(y_val, y_pred_val))
        n_val      = len(val_labeled)
        rec_val    = recall_score(   y_val, y_pred_val, labels=[0, 1], average=None, zero_division=0)
        prec_val   = precision_score(y_val, y_pred_val, labels=[0, 1], average=None, zero_division=0)
        noad_recall_val    = float(rec_val[0])
        adh_recall_val     = float(rec_val[1])
        noad_precision_val = float(prec_val[0])
        adh_precision_val  = float(prec_val[1])
    else:
        bacc_val = noad_recall_val = adh_recall_val = float("nan")
        noad_precision_val = adh_precision_val = float("nan")
        n_val    = 0

    return {
        "run":                run_dir.name,
        "cfg":                cfg,
        "npi":                meta["npi"],
        "repeat":             meta["repeat"],
        "k_train":            len(frames["train"]),
        "n_train":            len(train_labeled),
        "n_val":              n_val,
        "n_test":             len(test_labeled),
        "balanced_acc":       bacc,
        "adh_recall":         adh_recall,
        "noad_recall":        noad_recall,
        "adh_precision":      adh_precision,
        "noad_precision":     noad_precision,
        "bacc_train":         bacc_train,
        "bacc_val":           bacc_val,
        "adh_recall_val":     adh_recall_val,
        "noad_recall_val":    noad_recall_val,
        "adh_precision_val":  adh_precision_val,
        "noad_precision_val": noad_precision_val,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=None,
                    help="Override AE results directory (default: le_clean)")
    ap.add_argument("--ann-dir", default=None,
                    help="Override annotation CSV directory (default: le_clean subdir)")
    ap.add_argument("--out-prefix", default="le_clean",
                    help="Prefix for output CSV and PNG files (default: le_clean)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir) if args.exp_dir else LE_DIR
    ann_dir = Path(args.ann_dir) if args.ann_dir else (LAB_DIR / "le_clean")

    full_ann = pd.read_csv(FULL_ANN_FILE)

    run_dirs = sorted(exp_dir.glob("le_c*_npi*_r*"))
    print(f"Found {len(run_dirs)} run directories in {exp_dir}")

    records = []
    for rd in run_dirs:
        r = run_one(rd, full_ann, ann_dir=ann_dir)
        if r:
            records.append(r)
    print(f"Evaluated {len(records)} runs")

    df = pd.DataFrame(records)
    prefix  = args.out_prefix
    out_csv = REPO / "results" / f"{prefix}_results.csv"
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # ── summary table ────────────────────────────────────────────────────────
    agg_cols = ["balanced_acc", "adh_recall", "noad_recall",
                "adh_precision", "noad_precision", "bacc_train", "bacc_val",
                "adh_recall_val", "noad_recall_val",
                "adh_precision_val", "noad_precision_val"]
    summary = (df.groupby(["cfg", "npi", "k_train"])[agg_cols]
                 .agg(["mean", "std"])
                 .reset_index())
    summary.columns = ["cfg", "npi", "k_train"] + [
        f"{m}_{s}" for m in agg_cols for s in ("mean", "std")
    ]
    summary["npi_order"] = summary["npi"].apply(
        lambda x: NPI_ORDER.index(x) if x in NPI_ORDER else 99)
    summary = summary.sort_values(["cfg", "npi_order"])
    summary["mean_pct"]     = (summary["balanced_acc_mean"] * 100).round(1)
    summary["std_pct"]      = (summary["balanced_acc_std"]  * 100).round(1)
    print("\n" + summary[["cfg","k_train","npi","mean_pct","std_pct"]].to_string(index=False))

    # ── plot ─────────────────────────────────────────────────────────────────
    # Three curves per panel: balanced accuracy, adhesion recall, no-adhesion recall
    CURVES = [
        ("balanced_acc", "Balanced accuracy",   "o-",  "#333333", 2.0),
        ("adh_recall",   "Adhesion recall",     "s--", "#2ca02c", 1.8),
        ("noad_recall",  "No-adhesion recall",  "^:", "#1f77b4", 1.8),
    ]
    labels_cfg = {0: "cfg0: 1 train / 3 test frames",
                  1: "cfg1: 2 train / 2 test frames",
                  2: "cfg2: 3 train / 1 test frame"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor="white")

    for ax, cfg in zip(axes, [0, 1, 2]):
        s = summary[summary["cfg"] == cfg].copy().sort_values("npi_order")
        x = np.arange(len(s))

        for col, label, fmt, color, lw in CURVES:
            means = s[f"{col}_mean"].values * 100
            stds  = s[f"{col}_std"].values  * 100
            ax.errorbar(x, means, yerr=stds, fmt=fmt, color=color,
                        capsize=3, linewidth=lw, markersize=6, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(s["npi"].tolist(), fontsize=9)
        ax.set_xlabel("Labels per image (npi)", fontsize=10)
        ax.set_title(labels_cfg[cfg], fontsize=10, fontweight="bold")
        ax.set_ylim(40, 108)
        ax.axhline(90, color="#CCCCCC", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate balanced accuracy at each point
        for xi, (_, row) in zip(x, s.iterrows()):
            v = row["balanced_acc_mean"] * 100
            e = row["balanced_acc_std"]  * 100
            ax.text(xi, v + e + 1.2, f"{v:.0f}%",
                    ha="center", fontsize=7, color="#333333")

    axes[0].set_ylabel("Recall / balanced accuracy (%)", fontsize=10)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Label Efficiency — Clean Experiment (vinc / control)\n"
                 "SupCon AE + LGBM · image-held-out · 3 repeats\n"
                 "Adhesion recall = sensitivity  ·  No-adhesion recall = specificity",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    out_fig = REPO / "results" / f"{prefix}_curve.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_fig}")

    # ── precision plot ────────────────────────────────────────────────────────
    CURVES_PREC = [
        ("balanced_acc",   "Balanced accuracy",     "o-",  "#333333", 2.0),
        ("adh_precision",  "Adhesion precision",    "s--", "#d62728", 1.8),
        ("noad_precision", "No-adhesion precision", "^:",  "#9467bd", 1.8),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor="white")

    for ax, cfg in zip(axes2, [0, 1, 2]):
        s = summary[summary["cfg"] == cfg].copy().sort_values("npi_order")
        x = np.arange(len(s))

        for col, label, fmt, color, lw in CURVES_PREC:
            means = s[f"{col}_mean"].values * 100
            stds  = s[f"{col}_std"].values  * 100
            ax.errorbar(x, means, yerr=stds, fmt=fmt, color=color,
                        capsize=3, linewidth=lw, markersize=6, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(s["npi"].tolist(), fontsize=9)
        ax.set_xlabel("Labels per image (npi)", fontsize=10)
        ax.set_title(labels_cfg[cfg], fontsize=10, fontweight="bold")
        ax.set_ylim(40, 108)
        ax.axhline(90, color="#CCCCCC", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

        for xi, (_, row) in zip(x, s.iterrows()):
            v = row["balanced_acc_mean"] * 100
            e = row["balanced_acc_std"]  * 100
            ax.text(xi, v + e + 1.2, f"{v:.0f}%",
                    ha="center", fontsize=7, color="#333333")

    axes2[0].set_ylabel("Precision / balanced accuracy (%)", fontsize=10)
    axes2[0].legend(fontsize=9, loc="lower right")
    fig2.suptitle("Label Efficiency — Clean Experiment (vinc / control)\n"
                  "SupCon AE + LGBM · image-held-out · 3 repeats\n"
                  "Adhesion precision = PPV  ·  No-adhesion precision = NPV",
                  fontsize=10, fontweight="bold")
    fig2.tight_layout()
    out_fig2 = REPO / "results" / f"{prefix}_curve_precision.png"
    fig2.savefig(out_fig2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved: {out_fig2}")

    # ── train / val / test balanced-accuracy plot ─────────────────────────────
    # val is NaN for npi="all" (no labels left over); skip those points
    CURVES_TVT = [
        ("bacc_train",   "Train BAcc",      "o-",  "#e6550d", 1.8),
        ("bacc_val",     "Val BAcc\n(remaining same-image labels)", "s--", "#fd8d3c", 1.8),
        ("balanced_acc", "Test BAcc",        "^-",  "#333333", 2.0),
    ]

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor="white")

    for ax, cfg in zip(axes3, [0, 1, 2]):
        s = summary[summary["cfg"] == cfg].copy().sort_values("npi_order")
        x = np.arange(len(s))

        for col, label, fmt, color, lw in CURVES_TVT:
            means = s[f"{col}_mean"].values * 100
            stds  = s[f"{col}_std"].values  * 100
            # mask NaN (npi=all for val)
            valid = ~np.isnan(means)
            ax.errorbar(x[valid], means[valid], yerr=stds[valid],
                        fmt=fmt, color=color, capsize=3,
                        linewidth=lw, markersize=6, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(s["npi"].tolist(), fontsize=9)
        ax.set_xlabel("Labels per image (npi)", fontsize=10)
        ax.set_title(labels_cfg[cfg], fontsize=10, fontweight="bold")
        ax.set_ylim(40, 108)
        ax.axhline(90, color="#CCCCCC", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate test BAcc
        for xi, (_, row) in zip(x, s.iterrows()):
            v = row["balanced_acc_mean"] * 100
            e = row["balanced_acc_std"]  * 100
            ax.text(xi, v + e + 1.2, f"{v:.0f}%",
                    ha="center", fontsize=7, color="#333333")

    axes3[0].set_ylabel("Balanced accuracy (%)", fontsize=10)
    axes3[0].legend(fontsize=9, loc="lower right")
    fig3.suptitle(
        "Label Efficiency — Train / Val / Test Balanced Accuracy\n"
        "SupCon AE + LGBM · image-held-out · 3 repeats\n"
        "Val = remaining labeled patches from training frame(s) not selected for training",
        fontsize=10, fontweight="bold",
    )
    fig3.tight_layout()
    out_fig3 = REPO / "results" / f"{prefix}_curve_trainvaltest.png"
    fig3.savefig(out_fig3, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"Saved: {out_fig3}")

    # ── val vs test recall & precision breakdown ──────────────────────────────
    # Solid = test, dashed = val (same color per metric); NaN skipped for val
    CURVES_VT = [
        # (test_col,      val_col,              label,             color)
        ("balanced_acc", "bacc_val",           "BAcc",            "#333333"),
        ("adh_recall",   "adh_recall_val",     "Adh recall",      "#2ca02c"),
        ("noad_recall",  "noad_recall_val",     "NoAd recall",     "#1f77b4"),
        ("adh_precision","adh_precision_val",   "Adh precision",   "#d62728"),
        ("noad_precision","noad_precision_val", "NoAd precision",  "#9467bd"),
    ]

    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True, facecolor="white")

    for ax, cfg in zip(axes4, [0, 1, 2]):
        s = summary[summary["cfg"] == cfg].copy().sort_values("npi_order")
        x = np.arange(len(s))

        for test_col, val_col, label, color in CURVES_VT:
            t_means = s[f"{test_col}_mean"].values * 100
            t_stds  = s[f"{test_col}_std"].values  * 100
            v_means = s[f"{val_col}_mean"].values  * 100
            v_stds  = s[f"{val_col}_std"].values   * 100

            ax.errorbar(x, t_means, yerr=t_stds, fmt="o-", color=color,
                        capsize=3, linewidth=1.8, markersize=5,
                        label=f"{label} (test)")
            valid = ~np.isnan(v_means)
            ax.errorbar(x[valid], v_means[valid], yerr=v_stds[valid],
                        fmt="s--", color=color, capsize=3,
                        linewidth=1.4, markersize=4, alpha=0.6,
                        label=f"{label} (val)")

        ax.set_xticks(x)
        ax.set_xticklabels(s["npi"].tolist(), fontsize=9)
        ax.set_xlabel("Labels per image (npi)", fontsize=10)
        ax.set_title(labels_cfg[cfg], fontsize=10, fontweight="bold")
        ax.set_ylim(40, 108)
        ax.axhline(90, color="#CCCCCC", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

    axes4[0].set_ylabel("Score (%)", fontsize=10)
    axes4[0].legend(fontsize=7, loc="lower right", ncol=2)
    fig4.suptitle(
        "Label Efficiency — Val vs Test: Recall & Precision Breakdown\n"
        "SupCon AE + LGBM · solid = test frames · dashed = val (remaining same-image labels)\n"
        "Val NaN at npi=all (no labels left over)",
        fontsize=10, fontweight="bold",
    )
    fig4.tight_layout()
    out_fig4 = REPO / "results" / f"{prefix}_curve_val_vs_test.png"
    fig4.savefig(out_fig4, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig4)
    print(f"Saved: {out_fig4}")


if __name__ == "__main__":
    main()
