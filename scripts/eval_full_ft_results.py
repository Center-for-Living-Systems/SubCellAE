#!/usr/bin/env python3
"""
eval_full_ft_results.py

Re-evaluate GBM metrics from pre-computed full_ft latents.csv files.
Each sbatch job only saved the last fraction's results.csv; this script
aggregates all fracs without re-training.

Covers:
  ft_ycomp_corrected_s3v1_full_ft        (target-only, fracs 10/25/50/75%)
  ft_ycomp_combined_s3v1_full_ft         (target-only, fracs 10/25/50/75%)
  ft_pfak_combined_s3v1_full_ft          (target-only, fracs 10/25/50/75%)
  ft_ycomp_combined_s3v1_full_ft_ctrl_plus (ctrl+ycomp, fracs 0/10/25/50/75%)
  ft_pfak_combined_s3v1_full_ft_ctrl_plus  (ctrl+pfak, fracs 0/10/25/50/75%)
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR   = DATA_ROOT / "ae_results" / "contrastive_run"
LABEL_DIR = DATA_ROOT / "labelling"

YCOMP_LABEL_FILE = LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv"
PFAK_LABEL_FILE  = LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv"
CTRL_LABEL_FILE  = LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"

LABEL_ORDER = ["No adhesion", "adhesion"]
TEST_FRAC   = 0.20
SEED        = 42
Z_COLS      = [f"z_{i}" for i in range(12)]


def _to_binary(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def _unique_id(filename: str) -> str:
    return re.sub(r"_(f\d+x\d+y\d+ps\d+\.tiff?)", r"-\1", filename)


def _load_ycomp_labels() -> pd.DataFrame:
    df = pd.read_csv(YCOMP_LABEL_FILE)
    df = df[df["filename"].str.startswith("ycomp_")].copy()
    df["binary_label"] = df["label"].apply(_to_binary)
    df["unique_ID"] = df["filename"].apply(_unique_id)
    return df[["filename", "unique_ID", "binary_label"]].reset_index(drop=True)


def _load_pfak_labels() -> pd.DataFrame:
    df = pd.read_csv(PFAK_LABEL_FILE)
    df["binary_label"] = df["label"].apply(_to_binary)
    df["unique_ID"] = df["filename"].apply(_unique_id)
    return df[["filename", "unique_ID", "binary_label"]].reset_index(drop=True)


def _load_ctrl_labels() -> pd.DataFrame:
    df = pd.read_csv(CTRL_LABEL_FILE)
    df["binary_label"] = df["label"].apply(_to_binary)
    return df[["filename", "binary_label"]].reset_index(drop=True)


def _split(df: pd.DataFrame):
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_FRAC,
        stratify=df["binary_label"], random_state=SEED,
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _gbm_eval(z_train, y_train, z_test, y_test):
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        random_state=SEED,
    )
    clf.fit(z_train, y_train)
    y_pred = clf.predict(z_test)
    return {
        "bal_acc": balanced_accuracy_score(y_test, y_pred),
        "f1":      f1_score(y_test, y_pred, pos_label="adhesion"),
        "y_pred":  y_pred,
    }


def _save_efficiency_curve(results: list[dict], out_path: Path, title: str):
    fracs_pct = [int(round(r["frac"] * 100)) for r in results]
    bal_accs  = [r["bal_acc"] for r in results]
    f1s       = [r["f1"]      for r in results]
    n_targets = [r.get("n_target", r["n_train"]) for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fracs_pct, bal_accs, "o-",  label="Balanced Accuracy")
    ax.plot(fracs_pct, f1s,      "s--", label="F1 (adhesion)")
    for x, y, n in zip(fracs_pct, bal_accs, n_targets):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("% of target train pool used")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── per-case eval ─────────────────────────────────────────────────────────────

def eval_ycomp_full_ft(run_key: str, fracs: list[float], add_ctrl: bool):
    out_dir = RUN_DIR / run_key
    print(f"\n{'='*60}\n{run_key}\n{'='*60}")

    labels = _load_ycomp_labels()
    train_pool, test_df = _split(labels)
    ctrl_labels = _load_ctrl_labels() if add_ctrl else None

    results = []
    for frac in fracs:
        frac_name = f"frac{int(frac * 100):03d}"
        frac_dir  = out_dir / frac_name
        if not (frac_dir / "latents.csv").exists():
            print(f"  [{frac_name}] latents.csv not found — skip")
            continue

        lat = pd.read_csv(frac_dir / "latents.csv")
        lat_ycomp = lat[lat["filename"].str.startswith("ycomp_")].copy()

        n_tgt = max(1, int(round(frac * len(train_pool)))) if frac > 0 else 0

        if add_ctrl:
            lat_ctrl = lat[lat["filename"].str.startswith("control_")].copy()
            ctrl_matched = ctrl_labels.merge(lat_ctrl[["filename"] + Z_COLS],
                                             on="filename", how="inner")
            if n_tgt > 0:
                rng = np.random.RandomState(SEED)
                idx = rng.choice(len(train_pool), size=n_tgt, replace=False)
                frac_labels_df = train_pool.iloc[idx].copy()
                tgt_lat = frac_labels_df.merge(lat_ycomp[["filename"] + Z_COLS],
                                               on="filename", how="inner")
                train_lat = pd.concat([ctrl_matched, tgt_lat], ignore_index=True)
            else:
                train_lat = ctrl_matched
        else:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(len(train_pool), size=n_tgt, replace=False)
            frac_labels_df = train_pool.iloc[idx].copy()
            train_lat = frac_labels_df.merge(lat_ycomp[["filename"] + Z_COLS],
                                             on="filename", how="inner")

        test_lat = test_df.merge(lat_ycomp[["filename"] + Z_COLS],
                                 on="filename", how="inner")

        if len(train_lat) == 0 or len(test_lat) == 0:
            print(f"  [{frac_name}] WARNING: train={len(train_lat)} test={len(test_lat)} — skip")
            continue

        metrics = _gbm_eval(
            train_lat[Z_COLS].values, train_lat["binary_label"].values,
            test_lat[Z_COLS].values,  test_lat["binary_label"].values,
        )
        n_train_total = len(train_lat)
        print(f"  [{frac_name}] n_train={n_train_total} (tgt={n_tgt})  "
              f"bal_acc={metrics['bal_acc']:.3f}  f1={metrics['f1']:.3f}")

        cm_fig, cm_ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(test_lat["binary_label"].values, metrics["y_pred"],
                             labels=LABEL_ORDER),
            display_labels=LABEL_ORDER,
        ).plot(ax=cm_ax, colorbar=False)
        cm_ax.set_title(f"{run_key} {frac_name}")
        cm_fig.savefig(frac_dir / "confusion.png", dpi=120)
        plt.close(cm_fig)

        results.append({"frac": frac, "n_train": n_train_total, "n_target": n_tgt,
                        "bal_acc": metrics["bal_acc"], "f1": metrics["f1"]})

    if not results:
        print("  No results collected.")
        return

    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    suffix = "ctrl+ycomp" if add_ctrl else "ycomp only"
    _save_efficiency_curve(results, out_dir / "efficiency_curve.png",
                           f"{run_key}\nGBM on {suffix} latents")
    print(f"  Saved results.csv with {len(results)} rows → {out_dir}")


def eval_pfak_full_ft(run_key: str, fracs: list[float], add_ctrl: bool):
    out_dir = RUN_DIR / run_key
    print(f"\n{'='*60}\n{run_key}\n{'='*60}")

    labels = _load_pfak_labels()
    train_pool, test_df = _split(labels)
    ctrl_labels = _load_ctrl_labels() if add_ctrl else None

    results = []
    for frac in fracs:
        frac_name = f"frac{int(frac * 100):03d}"
        frac_dir  = out_dir / frac_name
        if not (frac_dir / "latents.csv").exists():
            print(f"  [{frac_name}] latents.csv not found — skip")
            continue

        lat = pd.read_csv(frac_dir / "latents.csv")
        # pfak control patches: condition_name="pfak_control" (or only ctrl_ if no vinc present)
        if "pfak_control" in lat.get("condition_name", pd.Series()).unique():
            lat_pfak = lat[lat["condition_name"] == "pfak_control"].copy()
        else:
            lat_pfak = lat[lat["filename"].str.startswith("control_")].copy()

        n_tgt = max(1, int(round(frac * len(train_pool)))) if frac > 0 else 0

        if add_ctrl:
            # vinc ctrl latents are in the same latents.csv (condition_name="vinc_control")
            if "vinc_control" in lat.get("condition_name", pd.Series()).unique():
                lat_vinc = lat[lat["condition_name"] == "vinc_control"].copy()
            else:
                lat_vinc = lat[lat["filename"].str.startswith("control_")].copy()
            ctrl_matched = ctrl_labels.merge(lat_vinc[["filename"] + Z_COLS],
                                             on="filename", how="inner")
            if n_tgt > 0:
                rng = np.random.RandomState(SEED)
                idx = rng.choice(len(train_pool), size=n_tgt, replace=False)
                frac_labels_df = train_pool.iloc[idx].copy()
                tgt_matched = frac_labels_df.merge(lat_pfak[["filename"] + Z_COLS],
                                                   on="filename", how="inner")
                train_lat = pd.concat([ctrl_matched, tgt_matched], ignore_index=True)
            else:
                train_lat = ctrl_matched
        else:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(len(train_pool), size=n_tgt, replace=False)
            frac_labels_df = train_pool.iloc[idx].copy()
            train_lat = frac_labels_df.merge(lat_pfak[["filename"] + Z_COLS],
                                             on="filename", how="inner")

        test_lat = test_df.merge(lat_pfak[["filename"] + Z_COLS],
                                 on="filename", how="inner")

        if len(train_lat) == 0 or len(test_lat) == 0:
            print(f"  [{frac_name}] WARNING: train={len(train_lat)} test={len(test_lat)} — skip")
            continue

        metrics = _gbm_eval(
            train_lat[Z_COLS].values, train_lat["binary_label"].values,
            test_lat[Z_COLS].values,  test_lat["binary_label"].values,
        )
        n_train_total = len(train_lat)
        print(f"  [{frac_name}] n_train={n_train_total} (tgt={n_tgt})  "
              f"bal_acc={metrics['bal_acc']:.3f}  f1={metrics['f1']:.3f}")

        cm_fig, cm_ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(test_lat["binary_label"].values, metrics["y_pred"],
                             labels=LABEL_ORDER),
            display_labels=LABEL_ORDER,
        ).plot(ax=cm_ax, colorbar=False)
        cm_ax.set_title(f"{run_key} {frac_name}")
        cm_fig.savefig(frac_dir / "confusion.png", dpi=120)
        plt.close(cm_fig)

        results.append({"frac": frac, "n_train": n_train_total, "n_target": n_tgt,
                        "bal_acc": metrics["bal_acc"], "f1": metrics["f1"]})

    if not results:
        print("  No results collected.")
        return

    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    suffix = "ctrl+pfak" if add_ctrl else "pfak only"
    _save_efficiency_curve(results, out_dir / "efficiency_curve.png",
                           f"{run_key}\nGBM on {suffix} latents")
    print(f"  Saved results.csv with {len(results)} rows → {out_dir}")


def main():
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))

    eval_ycomp_full_ft("ft_ycomp_corrected_s3v1_full_ft",
                       fracs=[0.10, 0.25, 0.50, 0.75], add_ctrl=False)
    eval_ycomp_full_ft("ft_ycomp_combined_s3v1_full_ft",
                       fracs=[0.10, 0.25, 0.50, 0.75], add_ctrl=False)
    eval_pfak_full_ft("ft_pfak_combined_s3v1_full_ft",
                      fracs=[0.10, 0.25, 0.50, 0.75], add_ctrl=False)
    eval_ycomp_full_ft("ft_ycomp_combined_s3v1_full_ft_ctrl_plus",
                       fracs=[0.00, 0.10, 0.25, 0.50, 0.75], add_ctrl=True)
    eval_pfak_full_ft("ft_pfak_combined_s3v1_full_ft_ctrl_plus",
                      fracs=[0.00, 0.10, 0.25, 0.50, 0.75], add_ctrl=True)

    print("\nAll done.")


if __name__ == "__main__":
    main()
