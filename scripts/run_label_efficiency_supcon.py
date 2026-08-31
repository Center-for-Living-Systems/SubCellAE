#!/usr/bin/env python3
"""
run_label_efficiency_supcon.py

Evaluate le_supcon runs (cfg0 only) and compare with le_clean cfg0.
Output:
  results/le_supcon_results.csv
  results/le_supcon_vs_clean_curve.png
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

REPO    = Path(__file__).resolve().parents[1]
DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB_DIR = DATA / "labelling"

SUPCON_DIR = DATA / "ae_results" / "contrastive_run" / "le_supcon"
CLEAN_DIR  = DATA / "ae_results" / "contrastive_run" / "le_clean"

FULL_ANN_FILE = LAB_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"
Z_COLS = [f"z_{i}" for i in range(12)]
NPI_ORDER = ["10", "25", "50", "75", "100", "all"]

CFG_FRAMES = {
    0: {"train": [0], "test": [1, 2, 3]},
}


def extract_frame(filename: str) -> int:
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else -1


def parse_run_name(name: str) -> dict:
    m = re.match(r"le_c(\d+)_npi(\w+)_r(\d+)$", name)
    if not m:
        raise ValueError(f"Unexpected run name: {name}")
    return {"cfg": int(m.group(1)), "npi": m.group(2), "repeat": int(m.group(3))}


def run_one(run_dir: Path, full_ann: pd.DataFrame) -> dict | None:
    meta   = parse_run_name(run_dir.name)
    cfg    = meta["cfg"]
    frames = CFG_FRAMES[cfg]
    ann_csv = LAB_DIR / "le_clean" / f"{run_dir.name}.csv"
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
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)

    test_latents = latents[latents["frame"].isin(frames["test"])].copy()
    test_labeled = test_latents.merge(full_ann[["filename", "label"]],
                                      on="filename", how="inner")
    if len(test_labeled) == 0:
        print(f"  [skip] no test labels: {run_dir.name}")
        return None

    y_test = le.transform(test_labeled["label"])
    X_test = test_labeled[Z_COLS].values
    bacc   = balanced_accuracy_score(y_test, clf.predict(X_test))

    return {
        "run":          run_dir.name,
        "npi":          meta["npi"],
        "repeat":       meta["repeat"],
        "n_train":      len(train_labeled),
        "n_test":       len(test_labeled),
        "balanced_acc": bacc,
    }


def eval_dir(run_dir_root: Path, full_ann: pd.DataFrame) -> pd.DataFrame:
    run_dirs = sorted(run_dir_root.glob("le_c0_npi*_r*"))
    print(f"Found {len(run_dirs)} runs in {run_dir_root.name}")
    records = [run_one(rd, full_ann) for rd in run_dirs]
    return pd.DataFrame([r for r in records if r])


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    s = (df.groupby("npi")["balanced_acc"]
           .agg(mean="mean", std="std").reset_index())
    s["mean_pct"] = (s["mean"] * 100).round(1)
    s["std_pct"]  = (s["std"]  * 100).round(1)
    s["npi_order"] = s["npi"].apply(
        lambda x: NPI_ORDER.index(x) if x in NPI_ORDER else 99)
    return s.sort_values("npi_order")


def main():
    full_ann = pd.read_csv(FULL_ANN_FILE)

    df_sup = eval_dir(SUPCON_DIR, full_ann)
    df_cln = eval_dir(CLEAN_DIR,  full_ann)
    # filter le_clean to cfg0 only
    df_cln = df_cln[df_cln["run"].str.startswith("le_c0_")]

    out_csv = REPO / "results" / "le_supcon_results.csv"
    df_sup.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    s_sup = summarise(df_sup)
    s_cln = summarise(df_cln)

    print("\n=== le_supcon cfg0 ===")
    print(s_sup[["npi", "mean_pct", "std_pct"]].to_string(index=False))
    print("\n=== le_clean cfg0 ===")
    print(s_cln[["npi", "mean_pct", "std_pct"]].to_string(index=False))

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    ax.set_facecolor("white")

    x_cln = np.arange(len(s_cln))
    x_sup = np.arange(len(s_sup))

    ax.errorbar(x_cln, s_cln["mean_pct"], yerr=s_cln["std_pct"],
                fmt="o-", color="#4E79A7", capsize=4, linewidth=1.8,
                markersize=7, label="le_clean (standard shuffle)")
    ax.errorbar(x_sup, s_sup["mean_pct"], yerr=s_sup["std_pct"],
                fmt="s--", color="#E15759", capsize=4, linewidth=1.8,
                markersize=7, label="le_supcon (guaranteed 2/class/batch)")

    ax.set_xticks(x_cln)
    ax.set_xticklabels(s_cln["npi"].tolist(), fontsize=9)
    ax.set_xlabel("n_per_img (K labels total ÷ n_frames_train)", fontsize=10)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=10)
    ax.set_title("cfg0 — train=[0], test=[1,2,3]\nSupCon: guaranteed labeled pairs vs standard shuffle",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(50, 105)
    ax.axhline(90, color="#AAAAAA", linestyle="--", linewidth=0.8, label="90% target")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out_png = REPO / "results" / "le_supcon_vs_clean_curve.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
