#!/usr/bin/env python3
"""
run_label_efficiency.py

Label-efficiency experiment for vinc_control dataset.

Answers: "How many labels — and from how many images — does a user need?"

Design:
  - 4 main labeled frames (0–3), each with 114–168 patches
  - Image-held-out evaluation: test patches come from frames not seen during training
  - Sweep over n_per_img (labels per training image) and k_train (# training images)
  - Stratified subsampling (balanced adh / no-adh) within each training image
  - 20 random repeats per condition for variance estimation
  - Classifier: LightGBM on orig AE latents (annabel_vinc_supcon2_s2v2)

Also runs a random-patch split baseline (patches from same images in train+test)
to show optimistic vs realistic accuracy.

Outputs (in out_dir):
  label_efficiency_results.csv     — one row per (combo, n_per_img, repeat)
  label_efficiency_curve.png       — main figure
  label_efficiency_diversity.png   — image-diversity comparison (same budget)

Usage:
  python scripts/run_label_efficiency.py [--split s2v2] [--repeats 20] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR   = DATA_ROOT / "ae_results" / "contrastive_run"
LABEL_DIR = DATA_ROOT / "labelling"

# Frames with substantial labels (≥100 patches)
MAIN_FRAMES = [0, 1, 2, 3]

# n_per_img values to sweep (labels per training image, stratified by class)
N_PER_IMG_VALUES = [10, 25, 50, 75, 100, "all"]

LABEL_ORDER = ["No adhesion", "adhesion"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABEL_ORDER)}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (latents_df, labels_df) joined and filtered to MAIN_FRAMES."""
    lat_path = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / "vinc_control_latents.csv"
    lat = pd.read_csv(lat_path)
    z_cols = [c for c in lat.columns if c.startswith("z_")]

    lab = pd.read_csv(LABEL_DIR / "vinc_control_label_combined_2cls.csv")
    lab["frame"] = lab["unique_ID"].apply(lambda u: int(re.search(r"f(\d+)", u).group(1)))
    lab = lab[lab["frame"].isin(MAIN_FRAMES)].copy()

    # join on unique_ID
    merged = lab.merge(lat[["unique_ID"] + z_cols], on="unique_ID", how="inner")
    print(f"Labeled patches in main frames: {len(merged)} "
          f"({(merged['label']=='adhesion').sum()} adh, "
          f"{(merged['label']=='No adhesion').sum()} no-adh)")
    print("Per-frame counts:")
    print(merged.groupby("frame")["label"].value_counts().unstack(fill_value=0).to_string())
    return merged, z_cols


# ── Classifier ───────────────────────────────────────────────────────────────

def _train_lgbm(X_train: np.ndarray, y_train: np.ndarray) -> object:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    w = compute_sample_weight("balanced", y_train)
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=w)
    return clf


def _eval(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    pred = clf.predict(X_test)
    bal = balanced_accuracy_score(y_test, pred)
    acc = (pred == y_test).mean()
    return {"balanced_acc": bal, "acc": acc}


# ── Subsampling ───────────────────────────────────────────────────────────────

def stratified_subsample(df: pd.DataFrame, n_per_img: int | str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Return stratified subsample from df (one frame's labeled patches).
    If n_per_img='all', return all rows.
    Otherwise, sample ceil(n/2) from each class.
    """
    if n_per_img == "all":
        return df
    n = int(n_per_img)
    adh  = df[df["label"] == "adhesion"]
    noad = df[df["label"] == "No adhesion"]
    n_a = n // 2
    n_b = n - n_a
    # silently clip to available
    n_a = min(n_a, len(adh))
    n_b = min(n_b, len(noad))
    parts = []
    if n_a > 0:
        parts.append(adh.iloc[rng.choice(len(adh), n_a, replace=False)])
    if n_b > 0:
        parts.append(noad.iloc[rng.choice(len(noad), n_b, replace=False)])
    return pd.concat(parts)


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_image_held_out(merged: pd.DataFrame, z_cols: list[str],
                       repeats: int, rng_seed: int = 0) -> pd.DataFrame:
    """
    Image-held-out evaluation.
    For each combination of k_train training frames from MAIN_FRAMES,
    for each n_per_img, for each repeat:
      - Subsample training labels stratified per frame
      - Train LightGBM, evaluate on held-out frames.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []

    for k_train in [1, 2, 3]:
        k_test = len(MAIN_FRAMES) - k_train
        for train_frames in itertools.combinations(MAIN_FRAMES, k_train):
            test_frames = [f for f in MAIN_FRAMES if f not in train_frames]
            train_key = "+".join(str(f) for f in train_frames)
            test_key  = "+".join(str(f) for f in test_frames)

            df_test = merged[merged["frame"].isin(test_frames)]
            X_test  = df_test[z_cols].values
            y_test  = df_test["label"].map(LABEL_TO_INT).values

            dfs_train_full = [merged[merged["frame"] == f] for f in train_frames]

            for n_per_img in N_PER_IMG_VALUES:
                # check feasibility: need at least 2 per class after subsampling
                feasible = True
                for df_f in dfs_train_full:
                    n_available = min(
                        (df_f["label"] == "adhesion").sum(),
                        (df_f["label"] == "No adhesion").sum(),
                    )
                    if n_per_img != "all" and int(n_per_img) // 2 > n_available:
                        feasible = False
                        break
                if not feasible:
                    continue

                for rep in range(repeats):
                    parts = [stratified_subsample(df_f, n_per_img, rng)
                             for df_f in dfs_train_full]
                    df_train = pd.concat(parts)
                    X_train  = df_train[z_cols].values
                    y_train  = df_train["label"].map(LABEL_TO_INT).values

                    if len(np.unique(y_train)) < 2:
                        continue

                    clf = _train_lgbm(X_train, y_train)
                    metrics = _eval(clf, X_test, y_test)
                    rows.append({
                        "eval_type":    "image_held_out",
                        "k_train":      k_train,
                        "k_test":       k_test,
                        "train_frames": train_key,
                        "test_frames":  test_key,
                        "n_per_img":    str(n_per_img),
                        "n_train_total": len(df_train),
                        "n_test_total":  len(df_test),
                        "repeat":        rep,
                        **metrics,
                    })

    return pd.DataFrame(rows)


def run_random_split_baseline(merged: pd.DataFrame, z_cols: list[str],
                               repeats: int, rng_seed: int = 1) -> pd.DataFrame:
    """
    Random patch-split baseline (optimistic): 80/20 random train/test,
    ignoring image boundaries.  For each n_train in roughly matching budgets.
    """
    rng = np.random.default_rng(rng_seed)
    X_all = merged[z_cols].values
    y_all = merged["label"].map(LABEL_TO_INT).values
    n_all = len(merged)
    rows  = []

    # match budgets to n_per_img × 3 training images (approx)
    budgets = [30, 75, 150, 225, 300, n_all]
    for n_train in budgets:
        if n_train >= n_all:
            n_train = n_all
        n_test = n_all - n_train
        if n_test < 10:
            continue
        for rep in range(repeats):
            idx = rng.permutation(n_all)
            tr, te = idx[:n_train], idx[n_train:]
            if len(np.unique(y_all[tr])) < 2:
                continue
            clf = _train_lgbm(X_all[tr], y_all[tr])
            metrics = _eval(clf, X_all[te], y_all[te])
            rows.append({
                "eval_type":     "random_split",
                "n_train_total": n_train,
                "n_test_total":  n_test,
                "repeat":        rep,
                **metrics,
            })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_efficiency_curve(results: pd.DataFrame, out_path: Path):
    """
    Main label-efficiency figure.

    Panel A: image-held-out balanced accuracy vs total training labels,
             one curve per k_train (1, 2, 3 training images).
             Line = mean; shaded = ±1 SD across repeats and frame combinations.
    Panel B: random-split baseline vs image-held-out (k_train=3) on same x-axis.
    """
    iho = results[results["eval_type"] == "image_held_out"].copy()
    rs  = results[results["eval_type"] == "random_split"].copy()

    # map n_per_img label to numeric total for x-axis (use actual n_train_total)
    # aggregate: mean and std of balanced_acc over repeats AND frame combos
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    # Panel A
    ax = axes[0]
    colors = {1: "#E06C75", 2: "#61AFEF", 3: "#98C379"}
    for k in [1, 2, 3]:
        sub = iho[iho["k_train"] == k]
        grp = sub.groupby("n_train_total")["balanced_acc"].agg(["mean", "std"]).reset_index()
        grp = grp.sort_values("n_train_total")
        ax.plot(grp["n_train_total"], grp["mean"] * 100, marker="o", color=colors[k],
                label=f"{k} training image{'s' if k>1 else ''}")
        ax.fill_between(grp["n_train_total"],
                        (grp["mean"] - grp["std"]) * 100,
                        (grp["mean"] + grp["std"]) * 100,
                        alpha=0.15, color=colors[k])

    ax.axhline(95, color="gray", linestyle="--", linewidth=0.8, label="95% target")
    ax.set_xlabel("Total training labels", fontsize=10)
    ax.set_ylabel("Balanced accuracy on held-out image(s) (%)", fontsize=10)
    ax.set_title("Label efficiency  —  image-held-out evaluation\n"
                 "Test patches from images NOT seen during training", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(40, 102)
    ax.tick_params(labelsize=8)

    # Panel B: image-held-out (k_train=3) vs random split
    ax = axes[1]
    sub3 = iho[iho["k_train"] == 3]
    grp3 = sub3.groupby("n_train_total")["balanced_acc"].agg(["mean", "std"]).reset_index().sort_values("n_train_total")
    ax.plot(grp3["n_train_total"], grp3["mean"] * 100, marker="o", color=colors[3],
            label="Image-held-out (3 train imgs)")
    ax.fill_between(grp3["n_train_total"],
                    (grp3["mean"] - grp3["std"]) * 100,
                    (grp3["mean"] + grp3["std"]) * 100,
                    alpha=0.15, color=colors[3])

    if len(rs) > 0:
        grp_rs = rs.groupby("n_train_total")["balanced_acc"].agg(["mean", "std"]).reset_index().sort_values("n_train_total")
        ax.plot(grp_rs["n_train_total"], grp_rs["mean"] * 100, marker="s",
                color="#C678DD", linestyle="--", label="Random patch split (optimistic)")
        ax.fill_between(grp_rs["n_train_total"],
                        (grp_rs["mean"] - grp_rs["std"]) * 100,
                        (grp_rs["mean"] + grp_rs["std"]) * 100,
                        alpha=0.10, color="#C678DD")

    ax.axhline(95, color="gray", linestyle="--", linewidth=0.8, label="95% target")
    ax.set_xlabel("Total training labels", fontsize=10)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=10)
    ax.set_title("Image-held-out vs random-split baseline\n"
                 "Gap = optimism from same-image train/test leakage", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(40, 102)
    ax.tick_params(labelsize=8)

    fig.suptitle("dataset1 / vinc / control  —  label efficiency  (LightGBM on orig AE latents)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_diversity_curve(results: pd.DataFrame, out_path: Path):
    """
    Image-diversity comparison: at similar total budgets, compare
    using 1 vs 2 vs 3 training images.
    """
    iho = results[results["eval_type"] == "image_held_out"].copy()

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    colors = {1: "#E06C75", 2: "#61AFEF", 3: "#98C379"}

    for k in [1, 2, 3]:
        sub = iho[iho["k_train"] == k]
        grp = sub.groupby("n_per_img")["balanced_acc"].agg(["mean", "std", "count"]).reset_index()
        # map n_per_img string to approximate total labels
        def _to_x(row):
            if row["n_per_img"] == "all":
                return sub[sub["n_per_img"] == "all"]["n_train_total"].mean()
            return int(row["n_per_img"]) * k
        grp["x_total"] = grp.apply(_to_x, axis=1)
        grp = grp.sort_values("x_total")
        ax.plot(grp["x_total"], grp["mean"] * 100, marker="o", color=colors[k],
                label=f"{k} training image{'s' if k>1 else ''} "
                      f"(~{100//k if k<4 else 'all'} labels/img at 100 total)")
        ax.fill_between(grp["x_total"],
                        (grp["mean"] - grp["std"]) * 100,
                        (grp["mean"] + grp["std"]) * 100,
                        alpha=0.15, color=colors[k])

    ax.axhline(95, color="gray", linestyle="--", linewidth=0.8, label="95% target")
    ax.set_xlabel("Total training labels (approx)", fontsize=10)
    ax.set_ylabel("Balanced accuracy on held-out image(s) (%)", fontsize=10)
    ax.set_title("Image diversity vs label count\n"
                 "Same annotation budget: few images (dense) vs many images (sparse)",
                 fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(40, 102)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="s2v2")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: {run_dir}/label_efficiency)")
    args = ap.parse_args()

    run_dir = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "label_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, z_cols = load_data(args.split)

    print(f"\nRunning image-held-out experiment  (repeats={args.repeats})…")
    df_iho = run_image_held_out(merged, z_cols, args.repeats, rng_seed=args.seed)

    print(f"\nRunning random-split baseline  (repeats={args.repeats})…")
    df_rs = run_random_split_baseline(merged, z_cols, args.repeats, rng_seed=args.seed + 1)

    results = pd.concat([df_iho, df_rs], ignore_index=True)
    csv_out = out_dir / "label_efficiency_results.csv"
    results.to_csv(str(csv_out), index=False)
    print(f"Saved: {csv_out.name}  ({len(results)} rows)")

    # Print summary table
    print("\nImage-held-out summary (mean balanced acc %):")
    summary = (df_iho.groupby(["k_train", "n_per_img"])["balanced_acc"]
               .agg(["mean", "std"])
               .assign(mean_pct=lambda d: (d["mean"] * 100).round(1),
                       std_pct=lambda d: (d["std"] * 100).round(1))
               .drop(columns=["mean", "std"]))
    # nicer display order
    order_map = {str(v): i for i, v in enumerate(N_PER_IMG_VALUES)}
    summary = summary.reset_index()
    summary["_ord"] = summary["n_per_img"].map(order_map)
    summary = summary.sort_values(["k_train", "_ord"]).drop(columns=["_ord"])
    print(summary.to_string(index=False))

    plot_efficiency_curve(results, out_dir / "label_efficiency_curve.png")
    plot_diversity_curve(df_iho,   out_dir / "label_efficiency_diversity.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
