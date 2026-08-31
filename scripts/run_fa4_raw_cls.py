#!/usr/bin/env python3
"""
run_fa4_raw_cls.py
==================
FA 4-class classification using raw pixel statistics (no AE).

Modes
-----
  features : Extract per-patch pixel statistics from pax and/or actin patches.
             Saves feature CSVs to RUN_DIR/fa4_raw_eval/.  CPU-only.
  classify : Run LightGBM label-efficiency experiments across five eval scenarios.
             CPU-only.
  plot     : Load results CSVs and generate efficiency-curve PNGs.  CPU-only.
  all      : features → classify → plot in sequence.

Options
-------
  --channels pax          (default) pax channel only
  --channels actin        actin (ch3) channel only
  --channels pax_actin    pax + actin channels concatenated

Features per channel (10 per channel):
  mean, std, p10, p25, p50, p75, p90, p99, skewness, kurtosis

Usage examples
--------------
  python scripts/run_fa4_raw_cls.py --mode features --channels pax
  python scripts/run_fa4_raw_cls.py --mode classify --channels pax
  python scripts/run_fa4_raw_cls.py --mode all --channels pax
  python scripts/run_fa4_raw_cls.py --mode all --channels pax_actin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR = DATA_ROOT / "labelling"
OUT_ROOT  = DATA_ROOT / "ae_results" / "contrastive_run" / "fa4_raw_eval"

PAX_PATCH_DIRS = {
    "vinc_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio"    / "vinc" / "control" / "tiff_patches32_mr10",
    "vinc_ycomp": DATA_ROOT / "ae_results" / "patches" / "cio"    / "vinc" / "ycomp"   / "tiff_patches32_mr10",
    "pfak_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "pfak" / "control" / "tiff_patches32_mr10",
}

ACTIN_PATCH_DIRS = {
    "vinc_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "vinc_ch3" / "control" / "tiff_patches32_mr10",
    "vinc_ycomp": DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "vinc_ch3" / "ycomp"   / "tiff_patches32_mr10",
    "pfak_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "pfak_ch3" / "control" / "tiff_patches32_mr10",
}

LABEL_FILES = {
    "vinc_ctrl":  LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv",
    "vinc_ycomp": LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv",
    "pfak_ctrl":  LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv",
}

FA_LABEL_ORDER_4 = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]

LABEL_SHORT = {
    "Nascent Adhesion":   "NA",
    "focal complex":      "FC",
    "focal adhesion":     "FA",
    "fibrillar adhesion": "Fib",
}

SEED = 42

FRACS_REPEATS = [
    (0.10, 10),
    (0.25,  4),
    (0.50,  4),
    (0.75,  4),
]

SCENARIOS = [
    ("vinc_only",   ["vinc_ctrl", "vinc_ycomp"], ["vinc_ctrl", "vinc_ycomp"], False),
    ("pfak_only",   ["pfak_ctrl"],               ["pfak_ctrl"],               False),
    ("ctrl->ycomp", ["vinc_ctrl"],               ["vinc_ycomp"],              True),
    ("vinc->pfak",  ["vinc_ctrl", "vinc_ycomp"], ["pfak_ctrl"],               True),
    ("pfak->vinc",  ["pfak_ctrl"],               ["vinc_ctrl", "vinc_ycomp"], True),
    ("combined",    ["vinc_ctrl", "vinc_ycomp", "pfak_ctrl"],
                    ["vinc_ctrl", "vinc_ycomp", "pfak_ctrl"],                 False),
]

SCENARIO_COLORS = {
    "vinc_only":   "#4E79A7",
    "pfak_only":   "#F28E2B",
    "ctrl->ycomp": "#B07AA1",
    "vinc->pfak":  "#E15759",
    "pfak->vinc":  "#76B7B2",
    "combined":    "#59A14F",
}

SCENARIO_LABELS = {
    "vinc_only":   "vinc only (within)",
    "pfak_only":   "pfak only (within)",
    "ctrl->ycomp": "ctrl → ycomp (cross-cond)",
    "vinc->pfak":  "vinc → pfak (cross-ds)",
    "pfak->vinc":  "pfak → vinc (cross-ds)",
    "combined":    "combined (within)",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _patch_stats(img: np.ndarray, prefix: str) -> dict:
    """Extract 10 pixel-level statistics from a 2-D patch, prefixed by channel name."""
    flat = img.ravel().astype(np.float64)
    skew = float(sp_stats.skew(flat))
    kurt = float(sp_stats.kurtosis(flat))
    p10, p25, p50, p75, p90, p99 = np.percentile(flat, [10, 25, 50, 75, 90, 99])
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std":  float(flat.std()),
        f"{prefix}_p10":  float(p10),
        f"{prefix}_p25":  float(p25),
        f"{prefix}_p50":  float(p50),
        f"{prefix}_p75":  float(p75),
        f"{prefix}_p90":  float(p90),
        f"{prefix}_p99":  float(p99),
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
    }


def extract_features(ds_key: str, channels: str) -> pd.DataFrame:
    """Extract pixel statistics for all patches of a dataset.

    Parameters
    ----------
    ds_key   : one of "vinc_ctrl", "vinc_ycomp", "pfak_ctrl"
    channels : "pax" | "actin" | "pax_actin"
    """
    import tifffile

    pax_dir   = PAX_PATCH_DIRS[ds_key]
    actin_dir = ACTIN_PATCH_DIRS[ds_key]

    use_pax   = channels in ("pax", "pax_actin")
    use_actin = channels in ("actin", "pax_actin")

    ref_dir = pax_dir if use_pax else actin_dir
    patch_paths = sorted(ref_dir.glob("*.tif"))
    if not patch_paths:
        raise FileNotFoundError(f"No .tif files in {ref_dir}")

    print(f"[features] {ds_key} ({channels}): {len(patch_paths)} patches")

    rows = []
    for i, pp in enumerate(patch_paths):
        row: dict = {"filename": pp.name, "dataset": ds_key}

        if use_pax:
            img_pax = tifffile.imread(str(pp)).astype(np.float32)
            if img_pax.ndim != 2:
                img_pax = img_pax.squeeze()
            row.update(_patch_stats(img_pax, "pax"))

        if use_actin:
            ap = actin_dir / pp.name
            if ap.exists():
                img_act = tifffile.imread(str(ap)).astype(np.float32)
                if img_act.ndim != 2:
                    img_act = img_act.squeeze()
                row.update(_patch_stats(img_act, "actin"))
            else:
                for k in _patch_stats(np.zeros((32, 32), dtype=np.float32), "actin"):
                    row[k] = np.nan

        rows.append(row)

        if (i + 1) % 5000 == 0:
            print(f"[features]   {i + 1}/{len(patch_paths)}")

    return pd.DataFrame(rows)


def _feat_cols(channels: str) -> list[str]:
    prefixes = []
    if channels in ("pax", "pax_actin"):
        prefixes.append("pax")
    if channels in ("actin", "pax_actin"):
        prefixes.append("actin")
    suffixes = ["mean", "std", "p10", "p25", "p50", "p75", "p90", "p99", "skew", "kurt"]
    return [f"{p}_{s}" for p in prefixes for s in suffixes]


# ---------------------------------------------------------------------------
# Shared helpers (labels, classifier)
# ---------------------------------------------------------------------------

def _load_labels(ds_key: str) -> pd.DataFrame:
    path = LABEL_FILES[ds_key]
    df = pd.read_csv(path)
    if ds_key == "vinc_ycomp":
        df = df[df["filename"].str.startswith("ycomp_")].copy()
    elif ds_key == "vinc_ctrl":
        df = df[~df["filename"].str.startswith("ycomp_")].copy()
    return df[["filename", "label"]].copy()


def make_classifier():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=3,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight as _csw

        class _GBDT:
            def __init__(self):
                self._clf = GradientBoostingClassifier(
                    n_estimators=300, learning_rate=0.05,
                    max_depth=4, random_state=SEED)

            def fit(self, X, y):
                self._clf.fit(X, y, sample_weight=_csw("balanced", y))
                return self

            def predict(self, X):
                return self._clf.predict(X)

        return _GBDT()


def _metrics_one_split(y_true, y_pred, classes) -> dict:
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0)
    row = {"bal_acc": bal_acc, "macro_f1": float(np.mean(f1))}
    for cls, f in zip(classes, f1):
        row[f"f1_{LABEL_SHORT[cls]}"] = float(f)
    return row


# ===========================================================================
# MODE: features
# ===========================================================================

def run_features(channels: str):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for ds_key in ["vinc_ctrl", "vinc_ycomp", "pfak_ctrl"]:
        out_csv = OUT_ROOT / f"features_{ds_key}_{channels}.csv"
        if out_csv.exists():
            print(f"[features] {out_csv.name} already exists — skipping")
            continue
        df = extract_features(ds_key, channels)
        df.to_csv(out_csv, index=False)
        print(f"[features]   saved {len(df)} rows → {out_csv.name}")


# ===========================================================================
# MODE: classify
# ===========================================================================

def _build_dataset(channels: str, ds_keys: list[str]) -> pd.DataFrame:
    frames = []
    for ds_key in ds_keys:
        feat_csv = OUT_ROOT / f"features_{ds_key}_{channels}.csv"
        if not feat_csv.exists():
            raise FileNotFoundError(
                f"{feat_csv.name} not found — run --mode features --channels {channels} first.")
        feat = pd.read_csv(feat_csv)
        labs = _load_labels(ds_key)
        merged = feat.merge(labs, on="filename", how="inner")
        merged["dataset"] = ds_key
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(FA_LABEL_ORDER_4)].copy()
    df["label"] = pd.Categorical(df["label"], categories=FA_LABEL_ORDER_4, ordered=True)
    return df


def run_classify(channels: str):
    from sklearn.model_selection import StratifiedShuffleSplit

    fcols = _feat_cols(channels)
    suffix = channels
    print(f"[classify] channels={channels}  features({len(fcols)}): {fcols}")

    all_ds_keys = set()
    for _, train_ds, test_ds, _ in SCENARIOS:
        all_ds_keys.update(train_ds + test_ds)

    ds_data: dict[str, pd.DataFrame] = {}
    for key in sorted(all_ds_keys):
        ds_data[key] = _build_dataset(channels, [key])
        print(f"[classify]   {key}: {len(ds_data[key])} FA-labelled patches, "
              f"classes: {dict(ds_data[key]['label'].value_counts())}")

    classes = FA_LABEL_ORDER_4
    all_summaries = []

    for scenario_name, train_keys, test_keys, cross_ds in SCENARIOS:
        print(f"\n[classify] Scenario: {scenario_name}")
        train_pool = pd.concat([ds_data[k] for k in train_keys], ignore_index=True)
        test_pool  = pd.concat([ds_data[k] for k in test_keys],  ignore_index=True)

        missing = [c for c in fcols if c not in train_pool.columns]
        if missing:
            print(f"  [skip] missing columns: {missing}")
            continue

        records = []

        for frac, n_repeats in FRACS_REPEATS:
            if cross_ds:
                rng = np.random.default_rng(SEED)
                for rep in range(n_repeats):
                    n_sample = max(1, int(len(train_pool) * frac))
                    train_idx = rng.choice(len(train_pool), size=n_sample, replace=False)
                    train_df  = train_pool.iloc[train_idx]
                    test_df   = test_pool

                    X_tr = train_df[fcols].values
                    y_tr = train_df["label"].values
                    X_te = test_df[fcols].values
                    y_te = test_df["label"].values

                    clf = make_classifier()
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)

                    m = _metrics_one_split(y_te, y_pred, classes)
                    m.update({"frac": frac, "repeat": rep,
                              "n_train": len(train_df), "n_test": len(test_df)})
                    records.append(m)
            else:
                X = train_pool[fcols].values
                y = train_pool["label"].values
                sss = StratifiedShuffleSplit(
                    n_splits=n_repeats, test_size=1.0 - frac, random_state=SEED)
                for rep, (tr_idx, te_idx) in enumerate(sss.split(X, y)):
                    X_tr, y_tr = X[tr_idx], y[tr_idx]
                    X_te, y_te = X[te_idx], y[te_idx]

                    clf = make_classifier()
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)

                    m = _metrics_one_split(y_te, y_pred, classes)
                    m.update({"frac": frac, "repeat": rep,
                              "n_train": len(X_tr), "n_test": len(X_te)})
                    records.append(m)

        df_res = pd.DataFrame(records)
        csv_out = OUT_ROOT / f"results_{scenario_name}_{suffix}.csv"
        df_res.to_csv(csv_out, index=False)
        print(f"[classify]   saved {len(df_res)} rows → {csv_out.name}")

        metric_cols = ["bal_acc", "macro_f1"] + [f"f1_{LABEL_SHORT[c]}" for c in FA_LABEL_ORDER_4]
        agg = df_res.groupby("frac")[metric_cols].agg(["mean", "std"]).reset_index()
        agg.columns = ["frac"] + [f"{m}_{s}" for m in metric_cols for s in ("mean", "std")]
        agg["scenario"] = scenario_name
        all_summaries.append(agg)

    summary = pd.concat(all_summaries, ignore_index=True)
    sum_out = OUT_ROOT / f"summary_{suffix}.csv"
    summary.to_csv(sum_out, index=False)
    print(f"\n[classify] Summary saved → {sum_out}")


# ===========================================================================
# MODE: plot
# ===========================================================================

def run_plot(channels: str):
    suffix = channels

    for metric_key, metric_label in [("bal_acc", "Balanced accuracy"), ("macro_f1", "Macro F1")]:
        fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4.2 * len(SCENARIOS), 4.8),
                                 sharey=False, facecolor="white")
        for ax, (scenario_name, _, _, _) in zip(axes, SCENARIOS):
            csv_path = OUT_ROOT / f"results_{scenario_name}_{suffix}.csv"
            if not csv_path.exists():
                ax.set_title(f"{scenario_name}\n(no data)", fontsize=9)
                ax.axis("off")
                continue

            df = pd.read_csv(csv_path)
            fracs = sorted(df["frac"].unique())
            means = [df[df["frac"] == f][metric_key].mean() * 100 for f in fracs]
            stds  = [df[df["frac"] == f][metric_key].std()  * 100 for f in fracs]
            color = SCENARIO_COLORS.get(scenario_name, "#888888")
            x = np.arange(len(fracs))
            ax.errorbar(x, means, yerr=stds, fmt="o-", color=color,
                        capsize=4, linewidth=1.8, markersize=7)
            for xi, m, s in zip(x, means, stds):
                ax.text(xi, m + s + 1.0, f"{m:.1f}", ha="center", fontsize=7.5, color=color)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(f*100)}%" for f in fracs], fontsize=9)
            ax.set_xlabel("Training fraction", fontsize=9)
            ax.set_ylabel(f"{metric_label} (%)", fontsize=9)
            ax.set_title(SCENARIO_LABELS.get(scenario_name, scenario_name),
                         fontsize=9.5, fontweight="bold")
            ax.set_ylim(0, 110)
            ax.axhline(50, color="#CCCCCC", linestyle="--", linewidth=0.8)
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)

        ch_label = channels.replace("_", " + ")
        fig.suptitle(
            f"FA 4-class — Raw pixel stats  ({ch_label})\n"
            f"{metric_label}  ·  LightGBM  (10 stats/channel)",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        png_out = OUT_ROOT / f"efficiency_{metric_key}_{suffix}.png"
        fig.savefig(str(png_out), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[plot] saved → {png_out}")

    # Per-class F1
    fig, axes = plt.subplots(
        len(FA_LABEL_ORDER_4), len(SCENARIOS),
        figsize=(4.0 * len(SCENARIOS), 3.5 * len(FA_LABEL_ORDER_4)),
        sharey=False, facecolor="white",
    )
    for col, (scenario_name, _, _, _) in enumerate(SCENARIOS):
        csv_path = OUT_ROOT / f"results_{scenario_name}_{suffix}.csv"
        for row, cls in enumerate(FA_LABEL_ORDER_4):
            ax = axes[row][col]
            col_key = f"f1_{LABEL_SHORT[cls]}"
            if not csv_path.exists():
                ax.axis("off")
                continue
            df = pd.read_csv(csv_path)
            if col_key not in df.columns:
                ax.axis("off")
                continue
            fracs = sorted(df["frac"].unique())
            means = [df[df["frac"] == f][col_key].mean() * 100 for f in fracs]
            stds  = [df[df["frac"] == f][col_key].std()  * 100 for f in fracs]
            color = SCENARIO_COLORS.get(scenario_name, "#888888")
            x = np.arange(len(fracs))
            ax.errorbar(x, means, yerr=stds, fmt="o-", color=color,
                        capsize=3, linewidth=1.5, markersize=5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(f*100)}%" for f in fracs], fontsize=7)
            ax.set_ylim(0, 110)
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(SCENARIO_LABELS.get(scenario_name, scenario_name),
                             fontsize=8, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"F1 ({cls})", fontsize=8)

    ch_label = channels.replace("_", " + ")
    fig.suptitle(
        f"FA 4-class per-class F1  ({ch_label})\nRaw pixel stats + LightGBM",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    png_out = OUT_ROOT / f"efficiency_perclass_{suffix}.png"
    fig.savefig(str(png_out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] saved → {png_out}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="FA4 classification using raw pixel statistics (no AE).")
    p.add_argument("--mode", choices=["features", "classify", "plot", "all"], default="all")
    p.add_argument("--channels", choices=["pax", "actin", "pax_actin"], default="pax",
                   help="Which channel(s) to use (default: pax).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode in ("features", "all"):
        run_features(args.channels)

    if args.mode in ("classify", "all"):
        run_classify(args.channels)

    if args.mode in ("plot", "all"):
        run_plot(args.channels)

    print("\n[done]")


if __name__ == "__main__":
    main()
