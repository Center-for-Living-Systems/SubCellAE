#!/usr/bin/env python3
"""
analyze_fa4_perclass.py
=======================
Plot per-class F1 lines (NA / FC / FA / Fib) vs label fraction for each scenario.

Usage
-----
  python scripts/analyze_fa4_perclass.py                    # Option A, zrecon
  python scripts/analyze_fa4_perclass.py --suffix A_zproj
  python scripts/analyze_fa4_perclass.py --suffix A_zrecon_smote
  python scripts/analyze_fa4_perclass.py --all              # generate all available variants
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR  = DATA_ROOT / "ae_results" / "contrastive_run" / "fa4_xds_eval"

SCENARIOS = ["vinc_only", "pfak_only", "vinc->pfak", "pfak->vinc", "combined"]
SCENARIO_LABELS = {
    "vinc_only":  "Vinc only",
    "pfak_only":  "pFAK only",
    "vinc->pfak": "Vinc→pFAK",
    "pfak->vinc": "pFAK→Vinc",
    "combined":   "Combined",
}

CLASS_COLS   = ["f1_NA", "f1_FC", "f1_FA", "f1_Fib"]
CLASS_NAMES  = ["Nascent (NA)", "Focal Complex (FC)", "Focal Adhesion (FA)", "Fibrillar (Fib)"]
CLASS_COLORS = ["#4393c3", "#f4a582", "#2ca02c", "#9467bd"]
FRAC_LABELS  = ["10%", "25%", "50%", "75%"]


def _load_results(suffix: str) -> dict[str, pd.DataFrame]:
    """Load per-trial results CSVs for each scenario; return {scenario: df}."""
    frames = {}
    for sc in SCENARIOS:
        p = EVAL_DIR / f"results_{sc}_{suffix}.csv"
        if p.exists():
            frames[sc] = pd.read_csv(p)
        else:
            frames[sc] = None
    return frames


def plot_perclass(suffix: str):
    frames = _load_results(suffix)
    any_data = any(v is not None for v in frames.values())
    if not any_data:
        print(f"[warn] No results found for suffix '{suffix}' — skipping.")
        return

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    fig.suptitle(
        f"Per-class F1 vs label fraction  (suffix: {suffix})\n"
        "Stage-2 SupCon AE + LightGBM  |  error bars = ±1 SD across repeats",
        fontsize=11, fontweight="bold",
    )

    for col, sc in enumerate(SCENARIOS):
        ax = axes[col]
        df = frames[sc]

        ax.set_title(SCENARIO_LABELS[sc], fontsize=10, fontweight="bold")
        ax.set_xlabel("Label fraction", fontsize=9)
        if col == 0:
            ax.set_ylabel("F1 score", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")

        if df is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        fracs = sorted(df["frac"].unique())
        x = np.arange(len(fracs))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(f*100)}%" for f in fracs], fontsize=8)

        # Per-class lines
        for fcol, cname, color in zip(CLASS_COLS, CLASS_NAMES, CLASS_COLORS):
            if fcol not in df.columns:
                continue
            means = [df[df["frac"] == f][fcol].mean() for f in fracs]
            stds  = [df[df["frac"] == f][fcol].std()  for f in fracs]
            ax.errorbar(x, means, yerr=stds, fmt="o-", color=color,
                        capsize=3, linewidth=1.8, markersize=5,
                        label=cname)

        # Macro F1 as dashed black reference
        if "macro_f1" in df.columns:
            macro_m = [df[df["frac"] == f]["macro_f1"].mean() for f in fracs]
            macro_s = [df[df["frac"] == f]["macro_f1"].std()  for f in fracs]
            ax.errorbar(x, macro_m, yerr=macro_s, fmt="s--", color="black",
                        capsize=3, linewidth=1.2, markersize=4,
                        label="Macro F1", alpha=0.7)

    # Shared legend on right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = EVAL_DIR / f"perclass_lines_{suffix}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="A_zrecon",
                        help="Result file suffix, e.g. A_zrecon or A_zproj_smote")
    parser.add_argument("--all", action="store_true",
                        help="Generate plots for all detected suffixes")
    args = parser.parse_args()

    if args.all:
        # auto-detect by looking for summary_*.csv files
        suffixes = sorted(
            p.stem.replace("summary_", "")
            for p in EVAL_DIR.glob("summary_*.csv")
        )
        print(f"[info] Found suffixes: {suffixes}")
        for s in suffixes:
            plot_perclass(s)
    else:
        plot_perclass(args.suffix)


if __name__ == "__main__":
    main()
