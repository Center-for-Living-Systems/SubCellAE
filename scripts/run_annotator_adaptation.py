#!/usr/bin/env python3
"""
run_annotator_adaptation.py

"Given a pre-trained model (SupCon AE trained on Annabel's vinc labels),
how many patches does a new annotator (Margaret) need to label to improve
the classifier?"

Design:
  - AE: annabel_vinc_supcon2_{split}  (fixed, not retrained)
  - Annabel's 539 patches: always in the LightGBM training set
  - Margaret's 377 patches: stratified 20% held out as fixed test set
  - Sweep N ∈ {0, 10, 25, 50, 75, 100, 150, 200, all} Margaret training patches
  - 20 repeats per N (different stratified random draws from training pool)
  - Classifier: GradientBoostingClassifier on AE latents (z_*)

Output:
  {run_dir}/annotator_adaptation/
    annotator_adaptation_results.csv
    annotator_adaptation_curve.png

Usage:
  python scripts/run_annotator_adaptation.py [--split s2v2] [--repeats 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR   = DATA_ROOT / "ae_results" / "contrastive_run"
LABEL_DIR = DATA_ROOT / "labelling"

LABEL_TO_INT = {"No adhesion": 0, "adhesion": 1}
N_SWEEP = [0, 10, 25, 50, 75, 100, 150, 200, "all"]


def load_latents(split: str) -> pd.DataFrame:
    path = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / "vinc_control_latents.csv"
    return pd.read_csv(path)


def load_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    annabel = pd.read_csv(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv")
    margaret = pd.read_csv(LABEL_DIR / "vinc_control_label_Margaret_2cls.csv")
    return annabel, margaret


def _train_clf(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    w = compute_sample_weight("balanced", y)
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X, y, sample_weight=w)
    return clf


def _eval(clf, X_test, y_test) -> dict:
    pred = clf.predict(X_test)
    return {
        "balanced_acc": balanced_accuracy_score(y_test, pred),
        "acc":          (pred == y_test).mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="s2v2")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="Fraction of Margaret labels held out as test set")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    run_dir = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    out_dir = run_dir / "annotator_adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load latents ─────────────────────────────────────────────────────────
    lat = load_latents(args.split)
    z_cols = [c for c in lat.columns if c.startswith("z_")]
    lat_idx = lat.set_index("unique_ID")

    # ── Load labels ──────────────────────────────────────────────────────────
    annabel, margaret = load_labels()

    # join latents
    def _join(df):
        df = df[df["unique_ID"].isin(lat_idx.index)].copy()
        df[z_cols] = lat_idx.loc[df["unique_ID"], z_cols].values
        df["y"] = df["label"].map(LABEL_TO_INT)
        return df.dropna(subset=["y"])

    ann = _join(annabel)
    mar = _join(margaret)
    print(f"Annabel: {len(ann)} patches ({(ann.y==1).sum()} adh, {(ann.y==0).sum()} no-adh)")
    print(f"Margaret: {len(mar)} patches ({(mar.y==1).sum()} adh, {(mar.y==0).sum()} no-adh)")

    X_ann = ann[z_cols].values
    y_ann = ann["y"].values

    # ── Fixed stratified test split from Margaret ────────────────────────────
    rng = np.random.default_rng(args.seed)
    mar_train_pool, mar_test = train_test_split(
        mar, test_size=args.test_frac, stratify=mar["y"], random_state=args.seed)
    X_test = mar_test[z_cols].values
    y_test = mar_test["y"].values
    print(f"\nMargaret test set  (fixed): {len(mar_test)} patches "
          f"({(mar_test.y==1).sum()} adh, {(mar_test.y==0).sum()} no-adh)")
    print(f"Margaret train pool:        {len(mar_train_pool)} patches")

    # ── N=0 baseline: Annabel-only LightGBM ─────────────────────────────────
    print(f"\nBaseline (N=0, Annabel only)…")
    clf0 = _train_clf(X_ann, y_ann)
    m0 = _eval(clf0, X_test, y_test)
    print(f"  balanced_acc = {m0['balanced_acc']*100:.1f}%  acc = {m0['acc']*100:.1f}%")

    # ── Sweep ────────────────────────────────────────────────────────────────
    rows = [{
        "n_margaret": 0, "n_train_total": len(ann),
        "repeat": 0, **m0,
    }]

    pool_adh  = mar_train_pool[mar_train_pool["y"] == 1]
    pool_noad = mar_train_pool[mar_train_pool["y"] == 0]

    for n_mar in N_SWEEP[1:]:  # skip 0, already done
        if n_mar == "all":
            n_actual = len(mar_train_pool)
        else:
            n_actual = int(n_mar)

        # stratified: split evenly, capped by available in each class
        n_each_want = n_actual // 2
        n_adh  = min(n_each_want, len(pool_adh))
        n_noad = min(n_actual - n_each_want, len(pool_noad))
        if n_adh + n_noad < 4:
            print(f"  N={n_mar}: not enough samples, skipping")
            continue

        bal_accs = []
        for rep in range(args.repeats):
            if n_mar == "all":
                sub = mar_train_pool
            else:
                adh_idx  = rng.choice(len(pool_adh),  n_adh,  replace=False)
                noad_idx = rng.choice(len(pool_noad), n_noad, replace=False)
                sub = pd.concat([pool_adh.iloc[adh_idx], pool_noad.iloc[noad_idx]])

            X_tr = np.vstack([X_ann, sub[z_cols].values])
            y_tr = np.concatenate([y_ann, sub["y"].values])

            clf = _train_clf(X_tr, y_tr)
            m = _eval(clf, X_test, y_test)
            bal_accs.append(m["balanced_acc"])
            rows.append({
                "n_margaret":    n_actual,
                "n_train_total": len(X_tr),
                "repeat":        rep,
                **m,
            })

        mean_ba = np.mean(bal_accs) * 100
        std_ba  = np.std(bal_accs)  * 100
        print(f"  N={n_mar:>4}: balanced_acc = {mean_ba:.1f}% ± {std_ba:.1f}%  "
              f"(n_train={len(X_ann)+n_actual})")

    # ── Margaret-only curve (no Annabel) ─────────────────────────────────────
    print("\nMargaret-only LightGBM (no Annabel labels)…")
    rows_mar = []
    for n_mar in N_SWEEP[1:]:
        if n_mar == "all":
            n_actual = len(mar_train_pool)
        else:
            n_actual = int(n_mar)
        n_adh  = min(n_actual // 2, len(pool_adh))
        n_noad = min(n_actual - n_actual // 2, len(pool_noad))
        if n_adh + n_noad < 4:
            continue
        bal_accs = []
        for rep in range(args.repeats):
            if n_mar == "all":
                sub = mar_train_pool
            else:
                adh_idx  = rng.choice(len(pool_adh),  n_adh,  replace=False)
                noad_idx = rng.choice(len(pool_noad), n_noad, replace=False)
                sub = pd.concat([pool_adh.iloc[adh_idx], pool_noad.iloc[noad_idx]])
            X_tr = sub[z_cols].values
            y_tr = sub["y"].values
            if len(np.unique(y_tr)) < 2:
                continue
            clf = _train_clf(X_tr, y_tr)
            m = _eval(clf, X_test, y_test)
            bal_accs.append(m["balanced_acc"])
            rows_mar.append({"n_margaret": n_actual, "repeat": rep, **m})
        mean_ba = np.mean(bal_accs) * 100
        std_ba  = np.std(bal_accs)  * 100
        print(f"  N={n_mar:>4}: balanced_acc = {mean_ba:.1f}% ± {std_ba:.1f}%")

    df_mar_only = pd.DataFrame(rows_mar)

    results = pd.DataFrame(rows)
    csv_out = out_dir / "annotator_adaptation_results.csv"
    results.to_csv(str(csv_out), index=False)
    df_mar_only.to_csv(str(out_dir / "annotator_adaptation_maronly_results.csv"), index=False)
    print(f"\nSaved: {csv_out.name}  ({len(results)} rows)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")

    grp = results.groupby("n_margaret")["balanced_acc"].agg(["mean", "std"]).reset_index().sort_values("n_margaret")
    ax.plot(grp["n_margaret"], grp["mean"] * 100,
            marker="o", color="#61AFEF", linewidth=2, zorder=3,
            label="Annabel (539) + N Margaret labels")
    ax.fill_between(grp["n_margaret"],
                    (grp["mean"] - grp["std"]) * 100,
                    (grp["mean"] + grp["std"]) * 100,
                    alpha=0.2, color="#61AFEF")

    if len(df_mar_only) > 0:
        grp2 = df_mar_only.groupby("n_margaret")["balanced_acc"].agg(["mean", "std"]).reset_index().sort_values("n_margaret")
        ax.plot(grp2["n_margaret"], grp2["mean"] * 100,
                marker="s", color="#98C379", linewidth=2, linestyle="--", zorder=3,
                label="Margaret labels only (no Annabel)")
        ax.fill_between(grp2["n_margaret"],
                        (grp2["mean"] - grp2["std"]) * 100,
                        (grp2["mean"] + grp2["std"]) * 100,
                        alpha=0.15, color="#98C379")

    base_val = grp.loc[grp["n_margaret"] == 0, "mean"].values[0] * 100
    ax.axhline(base_val, color="#E06C75", linestyle="--", linewidth=1.2,
               label=f"Annabel-only baseline  ({base_val:.1f}%)")
    ax.axhline(95, color="gray", linestyle=":", linewidth=0.9, label="95% target")

    ax.set_xlabel("Margaret labels used in LightGBM training", fontsize=11)
    ax.set_ylabel("Balanced accuracy on Margaret held-out patches (%)", fontsize=11)
    ax.set_title(
        "Annotator adaptation — dataset1 / vinc / control\n"
        "Pre-trained AE fixed (Annabel SupCon); LightGBM retrained with N Margaret labels\n"
        f"Test set: {len(mar_test)} Margaret patches held out (stratified 20%)",
        fontsize=10,
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(50, 102)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    out_png = out_dir / "annotator_adaptation_curve.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
