#!/usr/bin/env python3
"""
run_le_combined_s2v2.py
=======================
Evaluate combined image-count × label-count efficiency — s2v2 split.
  Train: frames 0+1  ·  Test: frames 2+3

Outputs
-------
  results/le_combined_s2v2_results.csv
  results/le_combined_s2v2_heatmap.png
  results/le_combined_s2v2_curves.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from subcellae.modelling.autoencoders import ContrastiveAE

DATA      = Path("/net/projects/CLS/lding/data/fa_data_analysis")
COMB_DIR  = DATA / "ae_results/contrastive_run/le_combined_s2v2"
PATCH_DIR = DATA / "ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
ANN_DIR   = DATA / "labelling/le_combined_s2v2"
FULL_ANN  = DATA / "labelling/vinc_control_label_Annabel_20260715_1554_2cls.csv"

Z_COLS        = [f"z_{i}" for i in range(12)]
N_VALUES      = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 21, 31, 41, 48]
NPI_LEVELS    = [10, 25, 50, 75, 100]
TRAIN_FRAMES  = {0, 1}
TEST_FRAMES   = {2, 3}
LATENT_DIM, PROJ_DIM, INPUT_PS, INPUT_DIVISOR = 12, 8, 32, 2.0


def _extract_frame(fn: str) -> int:
    m = re.search(r"_f(\d+)", fn)
    return int(m.group(1)) if m else -1


def _parse_run_name(name: str) -> dict | None:
    m = re.match(r"le_comb_s2_n(\d+)_npi(\d+)_r(\d+)$", name)
    if not m:
        return None
    return {"n_images": int(m.group(1)), "npi": int(m.group(2)), "series": int(m.group(3))}


def _load_model(run_dir: Path) -> ContrastiveAE:
    obj = torch.load(str(run_dir / "model_final.pt"), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        model = ContrastiveAE(latent_dim=LATENT_DIM, proj_dim=PROJ_DIM,
                              input_ps=INPUT_PS, no_ch=1, BN_flag=False, output_sigmoid=False)
        model.load_state_dict(obj)
    else:
        model = obj
    model.eval()
    return model


def _encode_frames(model: ContrastiveAE, frame_set: set[int], batch_size: int = 256) -> pd.DataFrame:
    patch_re = re.compile(r"^(.+)_f(\d{4})x\d+y\d+ps\d+\.tiff?$", re.IGNORECASE)
    patch_files = [p for p in sorted(PATCH_DIR.iterdir())
                   if (m := patch_re.match(p.name)) and int(m.group(2)) in frame_set]
    if not patch_files:
        return pd.DataFrame()
    records = []
    for i in range(0, len(patch_files), batch_size):
        batch = patch_files[i : i + batch_size]
        imgs  = [tifffile.imread(str(p)).astype(np.float32) / INPUT_DIVISOR for p in batch]
        x = torch.from_numpy(np.stack(imgs)).unsqueeze(1)
        with torch.no_grad():
            z = model.encode(x).numpy()
        for p, z_vec in zip(batch, z):
            row = {"filename": p.name}
            row.update({f"z_{j}": float(v) for j, v in enumerate(z_vec)})
            records.append(row)
    return pd.DataFrame(records)


def run_one(run_dir: Path, full_ann: pd.DataFrame) -> dict | None:
    meta = _parse_run_name(run_dir.name)
    if meta is None:
        return None
    n_images, npi, series = meta["n_images"], meta["npi"], meta["series"]

    lat_csv = run_dir / "latents.csv"
    ckpt    = run_dir / "model_final.pt"
    ann_csv = ANN_DIR / f"le_comb_s2_npi{npi}_r{series}.csv"

    if not lat_csv.exists() or not ckpt.exists():
        print(f"  [skip] not trained: {run_dir.name}")
        return None
    if not ann_csv.exists():
        print(f"  [skip] no annotation: {ann_csv.name}")
        return None

    latents = pd.read_csv(lat_csv)
    latents["frame"] = latents["filename"].apply(_extract_frame)
    ann_train = pd.read_csv(ann_csv)

    train_lat     = latents[latents["frame"].isin(TRAIN_FRAMES)]
    train_labeled = train_lat.merge(ann_train[["filename", "label"]], on="filename", how="inner")
    if len(train_labeled) == 0:
        print(f"  [skip] no training labels: {run_dir.name}")
        return None

    le      = LabelEncoder()
    y_train = le.fit_transform(train_labeled["label"])
    X_train = train_labeled[Z_COLS].values
    w_train = compute_sample_weight("balanced", y_train)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)

    model    = _load_model(run_dir)
    test_enc = _encode_frames(model, TEST_FRAMES)
    if test_enc.empty:
        return None

    test_enc["frame"] = test_enc["filename"].apply(_extract_frame)
    test_labeled = test_enc.merge(full_ann[["filename", "label"]], on="filename", how="inner")
    if len(test_labeled) == 0:
        return None

    y_test = le.transform(test_labeled["label"])
    X_test = test_labeled[Z_COLS].values
    y_pred = clf.predict(X_test)

    bacc       = balanced_accuracy_score(y_test, y_pred)
    recalls    = recall_score(   y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
    precisions = precision_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)

    print(f"  {run_dir.name}  BAcc={bacc:.3f}  adh_rec={recalls[1]:.3f}  noad_rec={recalls[0]:.3f}")
    return {"run": run_dir.name, "n_images": n_images, "npi": npi, "series": series,
            "n_train_labels": len(train_labeled), "n_test": len(test_labeled),
            "balanced_acc": float(bacc), "adh_recall": float(recalls[1]),
            "noad_recall": float(recalls[0]), "adh_precision": float(precisions[1]),
            "noad_precision": float(precisions[0])}


def _plot_heatmap(summary: pd.DataFrame, title: str, out: Path):
    piv = summary.pivot(index="npi", columns="n_images", values="balanced_acc_mean") * 100
    piv = piv.reindex(index=NPI_LEVELS, columns=N_VALUES)
    fig, ax = plt.subplots(figsize=(13, 4), facecolor="white")
    im = ax.imshow(piv.values, aspect="auto", cmap="YlGn", vmin=50, vmax=100, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="BAcc (%)")
    ax.set_xticks(range(len(N_VALUES))); ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=8)
    ax.set_yticks(range(len(NPI_LEVELS))); ax.set_yticklabels([str(n) for n in NPI_LEVELS], fontsize=9)
    ax.set_xlabel("N images in AE training", fontsize=10)
    ax.set_ylabel("Labels per frame (npi)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for ri, npi_v in enumerate(NPI_LEVELS):
        for ci, n in enumerate(N_VALUES):
            v = piv.loc[npi_v, n]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if v < 85 else "white")
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"Saved: {out}")


def _plot_curves(summary: pd.DataFrame, out: Path):
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    x = np.log2(N_VALUES)
    for npi_v, color in zip(NPI_LEVELS, colors):
        s = summary[summary["npi"] == npi_v].sort_values("n_images")
        means = s["balanced_acc_mean"].values * 100
        stds  = np.nan_to_num(s["balanced_acc_std"].values * 100, nan=0.0)
        ax.errorbar(x, means, yerr=stds, fmt="o-", color=color, capsize=3,
                    linewidth=1.8, markersize=5, label=f"npi={npi_v}")
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=8)
    ax.set_xlabel("N images in AE training (log₂ scale)", fontsize=11)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=11)
    ax.set_ylim(40, 108)
    ax.axhline(90, color="#CCCCCC", linestyle="--", linewidth=0.8, zorder=0)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_facecolor("white"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("s2v2 Combined Efficiency — BAcc vs N_images\n"
                 "SupCon AE + LGBM · train frames 0+1 · test frames 2+3 · 3 series",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"Saved: {out}")


def main():
    full_ann = pd.read_csv(FULL_ANN)
    full_ann["frame"] = full_ann["filename"].apply(_extract_frame)

    run_dirs = sorted(COMB_DIR.glob("le_comb_s2_n*_npi*_r*"))
    if not run_dirs:
        print(f"No run directories found in {COMB_DIR}"); return

    print(f"Found {len(run_dirs)} run directories")
    records = [r for rd in run_dirs if (r := run_one(rd, full_ann)) is not None]

    if not records:
        print("No completed runs."); return

    df = pd.DataFrame(records)
    out_dir = _REPO / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "le_combined_s2v2_results.csv", index=False)
    print(f"Saved: {out_dir / 'le_combined_s2v2_results.csv'}")

    agg_cols = ["balanced_acc", "adh_recall", "noad_recall", "adh_precision", "noad_precision"]
    summary = (df.groupby(["n_images", "npi"])[agg_cols]
                 .agg(["mean", "std"]).reset_index())
    summary.columns = ["n_images", "npi"] + [f"{m}_{s}" for m in agg_cols for s in ("mean", "std")]
    summary = summary.sort_values(["n_images", "npi"])

    _plot_heatmap(summary, "s2v2 BAcc (%) · rows=npi · cols=N_images · test frames 2+3",
                  out_dir / "le_combined_s2v2_heatmap.png")
    _plot_curves(summary, out_dir / "le_combined_s2v2_curves.png")
    print(f"\nDone. {len(df)} runs evaluated.")


if __name__ == "__main__":
    main()
