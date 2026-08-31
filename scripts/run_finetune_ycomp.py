#!/usr/bin/env python3
"""
run_finetune_ycomp.py

Fine-tuning efficiency test: transfer from vinc/control to vinc/ycomp.
Labels: Annabel's 685 binary (no-adh vs adh) labels on ycomp patches.

Two base models
  corrected_s3v1  SupCon on control-only (s3v1 split, fixed filename_col)
  combined_s3v1   SupCon on control+ycomp unlabeled (s3v1 split)

Two modes
  cls_only   freeze encoder → LightGBM on ycomp latents, no weight update
  full_ft    fine-tune entire model (SupCon, pretrained_checkpoint) on ycomp

Efficiency fractions: 10 / 25 / 50 / 75 % of the 80 % train pool.
Fixed test set      : 20 % of 685 labeled patches (stratified seed=42).

Output
  {RUN_DIR}/ft_ycomp_{base}_cls_only/
    results.csv   efficiency_curve.png
  {RUN_DIR}/ft_ycomp_{base}_full_ft/
    results.csv   efficiency_curve.png
    frac010/  frac025/  frac050/  frac075/   (one fine-tuned model each)

Usage
  python scripts/run_finetune_ycomp.py --base corrected_s3v1 --mode cls_only
  python scripts/run_finetune_ycomp.py --base combined_s3v1  --mode cls_only
  python scripts/run_finetune_ycomp.py --base corrected_s3v1 --mode full_ft --frac 0.10
  python scripts/run_finetune_ycomp.py --base corrected_s3v1 --mode full_ft --frac 0.25
  python scripts/run_finetune_ycomp.py --base corrected_s3v1 --mode full_ft --all-fracs
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR   = DATA_ROOT / "ae_results" / "contrastive_run"
YCOMP_DIR = DATA_ROOT / "ae_results" / "patches" / "cio/vinc/ycomp/tiff_patches32_mr10"
CONTROL_DIR = DATA_ROOT / "ae_results" / "patches" / "cio/vinc/control/tiff_patches32_mr10"
LABEL_DIR   = DATA_ROOT / "labelling"

LABEL_FILE   = LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv"
LABEL_ORDER  = ["No adhesion", "adhesion"]

BASE_DIRS = {
    "corrected_s3v1": RUN_DIR / "annabel_vinc_supcon2_corrected_s3v1",
    "combined_s3v1":  RUN_DIR / "annabel_vinc_supcon2_combined_s3v1",
}

FRACS       = [0.10, 0.25, 0.50, 0.75]
TEST_FRAC   = 0.20
SEED        = 42
Z_COLS      = [f"z_{i}" for i in range(12)]


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_binary(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def _unique_id(filename: str) -> str:
    """ycomp_f0000x...tif  →  ycomp-f0000x...tif  (matches _patch_name_to_annotation_key)."""
    return re.sub(r"_(f\d+x\d+y\d+ps\d+\.tiff?)", r"-\1", filename)


def _load_ycomp_labels() -> pd.DataFrame:
    """Return DataFrame with filename, binary_label for all ycomp patches."""
    df = pd.read_csv(LABEL_FILE)
    df = df[df["filename"].str.startswith("ycomp_")].copy()
    df["binary_label"] = df["label"].apply(_to_binary)
    df["unique_ID"]    = df["filename"].apply(_unique_id)
    return df[["filename", "unique_ID", "binary_label"]].reset_index(drop=True)


def _train_test_split_labels(df: pd.DataFrame):
    """Stratified 80/20 split on binary_label. Returns train_pool_df, test_df."""
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_FRAC,
        stratify=df["binary_label"],
        random_state=SEED,
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
    ax.plot(fracs_pct, bal_accs, "o-", label="Balanced Accuracy")
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


# ── cls_only ─────────────────────────────────────────────────────────────────

CTRL_LABEL_FILE = LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"


def _load_ctrl_latents_labeled(base: str) -> pd.DataFrame:
    """Load vinc/control labeled latents from base model (same latent space)."""
    ctrl_lab = pd.read_csv(CTRL_LABEL_FILE)
    ctrl_lab["binary_label"] = ctrl_lab["label"].apply(_to_binary)
    lat = pd.read_csv(BASE_DIRS[base] / "latents.csv")
    lat_ctrl = lat[lat["filename"].str.startswith("control_")]
    return ctrl_lab.merge(lat_ctrl[["filename"] + Z_COLS], on="filename", how="inner")


def _load_ycomp_latents_combined(base: str) -> pd.DataFrame:
    """For combined model: ycomp latents already in latents.csv."""
    lat = pd.read_csv(BASE_DIRS[base] / "latents.csv")
    return lat[lat["filename"].str.startswith("ycomp_")].reset_index(drop=True)


def _encode_ycomp_corrected() -> pd.DataFrame:
    """For corrected (control-only) model: run inference on ycomp patches."""
    from torch.utils.data import DataLoader, Dataset
    import tifffile

    class _TiffDir(Dataset):
        def __init__(self, d):
            self.paths = sorted(d.glob("*.tif")) + sorted(d.glob("*.tiff"))
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            img = tifffile.imread(str(self.paths[i])).astype(np.float32)
            return torch.tensor(img).unsqueeze(0), self.paths[i].name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = torch.load(BASE_DIRS["corrected_s3v1"] / "model_best.pt",
                        map_location=device, weights_only=False)
    model.eval()

    ds     = _TiffDir(YCOMP_DIR)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    names, zs = [], []
    with torch.no_grad():
        for x_batch, name_batch in loader:
            z = model.encode(x_batch.to(device)).cpu().numpy()
            zs.append(z)
            names.extend(name_batch)

    z_arr = np.concatenate(zs, axis=0)
    df    = pd.DataFrame(z_arr, columns=Z_COLS)
    df.insert(0, "filename", names)
    return df


def run_cls_only(base: str, fracs: list[float], out_dir: Path,
                  add_ctrl: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "ctrl+ycomp" if add_ctrl else "ycomp only"
    print(f"\n{'='*60}")
    print(f"cls_only  base={base}  training={mode_tag}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    labels = _load_ycomp_labels()
    train_pool, test_df = _train_test_split_labels(labels)

    print(f"Labels: {len(labels)} total | train_pool={len(train_pool)} test={len(test_df)}")
    print(f"Test class balance: {test_df['binary_label'].value_counts().to_dict()}")

    print("Loading latents …")
    if base == "combined_s3v1":
        lat = _load_ycomp_latents_combined(base)
    else:
        lat = _encode_ycomp_corrected()

    train_lat = train_pool.merge(lat[["filename"] + Z_COLS], on="filename", how="inner")
    test_lat  = test_df.merge(lat[["filename"] + Z_COLS],   on="filename", how="inner")
    print(f"Matched: train_pool={len(train_lat)}/{len(train_pool)}  test={len(test_lat)}/{len(test_df)}")

    z_test = test_lat[Z_COLS].values
    y_test = test_lat["binary_label"].values

    # optional: load control latents+labels from same model
    ctrl_lat = _load_ctrl_latents_labeled(base) if add_ctrl else None
    if ctrl_lat is not None:
        print(f"Control labels: {len(ctrl_lat)} patches")

    # frac=0 baseline (control only if add_ctrl, otherwise skip)
    all_fracs = ([0.0] + list(fracs)) if add_ctrl else list(fracs)

    results = []
    for frac in all_fracs:
        if frac == 0.0:
            # zero-shot: GBM trained on ctrl only (or skipped if no ctrl)
            z_tr = ctrl_lat[Z_COLS].values
            y_tr = ctrl_lat["binary_label"].values
            n    = len(ctrl_lat)
        else:
            n = max(1, int(round(frac * len(train_lat))))
            rng = np.random.RandomState(SEED)
            idx = rng.choice(len(train_lat), size=n, replace=False)
            tgt_z = train_lat.iloc[idx][Z_COLS].values
            tgt_y = train_lat.iloc[idx]["binary_label"].values
            if add_ctrl:
                z_tr = np.concatenate([ctrl_lat[Z_COLS].values, tgt_z])
                y_tr = np.concatenate([ctrl_lat["binary_label"].values, tgt_y])
            else:
                z_tr, y_tr = tgt_z, tgt_y

        metrics = _gbm_eval(z_tr, y_tr, z_test, y_test)
        n_tgt   = 0 if frac == 0.0 else max(1, int(round(frac * len(train_lat))))
        print(f"  frac={frac:.0%}  n_train={len(z_tr):3d} (tgt={n_tgt})  "
              f"bal_acc={metrics['bal_acc']:.3f}  f1={metrics['f1']:.3f}")

        cm_fig, cm_ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, metrics["y_pred"], labels=LABEL_ORDER),
            display_labels=LABEL_ORDER,
        ).plot(ax=cm_ax, colorbar=False)
        cm_ax.set_title(f"{base} cls_only ({mode_tag}) frac={frac:.0%}")
        cm_fig.savefig(out_dir / f"confusion_frac{int(frac*100):03d}.png", dpi=120)
        plt.close(cm_fig)

        results.append({"frac": frac, "n_train": len(z_tr), "n_target": n_tgt,
                        "bal_acc": metrics["bal_acc"], "f1": metrics["f1"]})

    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    curve_title = (f"cls_only  base={base}\n"
                   f"GBM trained on {mode_tag} latents")
    _save_efficiency_curve(results, out_dir / "efficiency_curve.png", curve_title)
    print(f"Done → {out_dir}")


# ── full_ft ──────────────────────────────────────────────────────────────────

def _make_frac_annotation_csv(frac_df: pd.DataFrame, out_csv: Path):
    """Write a 2cls annotation CSV with unique_ID column for AEConfig."""
    frac_df[["filename", "unique_ID", "binary_label"]].rename(
        columns={"binary_label": "label"}
    ).to_csv(out_csv, index=False)


def run_full_ft_one(base: str, frac: float, train_pool: pd.DataFrame,
                    test_df: pd.DataFrame, out_dir: Path,
                    epochs: int, lr: float, add_ctrl: bool = False):
    """Fine-tune one fraction, evaluate, return metrics dict."""
    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    frac_name = f"frac{int(frac * 100):03d}"
    ft_dir    = out_dir / frac_name
    ft_dir.mkdir(parents=True, exist_ok=True)

    n_tgt = max(1, int(round(frac * len(train_pool)))) if frac > 0 else 0
    frac_labels = pd.DataFrame()
    if n_tgt > 0:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(train_pool), size=n_tgt, replace=False)
        frac_labels = train_pool.iloc[idx].copy()

    base_ckpt = BASE_DIRS[base] / "model_best.pt"
    print(f"\n  frac={frac:.0%}  n_tgt={n_tgt}  add_ctrl={add_ctrl}  output={ft_dir.name}")

    # build patch_dirs
    patch_dirs = []
    if add_ctrl:
        ctrl_ann = ft_dir / "ctrl_labels.csv"
        ctrl_lab = pd.read_csv(CTRL_LABEL_FILE)
        ctrl_lab["binary_label"] = ctrl_lab["label"].apply(_to_binary)
        ctrl_lab[["filename", "unique_ID", "binary_label"]].rename(
            columns={"binary_label": "label"}
        ).to_csv(ctrl_ann, index=False)
        patch_dirs.append({
            "path":            str(CONTROL_DIR),
            "condition":       0,
            "condition_name":  "control",
            "annotation_file": str(ctrl_ann),
            "label_col":       "label",
            "filename_col":    "unique_ID",
            "label_order":     LABEL_ORDER,
            "val_split":       TEST_FRAC,
        })

    ycomp_entry = {
        "path":           str(YCOMP_DIR),
        "condition":      1 if add_ctrl else 0,
        "condition_name": "ycomp",
        "val_split":      TEST_FRAC,
    }
    if n_tgt > 0:
        ann_csv = ft_dir / "ycomp_labels_frac.csv"
        _make_frac_annotation_csv(frac_labels, ann_csv)
        ycomp_entry.update({
            "annotation_file": str(ann_csv),
            "label_col":       "label",
            "filename_col":    "unique_ID",
            "label_order":     LABEL_ORDER,
        })
    patch_dirs.append(ycomp_entry)

    cfg = AEConfig(
        result_dir=ft_dir,
        patch_dirs=patch_dirs,
        pretrained_checkpoint = str(base_ckpt),

        model_type      = "supcon",
        latent_dim      = 12,
        proj_dim        = 8,
        input_ps        = 32,
        no_ch           = 1,
        BN_flag         = False,
        dropout_flag    = False,
        output_sigmoid  = False,
        recon_loss_type = "nl1",

        noise_prob            = 0.0,
        temperature           = 0.5,
        lambda_recon          = 1.0,
        lambda_contrast       = 0.5,
        intensity_scale_range = (0.8, 1.2),

        epochs                  = epochs,
        lr                      = lr,
        batch_size              = 128,
        num_workers             = 0,
        val_split               = TEST_FRAC,
        group_split             = True,
        weight_decay            = 1e-4,
        warmup_epochs           = 0,
        lr_scheduler            = "none",
        early_stopping_patience = 0,
        min_epochs_for_best     = 0,

        save_recon  = False,
        device      = "auto",
    )

    run_ae_pipeline(cfg)

    # ── evaluate ─────────────────────────────────────────────────────────
    lat = pd.read_csv(ft_dir / "latents.csv")

    # GBM training set: control labeled latents (if add_ctrl) + frac ycomp labeled
    if add_ctrl:
        ctrl_lab_df = pd.read_csv(CTRL_LABEL_FILE)
        ctrl_lab_df["binary_label"] = ctrl_lab_df["label"].apply(_to_binary)
        lat_ctrl = lat[lat["filename"].str.startswith("control_")]
        ctrl_matched = ctrl_lab_df.merge(lat_ctrl[["filename"] + Z_COLS],
                                          on="filename", how="inner")
        if n_tgt > 0:
            tgt_lat = frac_labels.merge(
                lat[lat["filename"].str.startswith("ycomp_")][["filename"] + Z_COLS],
                on="filename", how="inner")
            train_lat = pd.concat([ctrl_matched, tgt_lat], ignore_index=True)
        else:
            train_lat = ctrl_matched
    else:
        train_lat = frac_labels.merge(lat[["filename"] + Z_COLS],
                                       on="filename", how="inner")

    test_lat = test_df.merge(
        lat[lat["filename"].str.startswith("ycomp_")][["filename"] + Z_COLS],
        on="filename", how="inner")

    n_train_total = len(train_lat)
    if len(test_lat) == 0 or n_train_total == 0:
        print(f"    [warn] matched: train={n_train_total} test={len(test_lat)}")
        return {"frac": frac, "n_train": n_train_total, "n_target": n_tgt,
                "bal_acc": float("nan"), "f1": float("nan")}

    metrics = _gbm_eval(
        train_lat[Z_COLS].values, train_lat["binary_label"].values,
        test_lat[Z_COLS].values,  test_lat["binary_label"].values,
    )
    print(f"    bal_acc={metrics['bal_acc']:.3f}  f1={metrics['f1']:.3f}")

    cm_fig, cm_ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix(test_lat["binary_label"].values, metrics["y_pred"],
                         labels=LABEL_ORDER),
        display_labels=LABEL_ORDER,
    ).plot(ax=cm_ax, colorbar=False)
    cm_ax.set_title(f"{base} full_ft {frac_name}")
    cm_fig.savefig(ft_dir / "confusion.png", dpi=120)
    plt.close(cm_fig)

    return {"frac": frac, "n_train": n_train_total, "n_target": n_tgt,
            "bal_acc": metrics["bal_acc"], "f1": metrics["f1"]}


def run_full_ft(base: str, fracs: list[float], out_dir: Path,
                epochs: int, lr: float, add_ctrl: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "ctrl+ycomp" if add_ctrl else "ycomp only"
    print(f"\n{'='*60}")
    print(f"full_ft  base={base}  training={mode_tag}  epochs={epochs}  lr={lr}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    labels = _load_ycomp_labels()
    train_pool, test_df = _train_test_split_labels(labels)
    print(f"Labels: {len(labels)}  train_pool={len(train_pool)}  test={len(test_df)}")

    results = []
    for frac in fracs:
        row = run_full_ft_one(base, frac, train_pool, test_df, out_dir,
                              epochs, lr, add_ctrl=add_ctrl)
        results.append(row)

    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    _save_efficiency_curve(results, out_dir / "efficiency_curve.png",
                           f"full_ft  base={base}\nAE fine-tuned on {mode_tag}")
    print(f"\nDone → {out_dir}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",   required=True, choices=list(BASE_DIRS))
    ap.add_argument("--mode",     required=True, choices=["cls_only", "full_ft"])
    ap.add_argument("--add-ctrl", action="store_true",
                    help="cls_only: always include vinc/control labels in GBM training")
    ap.add_argument("--frac",   type=float, default=None,
                    help="Single fraction (full_ft only). Use --all-fracs for all.")
    ap.add_argument("--all-fracs", action="store_true",
                    help="Run all fracs: 0.10, 0.25, 0.50, 0.75")
    ap.add_argument("--epochs", type=int,   default=100)
    ap.add_argument("--lr",     type=float, default=2e-4)
    args = ap.parse_args()

    if args.all_fracs or args.mode == "cls_only":
        fracs = FRACS
    elif args.frac is not None:
        fracs = [args.frac]
    else:
        fracs = FRACS

    if args.mode == "cls_only":
        suffix = "cls_ctrl_plus" if args.add_ctrl else "cls_only"
    else:
        suffix = "full_ft_ctrl_plus" if args.add_ctrl else "full_ft"
    out_dir = RUN_DIR / f"ft_ycomp_{args.base}_{suffix}"

    if args.mode == "cls_only":
        run_cls_only(args.base, fracs, out_dir, add_ctrl=args.add_ctrl)
    else:
        run_full_ft(args.base, fracs, out_dir, args.epochs, args.lr,
                    add_ctrl=args.add_ctrl)


if __name__ == "__main__":
    main()
