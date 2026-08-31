#!/usr/bin/env python3
"""
run_finetune_pfak.py

Fine-tuning efficiency test: transfer combined_s3v1 (vinc ctrl+ycomp) → pfak.
Labels: Annabel's 211 binary (no-adh vs adh) labels on pfak/control patches.

Base model: combined_s3v1  (SupCon on vinc control+ycomp unlabeled, s3v1)

Two modes
  cls_only   freeze encoder → encode pfak patches → GBM on label fraction
  full_ft    add pfak ctrl+ycomp to training, fine-tune full SupCon AE

Efficiency fractions: 10 / 25 / 50 / 75 % of the 80 % train pool.
Fixed test set      : 20 % of 211 labeled patches (stratified seed=42).

Output
  {RUN_DIR}/ft_pfak_combined_s3v1_cls_only/
    results.csv   efficiency_curve.png
  {RUN_DIR}/ft_pfak_combined_s3v1_full_ft/
    results.csv   efficiency_curve.png
    frac010/  frac025/  frac050/  frac075/

Usage
  python scripts/run_finetune_pfak.py --mode cls_only
  python scripts/run_finetune_pfak.py --mode full_ft --frac 0.10
  python scripts/run_finetune_pfak.py --mode full_ft --all-fracs
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

DATA_ROOT       = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR         = DATA_ROOT / "ae_results" / "contrastive_run"
PFAK_CTRL_DIR   = DATA_ROOT / "ae_results" / "patches" / "cio_rb/pfak/control/tiff_patches32_mr10"
PFAK_YCOMP_DIR  = DATA_ROOT / "ae_results" / "patches" / "cio_rb/pfak/ycomp/tiff_patches32_mr10"
LABEL_DIR       = DATA_ROOT / "labelling"

LABEL_FILE      = LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv"
CTRL_LABEL_FILE = LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"
BASE_DIR        = RUN_DIR / "annabel_vinc_supcon2_combined_s3v1"
BASE_NAME       = "combined_s3v1"

LABEL_ORDER     = ["No adhesion", "adhesion"]
FRACS           = [0.10, 0.25, 0.50, 0.75]
TEST_FRAC       = 0.20
SEED            = 42
Z_COLS          = [f"z_{i}" for i in range(12)]


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_binary(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def _unique_id(filename: str) -> str:
    """control_f0000x...tif  →  control-f0000x...tif"""
    return re.sub(r"_(f\d+x\d+y\d+ps\d+\.tiff?)", r"-\1", filename)


def _load_pfak_labels() -> pd.DataFrame:
    df = pd.read_csv(LABEL_FILE)
    df["binary_label"] = df["label"].apply(_to_binary)
    df["unique_ID"]    = df["filename"].apply(_unique_id)
    return df[["filename", "unique_ID", "binary_label"]].reset_index(drop=True)


def _train_test_split_labels(df: pd.DataFrame):
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_FRAC,
        stratify=df["binary_label"],
        random_state=SEED,
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _encode_pfak_ctrl() -> pd.DataFrame:
    """Encode pfak/control patches with frozen combined_s3v1 encoder."""
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
    model  = torch.load(BASE_DIR / "model_best.pt",
                        map_location=device, weights_only=False)
    model.eval()

    ds     = _TiffDir(PFAK_CTRL_DIR)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"  Encoding {len(ds)} pfak/control patches on {device} …")

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


def _load_ctrl_latents_labeled() -> pd.DataFrame:
    """Load vinc/control labeled latents from combined_s3v1 (same latent space)."""
    ctrl_lab = pd.read_csv(CTRL_LABEL_FILE)
    ctrl_lab["binary_label"] = ctrl_lab["label"].apply(_to_binary)
    lat = pd.read_csv(BASE_DIR / "latents.csv", low_memory=False)
    lat_ctrl = lat[lat["filename"].str.startswith("control_")]
    return ctrl_lab.merge(lat_ctrl[["filename"] + Z_COLS], on="filename", how="inner")


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


# ── cls_only ─────────────────────────────────────────────────────────────────

def run_cls_only(fracs: list[float], out_dir: Path, add_ctrl: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "ctrl+pfak" if add_ctrl else "pfak only"
    print(f"\n{'='*60}")
    print(f"cls_only  base={BASE_NAME}  training={mode_tag}  →  pfak")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    labels = _load_pfak_labels()
    train_pool, test_df = _train_test_split_labels(labels)
    print(f"Labels: {len(labels)}  train_pool={len(train_pool)}  test={len(test_df)}")
    print(f"Test class balance: {test_df['binary_label'].value_counts().to_dict()}")

    lat = _encode_pfak_ctrl()

    train_lat = train_pool.merge(lat[["filename"] + Z_COLS], on="filename", how="inner")
    test_lat  = test_df.merge(lat[["filename"] + Z_COLS],   on="filename", how="inner")
    print(f"Matched: train={len(train_lat)}/{len(train_pool)}  test={len(test_lat)}/{len(test_df)}")

    z_test = test_lat[Z_COLS].values
    y_test = test_lat["binary_label"].values

    ctrl_lat = _load_ctrl_latents_labeled() if add_ctrl else None
    if ctrl_lat is not None:
        print(f"Control labels: {len(ctrl_lat)} patches")

    all_fracs = ([0.0] + list(fracs)) if add_ctrl else list(fracs)

    results = []
    for frac in all_fracs:
        if frac == 0.0:
            z_tr = ctrl_lat[Z_COLS].values
            y_tr = ctrl_lat["binary_label"].values
            n_tgt = 0
        else:
            n_tgt = max(1, int(round(frac * len(train_lat))))
            rng = np.random.RandomState(SEED)
            idx = rng.choice(len(train_lat), size=n_tgt, replace=False)
            tgt_z = train_lat.iloc[idx][Z_COLS].values
            tgt_y = train_lat.iloc[idx]["binary_label"].values
            if add_ctrl:
                z_tr = np.concatenate([ctrl_lat[Z_COLS].values, tgt_z])
                y_tr = np.concatenate([ctrl_lat["binary_label"].values, tgt_y])
            else:
                z_tr, y_tr = tgt_z, tgt_y

        metrics = _gbm_eval(z_tr, y_tr, z_test, y_test)
        print(f"  frac={frac:.0%}  n_train={len(z_tr):3d} (tgt={n_tgt})  "
              f"bal_acc={metrics['bal_acc']:.3f}  f1={metrics['f1']:.3f}")

        cm_fig, cm_ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, metrics["y_pred"], labels=LABEL_ORDER),
            display_labels=LABEL_ORDER,
        ).plot(ax=cm_ax, colorbar=False)
        cm_ax.set_title(f"pfak cls_only ({mode_tag}) frac={frac:.0%}")
        cm_fig.savefig(out_dir / f"confusion_frac{int(frac*100):03d}.png", dpi=120)
        plt.close(cm_fig)

        results.append({"frac": frac, "n_train": len(z_tr), "n_target": n_tgt,
                        "bal_acc": metrics["bal_acc"], "f1": metrics["f1"]})

    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    _save_efficiency_curve(
        results, out_dir / "efficiency_curve.png",
        f"pfak cls_only — base={BASE_NAME}\nGBM trained on {mode_tag} latents",
    )
    print(f"Done → {out_dir}")


# ── full_ft ──────────────────────────────────────────────────────────────────

VINC_CTRL_DIR = DATA_ROOT / "ae_results" / "patches" / "cio/vinc/control/tiff_patches32_mr10"


def run_full_ft_one(frac: float, train_pool: pd.DataFrame, test_df: pd.DataFrame,
                    out_dir: Path, epochs: int, lr: float,
                    add_ctrl: bool = False) -> dict:
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

    print(f"\n  frac={frac:.0%}  n_tgt={n_tgt}  add_ctrl={add_ctrl}  output={ft_dir.name}")

    patch_dirs = []
    if add_ctrl:
        ctrl_ann = ft_dir / "ctrl_labels.csv"
        ctrl_lab = pd.read_csv(CTRL_LABEL_FILE)
        ctrl_lab["binary_label"] = ctrl_lab["label"].apply(_to_binary)
        ctrl_lab[["filename", "unique_ID", "binary_label"]].rename(
            columns={"binary_label": "label"}
        ).to_csv(ctrl_ann, index=False)
        patch_dirs.append({
            "path":            str(VINC_CTRL_DIR),
            "condition":       0,
            "condition_name":  "vinc_control",
            "annotation_file": str(ctrl_ann),
            "label_col":       "label",
            "filename_col":    "unique_ID",
            "label_order":     LABEL_ORDER,
            "val_split":       TEST_FRAC,
        })

    cond_offset = 1 if add_ctrl else 0
    pfak_ctrl_entry = {
        "path":           str(PFAK_CTRL_DIR),
        "condition":      cond_offset,
        "condition_name": "pfak_control",
        "val_split":      TEST_FRAC,
    }
    if n_tgt > 0:
        ann_csv = ft_dir / "pfak_labels_frac.csv"
        frac_labels[["filename", "unique_ID", "binary_label"]].rename(
            columns={"binary_label": "label"}
        ).to_csv(ann_csv, index=False)
        pfak_ctrl_entry.update({
            "annotation_file": str(ann_csv),
            "label_col":       "label",
            "filename_col":    "unique_ID",
            "label_order":     LABEL_ORDER,
        })
    patch_dirs.append(pfak_ctrl_entry)
    patch_dirs.append({
        "path":           str(PFAK_YCOMP_DIR),
        "condition":      cond_offset + 1,
        "condition_name": "pfak_ycomp",
        "val_split":      TEST_FRAC,
    })

    cfg = AEConfig(
        result_dir=ft_dir,
        patch_dirs=patch_dirs,
        pretrained_checkpoint = str(BASE_DIR / "model_best.pt"),

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

    lat = pd.read_csv(ft_dir / "latents.csv")
    lat_pfak = lat[lat["filename"].str.startswith("control_")].copy()

    if add_ctrl:
        ctrl_lab_df = pd.read_csv(CTRL_LABEL_FILE)
        ctrl_lab_df["binary_label"] = ctrl_lab_df["label"].apply(_to_binary)
        lat_vinc = lat[lat["condition_name"] == "vinc_control"] if "condition_name" in lat.columns \
                   else lat[~lat["filename"].str.startswith("control_") & ~lat["filename"].str.startswith("ycomp_")]
        # safer: match filenames to ctrl_lab_df
        ctrl_matched = ctrl_lab_df.merge(lat[["filename"] + Z_COLS], on="filename", how="inner")
        if n_tgt > 0:
            tgt_matched = frac_labels.merge(lat_pfak[["filename"] + Z_COLS],
                                             on="filename", how="inner")
            train_lat = pd.concat([ctrl_matched, tgt_matched], ignore_index=True)
        else:
            train_lat = ctrl_matched
    else:
        train_lat = frac_labels.merge(lat_pfak[["filename"] + Z_COLS],
                                       on="filename", how="inner")

    test_lat = test_df.merge(lat_pfak[["filename"] + Z_COLS], on="filename", how="inner")

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
    cm_ax.set_title(f"pfak full_ft {frac_name}")
    cm_fig.savefig(ft_dir / "confusion.png", dpi=120)
    plt.close(cm_fig)

    return {"frac": frac, "n_train": n_train_total, "n_target": n_tgt,
            "bal_acc": metrics["bal_acc"], "f1": metrics["f1"]}


def run_full_ft(fracs: list[float], out_dir: Path, epochs: int, lr: float,
                add_ctrl: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "ctrl+pfak" if add_ctrl else "pfak only"
    print(f"\n{'='*60}")
    print(f"full_ft  base={BASE_NAME}  training={mode_tag}")
    print(f"epochs={epochs}  lr={lr}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    labels = _load_pfak_labels()
    train_pool, test_df = _train_test_split_labels(labels)
    print(f"Labels: {len(labels)}  train_pool={len(train_pool)}  test={len(test_df)}")

    results = []
    for frac in fracs:
        row = run_full_ft_one(frac, train_pool, test_df, out_dir, epochs, lr,
                              add_ctrl=add_ctrl)
        results.append(row)

    pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    _save_efficiency_curve(
        results, out_dir / "efficiency_curve.png",
        f"pfak full_ft — base={BASE_NAME}\nAE fine-tuned on {mode_tag}",
    )
    print(f"\nDone → {out_dir}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",     required=True, choices=["cls_only", "full_ft"])
    ap.add_argument("--add-ctrl", action="store_true",
                    help="cls_only: include vinc/control labels in GBM training at each fraction")
    ap.add_argument("--frac", type=float, default=None,
                    help="Single fraction for full_ft. Use --all-fracs for all.")
    ap.add_argument("--all-fracs", action="store_true")
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
    out_dir = RUN_DIR / f"ft_pfak_{BASE_NAME}_{suffix}"

    if args.mode == "cls_only":
        run_cls_only(fracs, out_dir, add_ctrl=args.add_ctrl)
    else:
        run_full_ft(fracs, out_dir, args.epochs, args.lr, add_ctrl=args.add_ctrl)


if __name__ == "__main__":
    main()
