#!/usr/bin/env python3
"""
make_crossds_overlays.py

Generate prediction overlay figures for ppax, pfak, and nih3t3 datasets,
using the supcon2 s2v2 SupCon-2cls model (No adhesion vs adhesion).

ppax / pfak: predictions come from existing blind_test latents CSVs.
nih3t3:      run AE encoder on-the-fly, then apply LightGBM.

Saves PNGs to:
  {result_dir}/fa_cls_zrecon/overlay_crossds_{ds}_frame{NNNN}.png

Usage:
  python scripts/make_crossds_overlays.py [--split s2v2]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile
import joblib
import torch

REPO_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR    = DATA_ROOT / "ae_results" / "contrastive_run"
PATCH_ROOT = DATA_ROOT / "ae_results" / "patches" / "cio_rb"

PS = 32

# ── Colour palette (same as vinc overlays) ───────────────────────────────────
C_AD = np.array([0.13, 0.63, 0.24, 0.50])   # green  — adhesion
C_NA = np.array([0.55, 0.18, 0.72, 0.45])   # purple — No adhesion

LEG = [
    mpatches.Patch(color=C_AD[:3], alpha=0.9, label="adhesion (predicted)"),
    mpatches.Patch(color=C_NA[:3], alpha=0.9, label="No adhesion (predicted)"),
]

# Dataset config: patch directory and representative frame
DS_CFG = {
    "ppax": {
        "patch_dir": PATCH_ROOT / "ppax" / "control" / "tiff_patches32_mr10",
        "latents_key": "ppax_control",
        "frame": 1,           # 383 patches — densest ppax frame
        "label": "ppax / control  (phospho-paxillin)",
    },
    "pfak": {
        "patch_dir": PATCH_ROOT / "pfak" / "control" / "tiff_patches32_mr10",
        "latents_key": "pfak_control",
        "frame": 6,           # 427 patches — densest pfak frame
        "label": "pfak / control  (phospho-FAK)",
    },
    "nih3t3": {
        "patch_dir": PATCH_ROOT / "nih3t3" / "control" / "tiff_patches32_label",
        "latents_key": None,   # needs AE inference
        "frame": 8,            # 338 patches — densest nih3t3 frame
        "label": "nih3t3 / control  (vinculin, NIH3T3 cells)",
    },
}


def _parse_fname(fn: str):
    m = re.search(r"f(\d+)x(\d+)y(\d+)ps(\d+)", fn)
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _stitch_frame(filenames, xcoords, ycoords, patch_dir: Path):
    x_min, x_max = min(xcoords), max(xcoords)
    y_min, y_max = min(ycoords), max(ycoords)
    W = (x_max - x_min) + PS
    H = (y_max - y_min) + PS
    canvas = np.zeros((H, W), dtype=np.float32)

    for fn, xi, yi in zip(filenames, xcoords, ycoords):
        fp = patch_dir / Path(fn).name
        if not fp.exists():
            continue
        try:
            img = tifffile.imread(str(fp))
        except Exception:
            continue
        if img.ndim == 3:
            img = img[0]
        cx = xi - x_min
        cy = yi - y_min
        canvas[cy : cy + PS, cx : cx + PS] = img

    lo, hi = np.percentile(canvas[canvas > 0], [1, 99]) if canvas.max() > 0 else (0, 1)
    canvas = np.clip((canvas - lo) / max(hi - lo, 1e-6), 0, 1)
    return canvas, int(x_min), int(y_min)


def _pred_overlay(pred_labels, xcoords, ycoords, x_min, y_min, W, H):
    ov = np.zeros((H, W, 4), dtype=np.float32)
    for lbl, xi, yi in zip(pred_labels, xcoords, ycoords):
        cx, cy = xi - x_min, yi - y_min
        ov[cy : cy + PS, cx : cx + PS] = C_AD if lbl == "adhesion" else C_NA
    return ov


def _composite(bg, ov):
    bg_rgb = np.stack([bg, bg, bg], axis=-1)
    a = ov[..., 3:4]
    return np.clip(bg_rgb * (1 - a) + ov[..., :3] * a, 0, 1)


def get_predictions_from_latents(latents_csv: Path, lgbm_model) -> pd.DataFrame:
    """Load existing latents CSV and apply LightGBM to get pred_label."""
    df = pd.read_csv(latents_csv)
    df[["frame", "px", "py", "ps"]] = df["filename"].apply(
        lambda f: pd.Series(_parse_fname(f)))
    z_cols = [c for c in df.columns if c.startswith("z_")]
    X = df[z_cols].values.astype(np.float32)
    pred_int = lgbm_model.predict(X)
    df["pred_label"] = ["adhesion" if p == 1 else "No adhesion" for p in pred_int]
    return df


def run_nih3t3_inference(patch_dir: Path, frame: int,
                          ae_model, lgbm_model, device) -> pd.DataFrame:
    """Load nih3t3 patches for one frame, run AE encoder, apply LightGBM."""
    patches = sorted(patch_dir.glob(f"*f{frame:04d}*.tif"))
    if not patches:
        raise FileNotFoundError(f"No patches for frame {frame} in {patch_dir}")

    records = []
    imgs = []
    for p in patches:
        m = re.search(r"f(\d+)x(\d+)y(\d+)ps(\d+)", p.name)
        if not m:
            continue
        fr, px, py, ps_ = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        try:
            img = tifffile.imread(str(p))
        except Exception:
            continue
        if img.ndim == 3:
            img = img[0]
        imgs.append(img.astype(np.float32))
        records.append({"filename": p.name, "frame": fr, "px": px, "py": py, "ps": ps_})

    if not imgs:
        raise RuntimeError("No valid patches loaded")

    # Run AE encoder in batches
    batch_size = 256
    all_z = []
    ae_model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = np.stack(imgs[i : i + batch_size])[:, None]   # (B,1,H,W)
            t = torch.from_numpy(batch).to(device)
            _, z = ae_model(t)
            all_z.append(z.cpu().numpy())

    Z = np.concatenate(all_z, axis=0)
    df = pd.DataFrame(records)
    for j in range(Z.shape[1]):
        df[f"z_{j}"] = Z[:, j]

    # Apply LightGBM
    pred_int = lgbm_model.predict(Z.astype(np.float32))
    df["pred_label"] = ["adhesion" if p == 1 else "No adhesion" for p in pred_int]
    return df


def make_ds_overlay(ds: str, df_frame: pd.DataFrame, patch_dir: Path, out_path: Path):
    filenames = df_frame["filename"].tolist()
    xcoords   = df_frame["px"].tolist()
    ycoords   = df_frame["py"].tolist()
    pred_labels = df_frame["pred_label"].tolist()

    bg, x_min, y_min = _stitch_frame(filenames, xcoords, ycoords, patch_dir)
    H, W = bg.shape

    ov = _pred_overlay(pred_labels, xcoords, ycoords, x_min, y_min, W, H)
    comp = _composite(bg, ov)

    n_ad = sum(l == "adhesion"    for l in pred_labels)
    n_na = sum(l == "No adhesion" for l in pred_labels)
    cfg  = DS_CFG[ds]

    # Keep image aspect ratio: w/h
    aspect = W / H
    fig_w  = 12.0
    img_h  = fig_w / 2 / aspect   # each panel gets half fig_w
    fig_h  = max(4.0, img_h)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), facecolor="white",
                             gridspec_kw={"wspace": 0.04})
    for ax, img, title in zip(axes,
            [bg, comp],
            [f"{cfg['label']}  —  frame f{cfg['frame']:04d}",
             f"Prediction  —  adhesion={n_ad}  /  no-adh={n_na}"]):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  origin="upper", interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=11, pad=4)
        ax.axis("off")

    axes[1].legend(handles=LEG, loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}  ({W}×{H} canvas, {n_ad} adh / {n_na} no-adh)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="s2v2")
    args = ap.parse_args()

    result_dir = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    lgbm_path  = result_dir / "fa_cls_zrecon" / "model.pkl"
    out_dir    = result_dir / "fa_cls_zrecon"

    print(f"Loading LightGBM from {lgbm_path.name} ...")
    lgbm = joblib.load(str(lgbm_path))

    # Load AE model only if nih3t3 is needed
    device = torch.device("cpu")
    ae_model = None

    for ds, cfg in DS_CFG.items():
        print(f"\n── {ds} ──")
        frame = cfg["frame"]

        if cfg["latents_key"] is not None:
            # ppax / pfak: use existing blind test latents
            latents_csv = result_dir / "blind_test" / f"{cfg['latents_key']}_latents.csv"
            print(f"  Using latents: {latents_csv.name}")
            df = get_predictions_from_latents(latents_csv, lgbm)
            df_frame = df[df["frame"] == frame].copy()
        else:
            # nih3t3: run AE inference
            if ae_model is None:
                from subcellae.modelling.autoencoders import ContrastiveAE
                import yaml
                cfg_yaml = result_dir / f"ae_annabel_vinc_supcon2_{args.split}.yaml"
                with open(cfg_yaml) as f:
                    ycfg = yaml.safe_load(f)
                mc = ycfg.get("model", ycfg)
                ae_model = ContrastiveAE(
                    latent_dim=ycfg.get("latent_dim", 12),
                    proj_dim=ycfg.get("proj_dim", 8),
                    input_ps=32, no_ch=1,
                ).to(device)
                ckpt = result_dir / "model_best.pt"
                state = torch.load(str(ckpt), map_location=device, weights_only=False)
                ae_model.load_state_dict(state)
                ae_model.eval()
                print(f"  AE model loaded from {ckpt.name}")

            print(f"  Running AE inference on nih3t3 frame {frame} ...")
            df_frame = run_nih3t3_inference(
                cfg["patch_dir"], frame, ae_model, lgbm, device)

        if len(df_frame) == 0:
            print(f"  No patches for frame {frame}, skipping")
            continue
        print(f"  Frame {frame}: {len(df_frame)} patches")

        out_path = out_dir / f"overlay_crossds_{ds}_frame{frame:04d}.png"
        make_ds_overlay(ds, df_frame, cfg["patch_dir"], out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
