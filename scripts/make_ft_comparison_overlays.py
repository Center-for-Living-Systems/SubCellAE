#!/usr/bin/env python3
"""
make_ft_comparison_overlays.py

Generate side-by-side comparison overlays showing predictions:
  - BEFORE fine-tuning  : vinc-only encoder + vinc-only LightGBM
  - AFTER  fine-tuning  : fine-tuned encoder + crossds LightGBM
                          (or mode=lgbm_only: same encoder + crossds LightGBM)

For each dataset (ppax, pfak), produces figures for:
  - Labeled frame f0000  (patches used in fine-tuning)
  - An unlabeled frame   (generalization test)

Layout per figure: raw image | before predictions | after predictions

Usage:
  # Generate "before" half only (vinc encoder, two LightGBM models):
  python scripts/make_ft_comparison_overlays.py --split s2v2

  # After fine-tuning completes, also compute "after" using fine-tuned encoder:
  python scripts/make_ft_comparison_overlays.py --split s2v2 --include-ft
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

REPO_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR    = DATA_ROOT / "ae_results" / "contrastive_run"
PATCH_ROOT = DATA_ROOT / "ae_results" / "patches" / "cio_rb"
LABEL_DIR  = DATA_ROOT / "labelling"

PS = 32

C_AD = np.array([0.13, 0.63, 0.24, 0.55])   # green  — adhesion
C_NA = np.array([0.55, 0.18, 0.72, 0.50])   # purple — No adhesion

LEG = [
    mpatches.Patch(color=C_AD[:3], alpha=0.9, label="adhesion (predicted)"),
    mpatches.Patch(color=C_NA[:3], alpha=0.9, label="No adhesion (predicted)"),
]

# Dataset config: which frames to show
DS_CFG = {
    "ppax": {
        "patch_dir":    PATCH_ROOT / "ppax" / "control" / "tiff_patches32_mr10",
        "latents_key":  "ppax_control",
        "label_frame":  0,    # frame with labels (involved in fine-tuning)
        "extra_frame":  3,    # unlabeled frame (generalisation)
        "label":        "ppax / control  (phospho-paxillin)",
        "label_csv":    LABEL_DIR / "labels_ppax_20260521.csv",
    },
    "pfak": {
        "patch_dir":    PATCH_ROOT / "pfak" / "control" / "tiff_patches32_mr10",
        "latents_key":  "pfak_control",
        "label_frame":  0,    # frame with labels
        "extra_frame":  6,    # densest unlabeled frame
        "label":        "pfak / control  (phospho-FAK)",
        "label_csv":    LABEL_DIR / "labels_pfak_20260521.csv",
    },
}

ADHESION_CLASSES = {"focal adhesion", "Nascent Adhesion", "fibrillar adhesion", "focal complex"}


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
        cx, cy = xi - x_min, yi - y_min
        canvas[cy: cy + PS, cx: cx + PS] = img
    lo, hi = np.percentile(canvas[canvas > 0], [1, 99]) if canvas.max() > 0 else (0, 1)
    canvas = np.clip((canvas - lo) / max(hi - lo, 1e-6), 0, 1)
    return canvas, int(x_min), int(y_min)


def _pred_overlay(pred_labels, xcoords, ycoords, x_min, y_min, W, H):
    ov = np.zeros((H, W, 4), dtype=np.float32)
    for lbl, xi, yi in zip(pred_labels, xcoords, ycoords):
        cx, cy = xi - x_min, yi - y_min
        ov[cy: cy + PS, cx: cx + PS] = C_AD if lbl == "adhesion" else C_NA
    return ov


def _composite(bg, ov):
    bg_rgb = np.stack([bg, bg, bg], axis=-1)
    a = ov[..., 3:4]
    return np.clip(bg_rgb * (1 - a) + ov[..., :3] * a, 0, 1)


def _apply_lgbm(df, lgbm_model) -> pd.DataFrame:
    """Add pred_label column using a LightGBM model."""
    z_cols = [c for c in df.columns if c.startswith("z_")]
    X = df[z_cols].values.astype(np.float32)
    pred_int = lgbm_model.predict(X)
    df = df.copy()
    df["pred_label"] = ["adhesion" if p == 1 else "No adhesion" for p in pred_int]
    return df


def _run_ae_inference(ae_model, patch_dir: Path, frame: int, device) -> pd.DataFrame:
    """Run AE encoder on all patches in a frame, return DataFrame with z_* cols."""
    patches = sorted(patch_dir.glob(f"*f{frame:04d}*.tif"))
    if not patches:
        return pd.DataFrame()
    records, imgs = [], []
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
        records.append({"filename": p.name, "frame": fr, "px": px, "py": py})
    if not imgs:
        return pd.DataFrame()
    import torch  # deferred import — only needed for --include-ft mode
    batch_size = 256
    all_z = []
    ae_model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = np.stack(imgs[i: i + batch_size])[:, None]
            t = torch.from_numpy(batch).to(device)
            _, z = ae_model(t)
            all_z.append(z.cpu().numpy())
    Z = np.concatenate(all_z, axis=0)
    df = pd.DataFrame(records)
    for j in range(Z.shape[1]):
        df[f"z_{j}"] = Z[:, j]
    return df


def _load_gt_labels(ds: str, frame: int) -> dict:
    """Return {unique_ID: label_2cls} for the labeled frame."""
    cfg = DS_CFG[ds]
    if not cfg["label_csv"].exists():
        return {}
    raw = pd.read_csv(cfg["label_csv"])
    raw = raw[raw["classification"] != "Uncertain"].copy()
    raw["label_2cls"] = raw["classification"].apply(
        lambda c: "adhesion" if c in ADHESION_CLASSES else "No adhesion"
    )
    raw["frame"] = raw["unique_ID"].str.extract(r"(f\d+)")[0].str.lstrip("f").astype(int)
    sub = raw[raw["frame"] == frame]
    return dict(zip(sub["unique_ID"], sub["label_2cls"]))


def _make_comparison_figure(
    ds: str, frame: int, df_before: pd.DataFrame, df_after: pd.DataFrame | None,
    patch_dir: Path, gt_labels: dict, out_path: Path,
    before_label="Before fine-tuning", after_label="After fine-tuning (LightGBM retrained)",
    is_labeled_frame: bool = False,
):
    """Save a 2- or 3-panel comparison figure."""
    n_panels = 3 if df_after is not None else 2
    fnames   = df_before["filename"].tolist()
    xs       = df_before["px"].tolist()
    ys       = df_before["py"].tolist()

    bg, x_min, y_min = _stitch_frame(fnames, xs, ys, patch_dir)
    H, W = bg.shape

    ov_before = _pred_overlay(df_before["pred_label"].tolist(), xs, ys, x_min, y_min, W, H)
    comp_before = _composite(bg, ov_before)

    panels     = [bg, comp_before]
    titles_raw = ["Raw image", before_label]

    if df_after is not None:
        ov_after = _pred_overlay(df_after["pred_label"].tolist(), xs, ys, x_min, y_min, W, H)
        comp_after = _composite(bg, ov_after)
        panels.append(comp_after)
        titles_raw.append(after_label)

    cfg = DS_CFG[ds]
    n_ad_b = sum(l == "adhesion"    for l in df_before["pred_label"])
    n_na_b = sum(l == "No adhesion" for l in df_before["pred_label"])
    frame_tag = "labeled frame (in fine-tuning)" if is_labeled_frame else "unlabeled frame"

    # Full titles with counts
    titles = [
        f"{cfg['label']}  |  frame f{frame:04d}  [{frame_tag}]",
        f"{before_label}\n  adh={n_ad_b}  no-adh={n_na_b}",
        titles_raw[2] if n_panels == 3 else "",
    ]
    if df_after is not None:
        n_ad_a = sum(l == "adhesion"    for l in df_after["pred_label"])
        n_na_a = sum(l == "No adhesion" for l in df_after["pred_label"])
        titles[2] = f"{after_label}\n  adh={n_ad_a}  no-adh={n_na_a}"

    aspect = W / H
    fig_w  = 6.0 * n_panels
    img_h  = fig_w / n_panels / aspect
    fig_h  = max(4.5, img_h + 1.0)

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), facecolor="white",
                             gridspec_kw={"wspace": 0.04})
    if n_panels == 1:
        axes = [axes]

    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  origin="upper", interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")

    # Draw gt label borders on before/after panels (if labeled frame)
    if is_labeled_frame and gt_labels:
        df_map = {row["filename"]: row for _, row in df_before.iterrows()}
        # convert unique_ID (hyphen) → filename (underscore) by reversing first hyphen
        for uid, lbl in gt_labels.items():
            fn = uid.replace("-", "_", 1)   # first hyphen back to underscore
            if fn not in df_map:
                continue
            row = df_map[fn]
            xi, yi = row["px"] - x_min, row["py"] - y_min
            clr = C_AD[:3] if lbl == "adhesion" else C_NA[:3]
            for ax in axes[1:]:   # draw on prediction panels only
                rect = mpatches.Rectangle(
                    (xi, yi), PS, PS,
                    linewidth=1.0, edgecolor=clr, facecolor="none",
                )
                ax.add_patch(rect)

    axes[-1].legend(handles=LEG, loc="lower right", fontsize=8, framealpha=0.85)
    fig.suptitle(
        f"{'✓ Labeled patches outlined with colored borders  |  ' if is_labeled_frame and gt_labels else ''}"
        f"green=adhesion  purple=no-adhesion",
        fontsize=8, y=0.01, color="gray",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}  ({W}×{H}, n={len(fnames)})")


LABEL_DIR  = DATA_ROOT / "labelling"
PATCH_BASE = DATA_ROOT / "ae_results" / "patches"
ADHESION_CLASSES = {"focal adhesion", "Nascent Adhesion", "fibrillar adhesion", "focal complex"}
LABEL_ORDER = ["No adhesion", "adhesion"]

_LABELED_PATCH_DIRS = [
    {
        "patch_dir": PATCH_BASE / "cio/vinc/control/tiff_patches32_mr10",
        "label_csv": LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv",
        "label_col": "label", "uid_col": "unique_ID",
    },
    {
        "patch_dir": PATCH_BASE / "cio_rb/ppax/control/tiff_patches32_mr10",
        "label_csv": LABEL_DIR / "ppax_control_label_2cls.csv",
        "label_col": "label", "uid_col": "unique_ID",
    },
    {
        "patch_dir": PATCH_BASE / "cio_rb/pfak/control/tiff_patches32_mr10",
        "label_csv": LABEL_DIR / "pfak_control_label_2cls.csv",
        "label_col": "label", "uid_col": "unique_ID",
    },
]


def _uid_to_filename(uid: str) -> str:
    """Convert unique_ID (hyphen) back to patch filename (underscore)."""
    return uid.replace("-", "_", 1)


def _train_lgbm_on_ft_latents(ft_ae, device) -> object:
    """Run the fine-tuned AE encoder over all labeled patches and train a LightGBM."""
    import torch
    from lightgbm import LGBMClassifier
    import tifffile as tiff_local

    label_map = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    all_z, all_y = [], []

    for entry in _LABELED_PATCH_DIRS:
        label_df = pd.read_csv(entry["label_csv"])
        label_dict = dict(zip(label_df[entry["uid_col"]], label_df[entry["label_col"]]))
        patch_dir  = entry["patch_dir"]

        imgs, ys = [], []
        for uid, lbl in label_dict.items():
            if lbl not in label_map:
                continue
            fn = patch_dir / _uid_to_filename(uid)
            if not fn.exists():
                continue
            try:
                img = tiff_local.imread(str(fn)).astype(np.float32)
            except Exception:
                continue
            if img.ndim == 3:
                img = img[0]
            imgs.append(img)
            ys.append(label_map[lbl])

        if not imgs:
            continue

        batch_size = 256
        all_z_ds = []
        ft_ae.eval()
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = np.stack(imgs[i: i + batch_size])[:, None]
                t = torch.from_numpy(batch).to(device)
                _, z = ft_ae(t)
                all_z_ds.append(z.cpu().numpy())

        Z = np.concatenate(all_z_ds, axis=0)
        all_z.append(Z)
        all_y.extend(ys)
        print(f"    {entry['patch_dir'].parent.name}: {len(ys)} labeled patches")

    X = np.concatenate(all_z, axis=0).astype(np.float32)
    y = np.array(all_y)
    print(f"  LightGBM training on {len(X)} labeled patches  "
          f"(0={sum(y==0)} no-adh, 1={sum(y==1)} adh)")

    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=3, class_weight="balanced",
        random_state=42, verbose=-1, n_jobs=1,
    )
    clf.fit(X, y, feature_name=[f"z_{i}" for i in range(X.shape[1])])
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="s2v2")
    ap.add_argument("--include-ft", action="store_true",
                    help="Include fine-tuned encoder predictions (requires FT model to be ready).")
    args = ap.parse_args()

    orig_dir   = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    ft_dir     = RUN_DIR / f"annabel_vinc_ppax_pfak_supcon2_ft_labeled_{args.split}"
    out_dir    = orig_dir / "fa_cls_zrecon" / "ft_comparison"

    lgbm_orig     = joblib.load(str(orig_dir / "fa_cls_zrecon" / "model.pkl"))
    lgbm_crossds  = joblib.load(str(orig_dir / "fa_cls_zrecon_crossds_labels" / "model.pkl"))
    print(f"LightGBM models loaded (orig + crossds).")

    # Optionally load fine-tuned AE model and retrain LightGBM on its latents
    ft_ae    = None
    lgbm_ft  = None
    if args.include_ft:
        import torch
        ft_ckpt = ft_dir / "model_best.pt"
        if not ft_ckpt.exists():
            print(f"[WARN] Fine-tuned checkpoint not found: {ft_ckpt}")
            args.include_ft = False
        else:
            device = torch.device("cpu")
            saved = torch.load(str(ft_ckpt), map_location=device, weights_only=False)
            if hasattr(saved, "forward"):
                # checkpoint is the full model object
                ft_ae = saved.to(device)
            else:
                from subcellae.modelling.autoencoders import ContrastiveAE
                ft_ae = ContrastiveAE(latent_dim=12, proj_dim=8, input_ps=32, no_ch=1).to(device)
                ft_ae.load_state_dict(saved)
            ft_ae.eval()
            print(f"Fine-tuned AE loaded from {ft_ckpt.name}")

            print("  Training LightGBM on fine-tuned latents...")
            lgbm_ft = _train_lgbm_on_ft_latents(ft_ae, device)
            lgbm_ft_path = ft_dir / "lgbm_ft.pkl"
            joblib.dump(lgbm_ft, str(lgbm_ft_path))
            print(f"  LightGBM saved → {lgbm_ft_path.name}")

    for ds, cfg in DS_CFG.items():
        print(f"\n── {ds} ──")
        # Load full blind_test latents (computed by orig vinc encoder)
        lat_csv = orig_dir / "blind_test" / f"{cfg['latents_key']}_latents.csv"
        if not lat_csv.exists():
            print(f"  [SKIP] latents not found: {lat_csv}")
            continue
        lat_all = pd.read_csv(lat_csv)
        lat_all[["frame", "px", "py", "ps"]] = lat_all["filename"].apply(
            lambda f: pd.Series(_parse_fname(f)))

        for frame_type, frame in [("label", cfg["label_frame"]), ("unlabel", cfg["extra_frame"])]:
            is_labeled = (frame_type == "label")
            print(f"  frame f{frame:04d}  ({'labeled' if is_labeled else 'unlabeled'})")

            df_frame = lat_all[lat_all["frame"] == frame].copy()
            if df_frame.empty:
                print(f"    [SKIP] no patches")
                continue

            # Apply original LightGBM (before)
            df_before = _apply_lgbm(df_frame, lgbm_orig)
            # Apply crossds LightGBM (after-lgbm only)
            df_after_lgbm = _apply_lgbm(df_frame, lgbm_crossds)

            gt = _load_gt_labels(ds, frame) if is_labeled else {}

            # Figure 1: before vs after-lgbm (same encoder, different classifier)
            out_name = f"ft_cmp_{ds}_f{frame:04d}_lgbm.png"
            _make_comparison_figure(
                ds=ds, frame=frame,
                df_before=df_before, df_after=df_after_lgbm,
                patch_dir=cfg["patch_dir"],
                gt_labels=gt, out_path=out_dir / out_name,
                before_label="Before  (vinc-only LightGBM)",
                after_label="LightGBM retrained on vinc+ppax+pfak",
                is_labeled_frame=is_labeled,
            )

            # Figure 2 (optional): before vs after full fine-tuning
            if args.include_ft and ft_ae is not None and lgbm_ft is not None:
                import torch
                _device = torch.device("cpu")
                df_ft_lat = _run_ae_inference(ft_ae, cfg["patch_dir"], frame, _device)
                if not df_ft_lat.empty:
                    df_after_ft = _apply_lgbm(df_ft_lat, lgbm_ft)
                    out_name_ft = f"ft_cmp_{ds}_f{frame:04d}_encoder_ft.png"
                    _make_comparison_figure(
                        ds=ds, frame=frame,
                        df_before=df_before, df_after=df_after_ft,
                        patch_dir=cfg["patch_dir"],
                        gt_labels=gt, out_path=out_dir / out_name_ft,
                        before_label="Before  (vinc-only encoder + LightGBM)",
                        after_label="Fine-tuned AE + retrained LightGBM",
                        is_labeled_frame=is_labeled,
                    )

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
