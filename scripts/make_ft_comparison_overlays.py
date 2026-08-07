#!/usr/bin/env python3
"""
make_ft_comparison_overlays.py

4-panel comparison figures for all frames in ppax and pfak:
  Raw image | Before (vinc-only) | After cls FT (crossds LightGBM) | After full AE FT

For the labeled frame (f0000):
  Labeled patches are colored TP/TN/FP/FN; unlabeled patches use plain green/purple.

For unlabeled frames:
  All patches use plain green/purple predictions.

Outputs one PNG per frame per dataset → ft_cmp_{ds}_f{frame:04d}.png
(3-panel when --include-ft not set, 4-panel when it is)

Usage:
  python scripts/make_ft_comparison_overlays.py --split s2v2
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

# Prediction colours (plain, for unlabeled patches)
C_AD = np.array([0.13, 0.63, 0.24, 0.55])   # green  — adhesion
C_NA = np.array([0.55, 0.18, 0.72, 0.50])   # purple — No adhesion

# TP/TN/FP/FN colours for labeled patches (matching make_prediction_overlays.py)
C_TP = np.array([0.13, 0.63, 0.24, 0.75])   # green  — correct adhesion
C_TN = np.array([0.55, 0.18, 0.72, 0.75])   # purple — correct no-adhesion
C_FP = np.array([0.85, 0.12, 0.12, 0.75])   # red    — false positive (pred=adh, gt=no-adh)
C_FN = np.array([0.95, 0.55, 0.00, 0.75])   # orange — false negative (pred=no-adh, gt=adh)

LEG_PRED = [
    mpatches.Patch(color=C_AD[:3], alpha=0.9, label="adhesion (predicted)"),
    mpatches.Patch(color=C_NA[:3], alpha=0.9, label="No adhesion (predicted)"),
]
LEG_GT = [
    mpatches.Patch(color=C_TP[:3], label="TP  (adh / adh)"),
    mpatches.Patch(color=C_TN[:3], label="TN  (no-adh / no-adh)"),
    mpatches.Patch(color=C_FP[:3], label="FP  (pred=adh, gt=no-adh)"),
    mpatches.Patch(color=C_FN[:3], label="FN  (pred=no-adh, gt=adh)"),
]

DS_CFG = {
    "vinc_control": {
        "patch_dir":        DATA_ROOT / "ae_results" / "patches" / "cio/vinc/control/tiff_patches32_mr10",
        "latents_key":      "vinc_control",
        "label_frame":      None,   # labels spread across all frames
        "label":            "dataset1 / control  (paxillin)",
        "label_csv":        LABEL_DIR / "vinc_control_label_Margaret_2cls.csv",
        "ft_run":           "annabel_vinc_margaret_ft_labeled",
        "lgbm_ft_dirs":     "_VINC_LABELED_PATCH_DIRS",   # vinc-only: no ppax/pfak
    },
    "ppax": {
        "patch_dir":        PATCH_ROOT / "ppax" / "control" / "tiff_patches32_mr10",
        "latents_key":      "ppax_control",
        "label_frame":      0,
        "label":            "dataset3 / control  (phospho-paxillin)",
        "label_csv":        LABEL_DIR / "labels_ppax_20260521.csv",
        "ft_run":           "annabel_vinc_ppax_pfak_supcon2_ft_labeled",
        "lgbm_ft_dirs":     "_LABELED_PATCH_DIRS",         # all datasets
    },
    "pfak": {
        "patch_dir":        PATCH_ROOT / "pfak" / "control" / "tiff_patches32_mr10",
        "latents_key":      "pfak_control",
        "label_frame":      0,
        "label":            "dataset2 / control  (phospho-FAK)",
        "label_csv":        LABEL_DIR / "labels_pfak_20260521.csv",
        "ft_run":           "annabel_vinc_ppax_pfak_supcon2_ft_labeled",
        "lgbm_ft_dirs":     "_LABELED_PATCH_DIRS",         # all datasets
    },
}

ADHESION_CLASSES = {"focal adhesion", "Nascent Adhesion", "fibrillar adhesion", "focal complex"}

LABEL_DIR  = DATA_ROOT / "labelling"
PATCH_BASE = DATA_ROOT / "ae_results" / "patches"
LABEL_ORDER = ["No adhesion", "adhesion"]

_VINC_LABELED_PATCH_DIRS = [
    {
        "patch_dir": PATCH_BASE / "cio/vinc/control/tiff_patches32_mr10",
        "label_csv": LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv",
        "label_col": "label", "uid_col": "unique_ID",
    },
    {
        "patch_dir": PATCH_BASE / "cio/vinc/control/tiff_patches32_mr10",
        "label_csv": LABEL_DIR / "vinc_control_label_Margaret_2cls.csv",
        "label_col": "label", "uid_col": "unique_ID",
    },
]

_LABELED_PATCH_DIRS = _VINC_LABELED_PATCH_DIRS + [
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



def _composite(bg, ov):
    bg_rgb = np.stack([bg, bg, bg], axis=-1)
    a = ov[..., 3:4]
    return np.clip(bg_rgb * (1 - a) + ov[..., :3] * a, 0, 1)


def _apply_lgbm(df, lgbm_model) -> pd.DataFrame:
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
    import torch
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


def _load_gt_map(ds: str) -> dict:
    """Return {patch_filename: label_2cls} for all labeled patches in this dataset."""
    cfg = DS_CFG[ds]
    if not cfg["label_csv"].exists():
        return {}
    raw = pd.read_csv(cfg["label_csv"])
    if "label" in raw.columns:
        # Already 2-class format (unique_ID, label)
        raw = raw[raw["label"].isin(["adhesion", "No adhesion"])].copy()
        raw["label_2cls"] = raw["label"]
    else:
        # 5-class format with classification column
        raw = raw[raw["classification"] != "Uncertain"].copy()
        raw["label_2cls"] = raw["classification"].apply(
            lambda c: "adhesion" if c in ADHESION_CLASSES else "No adhesion"
        )
    raw["filename"] = raw["unique_ID"].apply(lambda u: u.replace("-", "_", 1))
    return dict(zip(raw["filename"], raw["label_2cls"]))


def _count_tp_tn_fp_fn(df: pd.DataFrame, gt_map: dict):
    tp = tn = fp = fn = 0
    for _, row in df.iterrows():
        fn_ = Path(row["filename"]).name
        if fn_ not in gt_map:
            continue
        gt  = gt_map[fn_]
        pred = row["pred_label"]
        if   gt == "adhesion"    and pred == "adhesion":    tp += 1
        elif gt == "No adhesion" and pred == "No adhesion": tn += 1
        elif gt == "No adhesion" and pred == "adhesion":    fp += 1
        else:                                               fn += 1
    return tp, tn, fp, fn


def _make_4panel_figure(
    ds: str, frame: int,
    df_before: pd.DataFrame,
    df_cls_ft: pd.DataFrame,
    df_enc_ft: pd.DataFrame | None,
    patch_dir: Path,
    gt_map: dict,
    out_path: Path,
):
    # A frame is "labeled" if any of its patches have ground-truth entries
    frame_fns  = {Path(fn).name for fn in df_before["filename"]}
    is_labeled = bool(gt_map) and bool(frame_fns & set(gt_map.keys()))
    effective_gt = gt_map if is_labeled else {}

    fnames = df_before["filename"].tolist()
    xs     = df_before["px"].tolist()
    ys     = df_before["py"].tolist()

    bg, x_min, y_min = _stitch_frame(fnames, xs, ys, patch_dir)
    H, W = bg.shape

    def _overlay(df):
        ov = np.zeros((H, W, 4), dtype=np.float32)
        for _, row in df.iterrows():
            cx, cy = int(row["px"]) - x_min, int(row["py"]) - y_min
            lbl = row["pred_label"]
            fn  = Path(row["filename"]).name
            if effective_gt and fn in effective_gt:
                gt = effective_gt[fn]
                if   gt == "adhesion"    and lbl == "adhesion":    color = C_TP
                elif gt == "No adhesion" and lbl == "No adhesion": color = C_TN
                elif gt == "No adhesion" and lbl == "adhesion":    color = C_FP
                else:                                               color = C_FN
            else:
                color = C_AD if lbl == "adhesion" else C_NA
            ov[cy: cy + PS, cx: cx + PS] = color  # noqa: E501
        return ov

    panels = [bg,
              _composite(bg, _overlay(df_before)),
              _composite(bg, _overlay(df_cls_ft))]
    if df_enc_ft is not None:
        panels.append(_composite(bg, _overlay(df_enc_ft)))

    cfg = DS_CFG[ds]
    frame_tag = f"labeled frame  (f{frame:04d})" if is_labeled \
                else f"unlabeled frame  (f{frame:04d})"

    def _title(df, label):
        n_ad = (df["pred_label"] == "adhesion").sum()
        n_na = (df["pred_label"] == "No adhesion").sum()
        if is_labeled:
            tp, tn, fp, fn_ = _count_tp_tn_fp_fn(df, gt_map)
            acc = (tp + tn) / max(tp + tn + fp + fn_, 1)
            return f"{label}\nadh={n_ad} no-adh={n_na}  |  acc={acc:.2f}  TP={tp} TN={tn} FP={fp} FN={fn_}"
        return f"{label}\nadh={n_ad}  no-adh={n_na}"

    if ds == "vinc_control":
        cls_ft_label = "Cls FT  (dataset1+Margaret LightGBM, orig encoder)"
        enc_ft_label = "Full AE FT  (AE fine-tuned on dataset1+Margaret)"
    else:
        cls_ft_label = "Cls FT  (crossds LightGBM, orig encoder)"
        enc_ft_label = "Full AE FT  (AE fine-tuned on dataset3+dataset2)"

    titles = [
        f"{cfg['label']}\n{frame_tag}",
        _title(df_before, "Before  (dataset1-only encoder + LightGBM)"),
        _title(df_cls_ft, cls_ft_label),
    ]
    if df_enc_ft is not None:
        titles.append(_title(df_enc_ft, enc_ft_label))

    n_panels = len(panels)
    aspect   = W / H
    fig_w    = 5.5 * n_panels
    img_h    = fig_w / n_panels / aspect
    fig_h    = max(5.0, img_h + 1.4)

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), facecolor="white",
                             gridspec_kw={"wspace": 0.04})

    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  origin="upper", interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")

    legend = LEG_GT if is_labeled else LEG_PRED
    axes[-1].legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.85)

    note = ("TP/TN/FP/FN on labeled patches  •  green/purple on unlabeled patches"
            if is_labeled else "green = adhesion  •  purple = No adhesion")
    fig.suptitle(note, fontsize=8, y=0.01, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}  ({W}×{H}, n={len(fnames)})")


def _uid_to_filename(uid: str) -> str:
    return uid.replace("-", "_", 1)


def _train_lgbm_on_ft_latents(ft_ae, device, patch_dirs=None) -> object:
    """Run the fine-tuned AE encoder over labeled patches and train a LightGBM.
    patch_dirs: list of entries like _LABELED_PATCH_DIRS; defaults to all datasets."""
    import torch
    from lightgbm import LGBMClassifier
    import tifffile as tiff_local

    if patch_dirs is None:
        patch_dirs = _LABELED_PATCH_DIRS

    label_map = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    all_z, all_y = [], []

    for entry in patch_dirs:
        label_df   = pd.read_csv(entry["label_csv"])
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

        all_z_ds = []
        ft_ae.eval()
        with torch.no_grad():
            for i in range(0, len(imgs), 256):
                batch = np.stack(imgs[i: i + 256])[:, None]
                t = torch.from_numpy(batch).to(device)
                _, z = ft_ae(t)
                all_z_ds.append(z.cpu().numpy())

        Z = np.concatenate(all_z_ds, axis=0)
        all_z.append(Z)
        all_y.extend(ys)
        print(f"    {entry['patch_dir'].parent.name}: {len(ys)} labeled patches")

    X = np.concatenate(all_z, axis=0).astype(np.float32)
    y = np.array(all_y)
    print(f"  LightGBM training on {len(X)} patches  "
          f"(0={sum(y==0)} no-adh, 1={sum(y==1)} adh)")

    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=3, class_weight="balanced",
        random_state=42, verbose=-1, n_jobs=1,
    )
    clf.fit(X, y, feature_name=[f"z_{i}" for i in range(X.shape[1])])
    return clf


def _train_lgbm_on_margaret_latents(lat_csv: Path, label_csv: Path,
                                     annabel_lat_csv: Path, annabel_label_csv: Path) -> object:
    """Train LightGBM using already-computed latents — no AE inference needed.
    Combines Annabel vinc (from latent CSV) + Margaret vinc (from blind-test latent CSV)."""
    from lightgbm import LGBMClassifier
    label_map = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}

    def _load_lat_labels(lcsv: Path, lblcsv: Path) -> tuple:
        lat  = pd.read_csv(lcsv)
        lbl  = pd.read_csv(lblcsv)
        # unique_ID → filename conversion
        lbl["filename"] = lbl["unique_ID"].apply(_uid_to_filename)
        merged = lat.merge(lbl[["filename", "label"]], on="filename", how="inner")
        merged = merged[merged["label"].isin(label_map)]
        z_cols = [c for c in merged.columns if c.startswith("z_")]
        X = merged[z_cols].values.astype(np.float32)
        y = merged["label"].map(label_map).values
        return X, y

    # Annabel's latents (training split latents already in blind_test dir)
    X_a, y_a = _load_lat_labels(annabel_lat_csv, annabel_label_csv)
    # Margaret's latents (from blind-test latents CSV)
    X_m, y_m = _load_lat_labels(lat_csv, label_csv)
    print(f"    Annabel vinc: {len(X_a)} patches")
    print(f"    Margaret vinc: {len(X_m)} patches")

    X = np.concatenate([X_a, X_m], axis=0)
    y = np.concatenate([y_a, y_m], axis=0)
    print(f"  LightGBM training on {len(X)} patches  "
          f"(0={sum(y==0)} no-adh, 1={sum(y==1)} adh)")

    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=3, class_weight="balanced",
        random_state=42, verbose=-1, n_jobs=1,
    )
    clf.fit(X, y, feature_name=[f"z_{i}" for i in range(X.shape[1])])
    return clf


def _load_ft_ae_and_lgbm(ft_run_dir: Path, ft_ae_cache: dict, patch_dirs=None):
    """Load (and cache) the fine-tuned AE + LightGBM for a given ft_run_dir.
    patch_dirs: which labeled patches to run through the FT AE when training LightGBM.
    Returns (ft_ae, lgbm_ft) or (None, None) if checkpoint missing."""
    key = str(ft_run_dir)
    if key in ft_ae_cache:
        return ft_ae_cache[key]

    import torch
    ft_ckpt   = ft_run_dir / "model_best.pt"
    lgbm_path = ft_run_dir / "lgbm_ft.pkl"

    if not ft_ckpt.exists():
        print(f"  [WARN] FT checkpoint not found: {ft_ckpt} — skipping AE FT panel")
        ft_ae_cache[key] = (None, None)
        return None, None

    device = torch.device("cpu")
    saved  = torch.load(str(ft_ckpt), map_location=device, weights_only=False)
    ft_ae  = (saved.to(device) if hasattr(saved, "forward") else None)
    if ft_ae is None:
        from subcellae.modelling.autoencoders import ContrastiveAE
        ft_ae = ContrastiveAE(latent_dim=12, proj_dim=8, input_ps=32, no_ch=1).to(device)
        ft_ae.load_state_dict(saved)
    ft_ae.eval()
    print(f"  FT AE loaded from {ft_run_dir.name}/model_best.pt")

    if lgbm_path.exists():
        lgbm_ft = joblib.load(str(lgbm_path))
        print(f"  LightGBM FT loaded from cache.")
    else:
        print("  Training LightGBM on fine-tuned latents...")
        lgbm_ft = _train_lgbm_on_ft_latents(ft_ae, device, patch_dirs=patch_dirs)
        joblib.dump(lgbm_ft, str(lgbm_path))
        print(f"  LightGBM saved → {lgbm_path.name}")

    ft_ae_cache[key] = (ft_ae, lgbm_ft)
    return ft_ae, lgbm_ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",      default="s2v2")
    ap.add_argument("--include-ft", action="store_true",
                    help="Include fine-tuned encoder predictions.")
    ap.add_argument("--ds",         default=None,
                    help="Only process this dataset key (e.g. vinc_control). Default: all.")
    args = ap.parse_args()

    orig_dir = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    out_dir  = orig_dir / "fa_cls_zrecon" / "ft_comparison"

    lgbm_orig    = joblib.load(str(orig_dir / "fa_cls_zrecon" / "model.pkl"))
    lgbm_crossds = joblib.load(str(orig_dir / "fa_cls_zrecon_crossds_labels" / "model.pkl"))
    print("LightGBM models loaded (orig + crossds).")

    # LightGBM trained on Annabel + Margaret vinc latents (classifier-only FT for vinc)
    lgbm_margaret_path = orig_dir / "fa_cls_zrecon" / "lgbm_margaret.pkl"
    if lgbm_margaret_path.exists():
        lgbm_margaret = joblib.load(str(lgbm_margaret_path))
        print("LightGBM (Margaret) loaded from cache.")
    else:
        print("Training LightGBM on Annabel + Margaret vinc latents...")
        annabel_label_csv = LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"
        annabel_lat_csv   = orig_dir / "blind_test" / "vinc_control_latents.csv"
        margaret_lbl_csv  = LABEL_DIR / "vinc_control_label_Margaret_2cls.csv"
        lgbm_margaret = _train_lgbm_on_margaret_latents(
            annabel_lat_csv, margaret_lbl_csv, annabel_lat_csv, annabel_label_csv)
        joblib.dump(lgbm_margaret, str(lgbm_margaret_path))
        print(f"  Saved → {lgbm_margaret_path.name}")

    ds_items   = [(k, v) for k, v in DS_CFG.items()
                  if args.ds is None or k == args.ds]
    ft_ae_cache: dict = {}  # cache FT models by ft_run_dir path (shared across datasets)

    for ds, cfg in ds_items:
        print(f"\n── {ds} ──")
        lat_csv = orig_dir / "blind_test" / f"{cfg['latents_key']}_latents.csv"
        if not lat_csv.exists():
            print(f"  [SKIP] latents not found: {lat_csv}")
            continue
        lat_all = pd.read_csv(lat_csv)
        lat_all[["frame", "px", "py", "ps"]] = lat_all["filename"].apply(
            lambda f: pd.Series(_parse_fname(f)))

        gt_map     = _load_gt_map(ds)
        all_frames = sorted(lat_all["frame"].unique())
        print(f"  frames: {len(all_frames)} total, {len(gt_map)} labeled patches")

        # Per-dataset: cls-FT LightGBM (vinc uses Margaret model; others use crossds)
        lgbm_cls_ft = lgbm_margaret if ds == "vinc_control" else lgbm_crossds

        # Per-dataset: load the correct FT AE (vinc → vinc+Margaret FT; ppax/pfak → crossds FT)
        # LightGBM FT is trained only on the dataset-appropriate labeled patches.
        ft_ae   = None
        lgbm_ft = None
        if args.include_ft:
            ft_run_dir  = RUN_DIR / f"{cfg['ft_run']}_{args.split}"
            dirs_name   = cfg.get("lgbm_ft_dirs", "_LABELED_PATCH_DIRS")
            lgbm_ft_patch_dirs = (
                _VINC_LABELED_PATCH_DIRS if dirs_name == "_VINC_LABELED_PATCH_DIRS"
                else _LABELED_PATCH_DIRS
            )
            ft_ae, lgbm_ft = _load_ft_ae_and_lgbm(
                ft_run_dir, ft_ae_cache, patch_dirs=lgbm_ft_patch_dirs)

        for frame in all_frames:
            df_frame = lat_all[lat_all["frame"] == frame].copy()
            if df_frame.empty:
                continue

            frame_fns  = {Path(fn).name for fn in df_frame["filename"]}
            has_labels = bool(gt_map) and bool(frame_fns & set(gt_map.keys()))
            print(f"  f{frame:04d}  ({'labeled' if has_labels else 'unlabeled'})", end="")

            df_before = _apply_lgbm(df_frame, lgbm_orig)
            df_cls_ft = _apply_lgbm(df_frame, lgbm_cls_ft)

            df_enc_ft = None
            if ft_ae is not None:
                import torch
                _device = torch.device("cpu")
                df_enc_lat = _run_ae_inference(ft_ae, cfg["patch_dir"], frame, _device)
                if not df_enc_lat.empty:
                    df_enc_ft = _apply_lgbm(df_enc_lat, lgbm_ft)

            print()
            out_name = f"ft_cmp_{ds}_f{frame:04d}.png"
            _make_4panel_figure(
                ds=ds, frame=frame,
                df_before=df_before,
                df_cls_ft=df_cls_ft,
                df_enc_ft=df_enc_ft,
                patch_dir=cfg["patch_dir"],
                gt_map=gt_map,
                out_path=out_dir / out_name,
            )

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
