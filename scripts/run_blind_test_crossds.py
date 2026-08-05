#!/usr/bin/env python3
"""
run_blind_test_crossds.py

Blind cross-dataset test: apply a trained Annabel-vinc AE + LightGBM classifier
to vinc (control+ycomp), ppax (control), and pfak (control) datasets, then
evaluate against Margaret's independent label CSVs (labels_*_20260521.csv).

For each of the 9 Annabel-vinc result dirs, for each of the 4 (ds, cond) pairs,
for each of the 2 feature spaces (z_recon, z_proj):
  1. Run AE inference using EnlargedCropDataset (pax channel, cio_mode_prt frames)
  2. Apply saved LightGBM model.pkl
  3. Match predictions to blind label CSV by unique_ID
  4. Exclude "Uncertain" patches
  5. Map 5-class predictions to 2-class if model is supcon2
  6. Save confusion matrices, F1, metrics to blind_test/<ds>_<cond>_<feat>/

Usage:
    python scripts/run_blind_test_crossds.py <result_dir>
    python scripts/run_blind_test_crossds.py <result_dir> [--device cuda]
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
import joblib
import re
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PATCHES_ROOT = DATA_ROOT / "ae_results" / "patches" / "cio"
FRAMES_ROOT  = DATA_ROOT / "ae_results" / "source_frames" / "cio_mode_prt"
LABEL_DIR    = DATA_ROOT / "labelling"

# (dataset, condition, label_csv)
EVAL_SETS = [
    ("vinc",  "control", "labels_vinc_20260521.csv"),
    ("vinc",  "ycomp",   "labels_vinc_20260521.csv"),
    ("ppax",  "control", "labels_ppax_20260521.csv"),
    ("pfak",  "control", "labels_pfak_20260521.csv"),
]

LABEL_ORDER_5 = [
    "No adhesion",
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]
LABEL_ORDER_2 = ["No adhesion", "adhesion"]
ADHESION_TYPES = {"focal adhesion", "Nascent Adhesion", "fibrillar adhesion", "focal complex"}

_COORD_RE = re.compile(r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)\.(tiff?)$', re.IGNORECASE)


def _patch_to_uid(fname: str) -> str:
    """control_f0001x0592y0560ps32.tif → control-f0001x0592y0560ps32.tif"""
    return re.sub(r'_(?=f\d{4})', '-', Path(fname).name, count=1)


def _load_config(result_dir: Path) -> dict:
    for yf in result_dir.glob("*.yaml"):
        with open(yf) as fh:
            cfg = yaml.safe_load(fh) or {}
        if "model" in cfg and "data" in cfg:
            return cfg
    return {}


def _infer_model_type(result_dir: Path) -> str:
    """Return 'supcon2', 'supcon5', or 'conae' from result dir name."""
    name = result_dir.name
    if "supcon2" in name:
        return "supcon2"
    if "supcon5" in name:
        return "supcon5"
    return "conae"


def run_inference(result_dir: Path, ds: str, cond: str, device: str,
                  cfg: dict, model=None) -> pd.DataFrame | None:
    """Run AE inference on (ds, cond) and return DataFrame with z_ and p_ columns.

    If model is None, loads model_best.pt from result_dir.
    Pass a pre-loaded model to avoid repeated disk I/O across datasets.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from subcellae.modelling.dataset import EnlargedCropDataset
    from torch.utils.data import DataLoader

    patch_dir = PATCHES_ROOT / ds / cond / "tiff_patches32_mr10"
    frame_dir = FRAMES_ROOT / ds / cond
    if not patch_dir.exists():
        print(f"  [skip] patch_dir not found: {patch_dir}", flush=True)
        return None
    if not frame_dir.exists():
        print(f"  [skip] frame_dir not found: {frame_dir}", flush=True)
        return None

    ec = cfg.get("enlarged_crop", {})
    channel      = ec.get("channel", "pax")
    context_size = int(ec.get("context_size", 58))
    pad_size     = int(ec.get("pad_size", 64))
    input_divisor = float(ec.get("input_divisor", 2.0))
    input_ps     = int(cfg.get("model", {}).get("input_ps", 32))

    print(f"  Building dataset: {ds}/{cond}  ch={channel}  ctx={context_size}", flush=True)
    dataset = EnlargedCropDataset(
        patch_dir    = str(patch_dir),
        frame_dir    = str(frame_dir),
        channel      = channel,
        condition    = 0,
        condition_name = cond,
        context_size = context_size,
        patch_size   = input_ps,
        pad_size     = pad_size,
        input_divisor = input_divisor,
    )
    if len(dataset) == 0:
        print(f"  [skip] empty dataset", flush=True)
        return None

    if model is None:
        model_path = result_dir / "model_best.pt"
        print(f"  Loading model from {model_path.name}", flush=True)
        model = torch.load(str(model_path), map_location=device, weights_only=False)
        model.eval()
        model.to(device)

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    all_z, all_p, all_paths = [], [], []
    h2 = context_size // 2
    c_start = h2 - input_ps // 2
    c_end   = c_start + input_ps

    with torch.no_grad():
        for batch in loader:
            x_raw = batch[0].to(device)               # (B, 1, ctx, ctx)
            x = x_raw[:, :, c_start:c_end, c_start:c_end]   # center-crop to input_ps
            x_hat, z = model(x)                        # z = z_recon (B, latent_dim)
            z_proj   = model.project(z)                # z_proj (B, proj_dim)
            all_z.append(z.cpu().numpy())
            all_p.append(z_proj.cpu().numpy())
            all_paths.extend(batch[4])                 # batch[4] = paths

    z_arr = np.concatenate(all_z, axis=0)   # (N, latent_dim)
    p_arr = np.concatenate(all_p, axis=0)   # (N, proj_dim)

    latent_dim = z_arr.shape[1]
    proj_dim   = p_arr.shape[1]

    df = pd.DataFrame()
    df["filename"] = [Path(p).name for p in all_paths]
    df["unique_ID"] = df["filename"].apply(_patch_to_uid)
    df["ds"]   = ds
    df["cond"] = cond
    for i in range(latent_dim):
        df[f"z_{i}"] = z_arr[:, i]
    for i in range(proj_dim):
        df[f"p_{i}"] = p_arr[:, i]

    print(f"  Inference done: {len(df)} patches", flush=True)
    return df


def _plot_cm(cm, labels, title, out_path: Path, normalize: bool = False):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = np.where(row_sums > 0, cm.astype(float) / np.where(row_sums > 0, row_sums, 1), 0.0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
    disp = ConfusionMatrixDisplay(cm_plot, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, values_format=fmt)
    ax.set_title(title, fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def evaluate_predictions(pred_df: pd.DataFrame, label_df: pd.DataFrame,
                         label_order: list[str], feat: str, model_type: str,
                         out_dir: Path, ds: str, cond: str):
    """
    Match predictions to blind labels, exclude Uncertain, compute metrics.
    pred_df: has columns unique_ID, pred_label_<feat>
    label_df: has columns unique_ID, classification (filter to ds+cond)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_col = f"pred_{feat}"
    merged   = label_df.merge(pred_df[["unique_ID", pred_col]], on="unique_ID", how="inner")
    n_total  = len(label_df)
    n_matched = len(merged)
    print(f"  [{ds}/{cond}/{feat}] Matched {n_matched}/{n_total} labeled patches", flush=True)

    # Drop Uncertain
    merged = merged[merged["classification"] != "Uncertain"].copy()

    if model_type == "supcon2":
        merged["classification"] = merged["classification"].apply(
            lambda x: "adhesion" if x in ADHESION_TYPES else x
        )
        eval_order = LABEL_ORDER_2
    else:
        eval_order = LABEL_ORDER_5

    y_true = merged["classification"].tolist()
    y_pred = merged[pred_col].tolist()

    if not y_true:
        print(f"  [skip] no matched labels after filtering", flush=True)
        return

    present = sorted(set(y_true) | set(y_pred), key=lambda x: eval_order.index(x) if x in eval_order else 999)
    present_names = [x for x in eval_order if x in present] + [x for x in present if x not in eval_order]

    report_dict = classification_report(
        y_true, y_pred,
        labels=present_names,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        y_true, y_pred,
        labels=present_names,
        target_names=present_names,
        zero_division=0,
    )

    # Summary metrics
    metrics = {
        "ds": ds, "cond": cond, "feat": feat,
        "n_matched": n_matched, "n_labeled": n_total, "n_eval": len(merged),
        "accuracy": report_dict.get("accuracy", 0.0),
        "macro_f1": report_dict.get("macro avg", {}).get("f1-score", 0.0),
        "weighted_f1": report_dict.get("weighted avg", {}).get("f1-score", 0.0),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "metrics.txt").write_text(report_str)
    print(report_str, flush=True)

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=present_names)
    title = f"{ds}/{cond}  {feat}  blind test"
    _plot_cm(cm, present_names, title + " (counts)",    out_dir / "confusion_matrix_counts.png", normalize=False)
    _plot_cm(cm, present_names, title + " (row-norm)",  out_dir / "confusion_matrix_norm.png",   normalize=True)

    # Per-patch predictions
    merged[["unique_ID", "classification", pred_col]].to_csv(out_dir / "predictions.csv", index=False)

    print(f"  Saved to {out_dir.relative_to(DATA_ROOT)}", flush=True)


def run(result_dir: Path, device: str = "cpu"):
    if not result_dir.exists():
        sys.exit(f"result_dir not found: {result_dir}")

    cfg        = _load_config(result_dir)
    model_type = _infer_model_type(result_dir)
    label_order = LABEL_ORDER_2 if model_type == "supcon2" else LABEL_ORDER_5
    blind_dir  = result_dir / "blind_test"
    blind_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"Result dir : {result_dir.name}", flush=True)
    print(f"Model type : {model_type}  |  label order: {label_order}", flush=True)
    print(f"Device     : {device}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Load LightGBM classifiers
    cls_paths = {
        "zrecon": result_dir / "fa_cls_zrecon" / "model.pkl",
        "zproj":  result_dir / "fa_cls_zproj"  / "model.pkl",
    }
    cls_models = {}
    for feat, pkl_path in cls_paths.items():
        if not pkl_path.exists():
            print(f"  [warn] classifier not found: {pkl_path}", flush=True)
            continue
        cls_models[feat] = joblib.load(str(pkl_path))
        print(f"  Loaded classifier: {feat}", flush=True)

    if not cls_models:
        sys.exit("No classifiers found — aborting")

    # Load AE model once (reused across all datasets)
    model_path = result_dir / "model_best.pt"
    print(f"\n  Loading AE model from {model_path.name} …", flush=True)
    ae_model = torch.load(str(model_path), map_location=device, weights_only=False)
    ae_model.eval()
    ae_model.to(device)
    print(f"  Model loaded.", flush=True)

    # Collect per-dataset label CSVs
    label_dfs: dict[str, pd.DataFrame] = {}
    for lcsv in ["labels_vinc_20260521.csv", "labels_ppax_20260521.csv", "labels_pfak_20260521.csv"]:
        key = lcsv.split("_")[1]   # "vinc", "ppax", "pfak"
        lpath = LABEL_DIR / lcsv
        if not lpath.exists():
            print(f"  [warn] label CSV not found: {lpath}", flush=True)
            continue
        label_dfs[key] = pd.read_csv(lpath)

    # Run inference + eval for each (ds, cond)
    for ds, cond, lcsv_key in [
            ("vinc",  "control", "vinc"),
            ("vinc",  "ycomp",   "vinc"),
            ("ppax",  "control", "ppax"),
            ("pfak",  "control", "pfak"),
    ]:
        print(f"\n--- Inference: {ds}/{cond} ---", flush=True)
        inf_df = run_inference(result_dir, ds, cond, device, cfg, model=ae_model)
        if inf_df is None:
            continue

        # Save full latents for this (ds, cond)
        inf_df.to_csv(blind_dir / f"{ds}_{cond}_latents.csv", index=False)

        if lcsv_key not in label_dfs:
            print(f"  [skip eval] no label CSV for {ds}", flush=True)
            continue

        label_df_all = label_dfs[lcsv_key]
        # Filter to this condition
        label_df = label_df_all[label_df_all["condition"] == cond].copy()
        print(f"  Label CSV: {len(label_df)} rows for {ds}/{cond}", flush=True)

        for feat, lgbm_model in cls_models.items():
            prefix = "z_" if feat == "zrecon" else "p_"
            feat_cols = [c for c in inf_df.columns if c.startswith(prefix)]
            X = inf_df[feat_cols].values

            preds = lgbm_model.predict(X)
            pred_series = pd.Series(preds, index=inf_df.index)
            # Map integer predictions back to class names
            if hasattr(lgbm_model, "classes_"):
                classes = lgbm_model.classes_
                pred_labels = [label_order[int(c)] if isinstance(c, (int, np.integer)) and int(c) < len(label_order)
                               else str(c) for c in preds]
            else:
                pred_labels = [label_order[int(p)] if int(p) < len(label_order) else str(p) for p in preds]

            pred_df = inf_df[["unique_ID"]].copy()
            pred_df[f"pred_{feat}"] = pred_labels

            out_sub = blind_dir / f"{ds}_{cond}_{feat}"
            evaluate_predictions(
                pred_df, label_df, label_order, feat, model_type, out_sub, ds, cond
            )

    print(f"\nDone.  Results in {blind_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--device", default="cpu",
                    help="Torch device (default: cpu; use 'cuda' for GPU)")
    args = ap.parse_args()
    run(args.result_dir, device=args.device)


if __name__ == "__main__":
    main()
