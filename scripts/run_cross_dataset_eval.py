#!/usr/bin/env python3
"""
Evaluate trained AE models on multiple datasets and generate
violin plots of MSE / L1 / Hessian-L1 reconstruction quality.

Datasets evaluated:
  vinc   — training dataset (reuses existing recon TIFs, no re-inference)
  pfak   — new dataset (runs model inference)
  ppax   — new dataset
  nih3t3 — new dataset

For vinc train/val patches, groups are split by FA type (annotation_label_name).
For external datasets, groups are dataset_condition.

Modes:
  variants (default)  — iterate over hard-coded VARIANTS subdirs of <run_dir>
  sweep               — auto-discover all leaf model dirs under <run_dir>
                        (looks for model_final.pt or model_best.pt recursively)

Usage:
  python scripts/run_cross_dataset_eval.py <run_dir>
  python scripts/run_cross_dataset_eval.py <run_dir> --mode sweep \\
      --root-folder /net/projects/CLS/lding/data/fa_data_analysis
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.modelling.dataset import PatchDataset, MultiChannelPatchDataset


# ── constants ─────────────────────────────────────────────────────────────────

VARIANTS = [
    "semicon_both","baseline", "semisup_fa", "semisup_pos", "semisup_both",
    "conae", "semicon_fa", "semicon_pos",
]

KNOWN_DATASETS = {"vinc", "nih3t3", "ppax", "pfak"}

# (dataset_name, condition_name, relative patch path under root_folder)
# vinc is listed first so it appears at the left of the test-side x-axis.
EXTERNAL_DATASETS = [
    ("vinc",   "control", "ae_results/pax_ch_patch/cio_rb/vinc/control/tiff_patches32"),
    ("vinc",   "ycomp",   "ae_results/pax_ch_patch/cio_rb/vinc/ycomp/tiff_patches32"),
    ("pfak",   "control", "ae_results/pax_ch_patch/cio_rb/pfak/control/tiff_patches32"),
    ("pfak",   "ycomp",   "ae_results/pax_ch_patch/cio_rb/pfak/ycomp/tiff_patches32"),
    ("ppax",   "control", "ae_results/pax_ch_patch/cio_rb/ppax/control/tiff_patches32"),
    ("ppax",   "ycomp",   "ae_results/pax_ch_patch/cio_rb/ppax/ycomp/tiff_patches32"),
    ("nih3t3", "control", "ae_results/pax_ch_patch/cio_rb/nih3t3/control/tiff_patches32"),
    ("nih3t3", "ycomp",   "ae_results/pax_ch_patch/cio_rb/nih3t3/ycomp/tiff_patches32"),
]

# actin (ch3) single-channel external datasets
EXTERNAL_DATASETS_CH3 = [
    ("pfak",   "control", "ae_results/patches/cio_rb/pfak_ch3/control/tiff_patches32_mr10"),
    ("pfak",   "ycomp",   "ae_results/patches/cio_rb/pfak_ch3/ycomp/tiff_patches32_mr10"),
    ("ppax",   "control", "ae_results/patches/cio_rb/ppax_ch3/control/tiff_patches32_mr10"),
    ("ppax",   "ycomp",   "ae_results/patches/cio_rb/ppax_ch3/ycomp/tiff_patches32_mr10"),
    ("nih3t3", "control", "ae_results/patches/cio_rb/nih3t3_ch3/control/tiff_patches32_mr10"),
    ("nih3t3", "ycomp",   "ae_results/patches/cio_rb/nih3t3_ch3/ycomp/tiff_patches32_mr10"),
]

# 2-channel: (dataset_name, condition_name, ch1_rel_path, ch3_rel_path)
EXTERNAL_DATASETS_2CH = [
    ("pfak",   "control",
     "ae_results/patches/cio_rb/pfak/control/tiff_patches32_mr10",
     "ae_results/patches/cio_rb/pfak_ch3/control/tiff_patches32_mr10"),
    ("pfak",   "ycomp",
     "ae_results/patches/cio_rb/pfak/ycomp/tiff_patches32_mr10",
     "ae_results/patches/cio_rb/pfak_ch3/ycomp/tiff_patches32_mr10"),
    ("ppax",   "control",
     "ae_results/patches/cio_rb/ppax/control/tiff_patches32_mr10",
     "ae_results/patches/cio_rb/ppax_ch3/control/tiff_patches32_mr10"),
    ("ppax",   "ycomp",
     "ae_results/patches/cio_rb/ppax/ycomp/tiff_patches32_mr10",
     "ae_results/patches/cio_rb/ppax_ch3/ycomp/tiff_patches32_mr10"),
    ("nih3t3", "control",
     "ae_results/patches/cio_rb/nih3t3/control/tiff_patches32_mr10",
     "ae_results/patches/cio_rb/nih3t3_ch3/control/tiff_patches32_mr10"),
    ("nih3t3", "ycomp",
     "ae_results/patches/cio_rb/nih3t3/ycomp/tiff_patches32_mr10",
     "ae_results/patches/cio_rb/nih3t3_ch3/ycomp/tiff_patches32_mr10"),
]

METRICS = ["recon_nl1", "recon_mse", "recon_l1", "recon_hessian_l1",
           "recon_l1_ch0", "recon_nl1_ch0", "recon_l1_ch1", "recon_nl1_ch1"]
METRIC_LABELS = {
    "recon_nl1":        "Normalised L1  (L1 / mean|raw|)",
    "recon_mse":        "MSE",
    "recon_l1":         "L1 (MAE)",
    "recon_hessian_l1": "Hessian L1",
    "recon_l1_ch0":     "L1 — paxillin (ch0)",
    "recon_nl1_ch0":    "Normalised L1 — paxillin (ch0)",
    "recon_l1_ch1":     "L1 — actin (ch1)",
    "recon_nl1_ch1":    "Normalised L1 — actin (ch1)",
}

# Test-side group order (right of the separator on the violin x-axis).
# Datasets that were in training are skipped at eval time and won't appear here.
EXTERNAL_GROUP_ORDER = [
    "vinc_control",  "vinc_ycomp",
    "pfak_control",  "pfak_ycomp",
    "ppax_control",  "ppax_ycomp",
    "nih3t3_control","nih3t3_ycomp",
]


# ── patch metrics ─────────────────────────────────────────────────────────────

def _patch_hessian_l1(raw: np.ndarray, recon: np.ndarray) -> float:
    """Mean Frobenius norm of the Hessian of the residual (raw − recon)."""
    if raw.ndim == 3:
        return float(np.mean([_patch_hessian_l1(raw[c], recon[c])
                               for c in range(raw.shape[0])]))
    d = raw.astype(np.float64) - recon.astype(np.float64)
    dIxx = d[1:-1, 2:]  + d[1:-1, :-2] - 2 * d[1:-1, 1:-1]
    dIyy = d[2:,  1:-1] + d[:-2, 1:-1] - 2 * d[1:-1, 1:-1]
    dIxy = (d[2:, 2:] - d[2:, :-2] - d[:-2, 2:] + d[:-2, :-2]) / 4
    H_diff = np.sqrt(dIxx ** 2 + 2 * dIxy ** 2 + dIyy ** 2)
    return float(np.mean(H_diff))


def _channel_metrics(r64: np.ndarray, p64: np.ndarray) -> dict:
    """L1 and nL1 for a single 2-D patch (one channel)."""
    diff = r64 - p64
    l1   = float(np.mean(np.abs(diff)))
    return {"l1": l1, "nl1": l1 / (float(np.mean(np.abs(r64))) + 1e-8)}


def _metrics_from_arrays(raws: list[np.ndarray],
                          recons: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for r, p in zip(raws, recons):
        r64 = r.astype(np.float64)
        p64 = p.astype(np.float64)
        diff = r64 - p64
        l1   = float(np.mean(np.abs(diff)))
        row = {
            "recon_nl1":        l1 / (float(np.mean(np.abs(r64))) + 1e-8),
            "recon_mse":        float(np.mean(diff ** 2)),
            "recon_l1":         l1,
            "recon_hessian_l1": _patch_hessian_l1(r, p),
        }
        # per-channel metrics for 2-channel patches (C, H, W)
        if r64.ndim == 3 and r64.shape[0] >= 2:
            ch0 = _channel_metrics(r64[0], p64[0])
            ch1 = _channel_metrics(r64[1], p64[1])
            row["recon_l1_ch0"]  = ch0["l1"]
            row["recon_nl1_ch0"] = ch0["nl1"]
            row["recon_l1_ch1"]  = ch1["l1"]
            row["recon_nl1_ch1"] = ch1["nl1"]
        rows.append(row)
    return pd.DataFrame(rows)


# ── training-data metrics: read from existing recon TIFs ─────────────────────

def _dataset_from_condition(condition_name: str) -> str:
    """'ppax_control' → 'ppax', 'nih3t3_ycomp' → 'nih3t3'."""
    for ds in KNOWN_DATASETS:
        if condition_name == ds or condition_name.startswith(ds + "_"):
            return ds
    return condition_name.split("_")[0]


def _get_training_datasets(variant_dir: Path) -> set[str]:
    """Read the model config to find which canonical datasets were in training."""
    cfg = _read_model_config(variant_dir)
    datasets: set[str] = set()
    for entry in cfg.get("data", {}).get("patch_dirs", []):
        cond = str(entry.get("condition_name", ""))
        ds = _dataset_from_condition(cond)
        if ds in KNOWN_DATASETS:
            datasets.add(ds)
    return datasets or {"vinc"}   # safe fallback for legacy configs


def _training_recon_metrics(variant_dir: Path) -> pd.DataFrame | None:
    """Return patch-level metrics for training data, labeled by actual dataset.

    Replaces the old _vinc_metrics which hardcoded dataset='vinc'.  Now reads
    condition_name from patches_index.csv so that ppax, nih3t3, etc. are labeled
    correctly.  FA-type groups are still built for vinc rows that have annotations.
    """
    recon_dir = variant_dir / "recon"
    raw_tif   = recon_dir / "patches_raw.tif"
    rec_tif   = recon_dir / "patches_recon.tif"
    idx_csv   = recon_dir / "patches_index.csv"
    if not (raw_tif.exists() and rec_tif.exists() and idx_csv.exists()):
        return None

    raw_all = tifffile.imread(str(raw_tif))
    rec_all = tifffile.imread(str(rec_tif))
    idx_df  = pd.read_csv(idx_csv)
    met_df  = _metrics_from_arrays(list(raw_all), list(rec_all))
    df = pd.concat([idx_df.reset_index(drop=True), met_df.reset_index(drop=True)],
                   axis=1)

    # Extract true dataset name from condition_name (e.g. "ppax_control" → "ppax")
    df["dataset"] = df["condition_name"].apply(_dataset_from_condition)

    # Group by condition_name + split, consistent across all datasets
    df["group"] = df["condition_name"] + "_" + df["split"]  # e.g. "vinc_control_train"
    return df


# ── external dataset: run inference ──────────────────────────────────────────

def _infer_dataset(model, patch_dir: Path, device: str,
                   batch_size: int,
                   hist_map: "np.ndarray | None" = None,
                   input_divisor: float = 1.0) -> tuple[list, list]:
    """Run model on all patches in patch_dir; return (raws, recons).

    input_divisor : divide patches by this before model inference (matches sc2
      training preprocessing).  Returned raws are also divided so that metrics
      are computed in the same intensity space as the stored vinc patches.

    hist_map : (2, N) array with [src_q, ref_q].  When provided, the forward
      map is applied after the input_divisor scaling.
    """
    ds = PatchDataset(str(patch_dir), condition=0, condition_name="")
    if len(ds) == 0:
        return [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)

    cls_name = type(model).__name__
    if "SemiSup" in cls_name:
        model_type = "semisup"
    elif "Contrastive" in cls_name:
        model_type = "contrastive"
    elif "VAE" in cls_name:
        model_type = "vae"
    else:
        model_type = "ae"

    src_q = hist_map[0] if hist_map is not None else None
    ref_q = hist_map[1] if hist_map is not None else None

    raws, recons = [], []
    with torch.no_grad():
        for batch in loader:
            x_orig = batch[0]                         # (B, H, W) or (B,1,H,W)
            if x_orig.dim() == 3:
                x_orig = x_orig.unsqueeze(1)          # (B,1,H,W)

            # apply input_divisor to match training preprocessing
            if input_divisor != 1.0:
                x_orig = x_orig / input_divisor

            if hist_map is not None:
                x_np = x_orig.numpy()
                x_mapped = np.interp(x_np, src_q, ref_q).astype(np.float32)
                x_in = torch.from_numpy(x_mapped).to(device)
            else:
                x_in = x_orig.to(device)

            if model_type == "vae":
                x_hat, mu, _, _ = model(x_in)
            elif model_type == "semisup":
                x_hat, _, _ = model(x_in)
            else:
                x_hat, _ = model(x_in)

            x_hat_np = x_hat.cpu().numpy()
            if hist_map is not None:
                x_hat_np = np.interp(x_hat_np, ref_q, src_q).astype(np.float32)

            for raw_p, rec_p in zip(x_orig.numpy(), x_hat_np):
                if raw_p.shape[0] == 1:
                    raw_p, rec_p = raw_p[0], rec_p[0]
                raws.append(raw_p.astype(np.float32))
                recons.append(rec_p.astype(np.float32))
    return raws, recons


def _infer_dataset_2ch(model, ch1_dir: Path, ch3_dir: Path,
                       device: str, batch_size: int,
                       input_divisor: float = 1.0) -> tuple[list, list]:
    """Run 2-channel model on stacked (ch1, ch3) patches; return (raws, recons)."""
    if not ch1_dir.exists() or not ch3_dir.exists():
        return [], []
    ds = MultiChannelPatchDataset([str(ch1_dir), str(ch3_dir)],
                                  condition=0, condition_name="")
    if len(ds) == 0:
        return [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)

    cls_name = type(model).__name__
    model_type = "contrastive" if "Contrastive" in cls_name else \
                 "semisup" if "SemiSup" in cls_name else \
                 "vae" if "VAE" in cls_name else "ae"

    raws, recons = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0]                         # (B, 2, H, W) float
            if input_divisor != 1.0:
                x = x / input_divisor
            x_dev = x.to(device)
            if model_type == "vae":
                x_hat, mu, _, _ = model(x_dev)
            elif model_type == "semisup":
                x_hat, _, _ = model(x_dev)
            else:
                x_hat, _ = model(x_dev)
            for raw_p, rec_p in zip(x.numpy(), x_hat.cpu().numpy()):
                raws.append(raw_p.astype(np.float32))    # (2, H, W)
                recons.append(rec_p.astype(np.float32))
    return raws, recons


def _external_metrics(model, dataset_name: str, condition_name: str,
                      patch_dir: Path, device: str,
                      batch_size: int,
                      hist_map: "np.ndarray | None" = None,
                      input_divisor: float = 1.0) -> pd.DataFrame | None:
    if not patch_dir.exists():
        print(f"    [skip] patch dir not found: {patch_dir}")
        return None
    print(f"    inference on {dataset_name}/{condition_name} "
          f"({len(list(patch_dir.glob('*.tif')))} patches) "
          f"[÷{input_divisor}] …", flush=True)
    raws, recons = _infer_dataset(model, patch_dir, device, batch_size,
                                  hist_map=hist_map,
                                  input_divisor=input_divisor)
    if not raws:
        return None
    met_df = _metrics_from_arrays(raws, recons)
    met_df["dataset"]        = dataset_name
    met_df["condition_name"] = condition_name
    met_df["split"]          = "test"
    met_df["group"]          = dataset_name + "_" + condition_name
    return met_df


def _external_metrics_2ch(model, dataset_name: str, condition_name: str,
                           ch1_dir: Path, ch3_dir: Path,
                           device: str, batch_size: int,
                           input_divisor: float = 1.0) -> pd.DataFrame | None:
    if not ch1_dir.exists():
        print(f"    [skip] ch1 dir not found: {ch1_dir}")
        return None
    if not ch3_dir.exists():
        print(f"    [skip] ch3 dir not found: {ch3_dir}")
        return None
    n_patches = len(list(ch1_dir.glob("*.tif")))
    print(f"    inference 2ch on {dataset_name}/{condition_name} ({n_patches} patches) "
          f"[÷{input_divisor}] …", flush=True)
    raws, recons = _infer_dataset_2ch(model, ch1_dir, ch3_dir, device, batch_size,
                                      input_divisor=input_divisor)
    if not raws:
        return None
    met_df = _metrics_from_arrays(raws, recons)
    met_df["dataset"]        = dataset_name
    met_df["condition_name"] = condition_name
    met_df["split"]          = "test"
    met_df["group"]          = dataset_name + "_" + condition_name
    return met_df


# ── plotting ──────────────────────────────────────────────────────────────────

_LABEL_SUBS = [
    ("nascent adhesion",   "Nas Adh"),
    ("no adhesion",        "No Adh"),
    ("focal complex",      "Foc Cpx"),
    ("focal adhesion",     "Foc Adh"),
    ("fibrillar adhesion", "Fib Adh"),
    ("stress fiber",       "Str Fib"),
    ("unlabeled",          "unlbl"),
    ("nih3t3",             "ds4"),
    ("pfak",               "ds2"),
    ("ppax",               "ds3"),
    ("vinc",               "ds1"),
    ("control",            "ctrl"),
    ("ycomp",              "yc"),
    ("train",              "tr"),
    ("val",                "val"),
    ("test",               "tst"),
]

def _shorten(label: str) -> str:
    """Abbreviate long group labels for compact x-axis display."""
    import re
    result = label
    for long, short in _LABEL_SUBS:
        result = re.sub(re.escape(long), short, result, flags=re.IGNORECASE)
    return result


def _build_group_order(variant_df: pd.DataFrame) -> list[str]:
    """Return ordered x-axis group list: training groups first, then test groups.

    Training groups: anything with split in {train, val} — sorted alphabetically.
    Test groups: the canonical EXTERNAL_GROUP_ORDER entries that are present.
    """
    ext_set = set(EXTERNAL_GROUP_ORDER)
    train_groups = sorted(
        g for g in variant_df["group"].unique() if g not in ext_set
    )
    test_groups = [g for g in EXTERNAL_GROUP_ORDER
                   if g in variant_df["group"].values]
    return train_groups + test_groups


def _violin_plot_single(variant_df: pd.DataFrame, variant_name: str,
                         metric: str, save_path: Path) -> None:
    """One figure per variant per metric, x = groups."""
    group_order = _build_group_order(variant_df)
    if not group_order:
        return

    sub = variant_df[variant_df[metric].notna()].copy()
    present = [g for g in group_order if g in sub["group"].values]
    if sub.empty or not present:
        return

    display_labels = [_shorten(g) for g in present]

    fig, ax = plt.subplots(figsize=(max(12, len(present) * 1.6), 9.5))
    sns.violinplot(data=sub, x="group", y=metric,
                   order=present, ax=ax,
                   inner="box", cut=0, linewidth=0.8, width=1.2,
                   color="cornflowerblue")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=40)
    ax.tick_params(axis="y", labelsize=36)
    # ax.set_title(f"{variant_name} — {METRIC_LABELS[metric]}", fontsize=36)
    ax.set_xlabel("")
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=36)

    # fixed [0,1] for nL1 metrics (comparable across models); auto-range for others
    if "nl1" in metric:
        ax.set_ylim(0.0, 1.0)
    else:
        y99 = float(sub[metric].quantile(0.99))
        ymin = float(sub[metric].quantile(0.001))
        ax.set_ylim(max(0, ymin * 0.95), y99 * 1.05)

    # vertical separator between training groups and test groups
    ext_set = set(EXTERNAL_GROUP_ORDER)
    n_train = sum(1 for g in present if g not in ext_set)
    if 0 < n_train < len(present):
        ax.axvline(n_train - 0.5, color="grey", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def _find_model_dirs(root: Path) -> list[tuple[str, Path]]:
    """Recursively find all dirs containing model_final.pt or model_best.pt.

    Returns list of (name, dir) where name is the path relative to root
    with '/' replaced by '_'.
    """
    results = []
    for model_pt in sorted(root.rglob("model_final.pt")) + sorted(root.rglob("model_best.pt")):
        d = model_pt.parent
        rel = d.relative_to(root)
        name = str(rel).replace("/", "_")
        # prefer model_final.pt; skip if already added from model_final scan
        if not any(p == d for _, p in results):
            results.append((name, d))
    return results


def _read_model_config(variant_dir: Path) -> dict:
    """Load the first YAML config found in variant_dir; return empty dict on failure."""
    try:
        import yaml
        for yf in list(variant_dir.glob("*.yaml")) + list(variant_dir.glob("*.yml")):
            with open(yf) as fh:
                return yaml.safe_load(fh) or {}
    except Exception:
        pass
    return {}


def _read_input_divisor(variant_dir: Path) -> float:
    """Read enlarged_crop.input_divisor from the YAML config in variant_dir."""
    cfg = _read_model_config(variant_dir)
    ec = cfg.get("enlarged_crop", {})
    if ec.get("enabled", False):
        return float(ec.get("input_divisor", 1.0))
    return 1.0


def _is_ch3_model(variant_dir: Path) -> bool:
    """Return True if the model was trained on actin (ch3) single-channel patches."""
    cfg = _read_model_config(variant_dir)
    # Check enlarged_crop channel
    ec = cfg.get("enlarged_crop", {})
    if ec.get("channel") == "act":
        return True
    # Fallback: check training patch paths
    for entry in cfg.get("data", {}).get("patch_dirs", []):
        path = str(entry.get("path", ""))
        if "_ch3" in path and "channel_dirs" not in entry:
            return True
    return False


def _is_2ch_model(variant_dir: Path) -> bool:
    """Return True if the model was trained on 2-channel (ch1+ch3) stacked patches."""
    cfg = _read_model_config(variant_dir)
    for entry in cfg.get("data", {}).get("patch_dirs", []):
        if "channel_dirs" in entry:
            return True
    if cfg.get("model", {}).get("no_ch", 1) >= 2:
        return True
    return False


def _load_hist_map_for_ds(hist_map_dir: Path | None, ds_name: str):
    """Load the forward+inverse map for a dataset from hist_map_dir."""
    if hist_map_dir is None:
        return None
    map_file = hist_map_dir / f"{ds_name}_map.npz"
    if not map_file.exists():
        return None
    data = np.load(str(map_file))
    return np.stack([data["src_q"], data["ref_q"]])   # (2, N): row0=src_q, row1=ref_q


def _run_variant(variant: str, variant_dir: Path, run_dir: Path,
                 root_folder: Path, device: str, batch_size: int,
                 hist_map_dir: Path | None = None,
                 two_channel: bool = False) -> pd.DataFrame | None:
    model_pt = variant_dir / "model_final.pt"
    if not model_pt.exists():
        model_pt = variant_dir / "model_best.pt"
    if not model_pt.exists():
        print(f"[skip] {variant} — no model checkpoint found")
        return None

    print(f"── {variant} ──────────────────────────────")
    input_divisor    = _read_input_divisor(variant_dir)
    ch3_model        = _is_ch3_model(variant_dir)
    training_datasets = _get_training_datasets(variant_dir)
    # auto-detect 2ch models even when --two-channel flag was not passed
    if not two_channel:
        two_channel = _is_2ch_model(variant_dir)
    ext_ds_list = EXTERNAL_DATASETS_CH3 if ch3_model else EXTERNAL_DATASETS
    print(f"  input_divisor: {input_divisor}  ch3_model: {ch3_model}"
          f"  two_channel: {two_channel}  training_datasets: {sorted(training_datasets)}")
    variant_rows = []

    # 1. Training data from existing recon TIFs (correctly labeled by actual dataset)
    train_df = _training_recon_metrics(variant_dir)
    if train_df is not None:
        train_df["variant"] = variant
        variant_rows.append(train_df)
        print(f"  training recon: {len(train_df)} patches, "
              f"datasets={sorted(train_df['dataset'].unique())}, "
              f"{train_df['group'].nunique()} groups")
    else:
        print(f"  training recon: TIFs not found — skipping")

    # 2. External / test datasets — skip any that were in training (already covered above)
    print(f"  loading model …", flush=True)
    model = torch.load(str(model_pt), map_location=device, weights_only=False)
    model.eval()

    if two_channel:
        for ds_name, cond_name, ch1_rel, ch3_rel in EXTERNAL_DATASETS_2CH:
            if ds_name in training_datasets:
                print(f"    [skip] {ds_name}/{cond_name} — in training data")
                continue
            ch1_dir = root_folder / ch1_rel
            ch3_dir = root_folder / ch3_rel
            ext_df = _external_metrics_2ch(model, ds_name, cond_name,
                                            ch1_dir, ch3_dir, device, batch_size,
                                            input_divisor=input_divisor)
            if ext_df is not None:
                ext_df["variant"] = variant
                variant_rows.append(ext_df)
    else:
        for ds_name, cond_name, rel_path in ext_ds_list:
            if ds_name in training_datasets:
                print(f"    [skip] {ds_name}/{cond_name} — in training data")
                continue
            patch_dir = root_folder / rel_path
            hist_map  = _load_hist_map_for_ds(hist_map_dir, ds_name)
            ext_df = _external_metrics(model, ds_name, cond_name,
                                        patch_dir, device, batch_size,
                                        hist_map=hist_map,
                                        input_divisor=input_divisor)
            if ext_df is not None:
                ext_df["variant"] = variant
                variant_rows.append(ext_df)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    if not variant_rows:
        return None

    variant_df = pd.concat(variant_rows, ignore_index=True)
    for metric in METRICS:
        if metric not in variant_df.columns:
            continue
        save_path = variant_dir / f"cross_dataset_{metric}.png"
        _violin_plot_single(variant_df, variant, metric, save_path)
    print()
    return variant_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root-folder",
                        default="/net/projects/CLS/lding/data/fa_data_analysis")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=["variants", "sweep"], default="variants",
                        help="variants: iterate hard-coded VARIANTS subdirs; "
                             "sweep: auto-discover all model dirs recursively")
    parser.add_argument("--hist-map-dir", type=Path, default=None,
                        help="Directory with {ds}_map.npz files.  When set, external "
                             "dataset metrics are computed in original intensity space: "
                             "forward map applied to model input, inverse map applied to "
                             "recon output before computing L1/MSE/Hessian.")
    parser.add_argument("--two-channel", action="store_true",
                        help="Use 2-channel inference (pax ch1 + actin ch3) for external "
                             "datasets.  Requires ch3 patches under patches/cio_rb/*_ch3/.")
    args = parser.parse_args()

    run_dir     = args.run_dir
    root_folder = Path(args.root_folder)
    batch_size  = args.batch_size
    device      = ("cuda" if torch.cuda.is_available() else "cpu") \
                  if args.device == "auto" else args.device

    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    print(f"Run dir    : {run_dir}")
    print(f"Mode       : {args.mode}")
    print(f"Root folder: {root_folder}")
    print(f"Device     : {device}")
    print(f"Batch size : {batch_size}")
    print()

    if args.mode == "sweep":
        candidates = _find_model_dirs(run_dir)
        if not candidates:
            sys.exit("No model checkpoints found under run_dir.")
        print(f"Found {len(candidates)} model dirs.")
    else:
        candidates = [
            (v, run_dir / v)
            for v in VARIANTS
            if (run_dir / v).is_dir()
        ]

    all_rows = []
    for variant, variant_dir in candidates:
        df = _run_variant(variant, variant_dir, run_dir,
                          root_folder, device, batch_size,
                          hist_map_dir=args.hist_map_dir,
                          two_channel=args.two_channel)
        if df is not None:
            all_rows.append(df)

    if not all_rows:
        sys.exit("No results collected.")

    combined = pd.concat(all_rows, ignore_index=True)
    out_csv = run_dir / "cross_dataset_recon_metrics.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Combined metrics → {out_csv}  ({len(combined)} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
