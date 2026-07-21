#!/usr/bin/env python3
"""
run_protein_sweep_eval.py
=========================
Evaluate a trained protein-sweep model on all datasets with matching content,
then produce per-run metrics CSV and violin plots.

Evaluation dataset rule:
  - pax/zyx/act models → evaluate on all 4 datasets (all have these channels)
  - vinc model         → evaluate on vinc + nih3t3 only (both have vinculin ch0)
  - pfak model         → evaluate on pfak only
  - ppax model         → evaluate on ppax only
  - 4ch_vinc model     → evaluate on vinc + nih3t3 (matching ch0 group)
  - 4ch_pfak model     → evaluate on pfak only
  - 4ch_ppax model     → evaluate on ppax only
  - 3ch_pza model      → evaluate on all 4 datasets (all have pax+zyx+act)

Reads the YAML config copied into <run_dir> at training time to auto-detect
the channel/protein type.

Usage:
  python scripts/run_protein_sweep_eval.py <run_dir>
  python scripts/run_protein_sweep_eval.py <sweep_root> --all   # process all subdirs
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
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.modelling.dataset import PatchDataset, MultiChannelPatchDataset

# ---------------------------------------------------------------------------
ROOT        = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PATCH_BASE  = ROOT / "ae_results/patches/cio_rb"

ALL_DATASETS    = ["vinc", "pfak", "ppax", "nih3t3"]
ALL_CONDITIONS  = ["control", "ycomp"]
VINC_GROUP      = ["vinc", "nih3t3"]   # datasets sharing vinculin ch0

# Which datasets have each single-channel protein
PROTEIN_DATASETS = {
    "pax":  ALL_DATASETS,
    "zyx":  ALL_DATASETS,
    "act":  ALL_DATASETS,
    "vinc": VINC_GROUP,
    "pfak": ["pfak"],
    "ppax": ["ppax"],
}

# Patch directory suffix for each protein
PROTEIN_SUFFIX = {
    "pax":  "",
    "zyx":  "_ch2",
    "act":  "_ch3",
    "vinc": "_ch0",
    "pfak": "_ch0",
    "ppax": "_ch0",
}

# For multi-channel models, the per-channel suffixes (ch0, pax, zyx, act)
MULTICHANNEL_SUFFIXES = ["_ch0", "", "_ch2", "_ch3"]

_DS_SHORT   = {"vinc": "ds1", "pfak": "ds2", "ppax": "ds3", "nih3t3": "ds4"}
_COND_SHORT = {"control": "ctrl", "ycomp": "yc"}


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _read_config(run_dir: Path) -> dict:
    for yf in list(run_dir.glob("*.yaml")) + list(run_dir.glob("*.yml")):
        with open(yf) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _detect_channel_type(cfg: dict) -> dict:
    """Return info about what this model was trained on.

    Returns dict with keys:
      model_type  : "single" | "4ch" | "3ch"
      proteins    : list of protein names (ordered, matches channel order)
      eval_datasets: list of dataset names to evaluate on
      ch_suffixes : list of patch suffixes per channel (for multi-ch inference)
    """
    enlcrop = cfg.get("enlarged_crop", {})
    ch = enlcrop.get("channel", None)
    no_ch = int(cfg.get("model", {}).get("no_ch", 1))

    if isinstance(ch, list):
        # multi-channel
        if len(ch) == 4:
            # 4ch: [ch0_name, "pax", "zyx", "act"]
            ch0_name = ch[0]
            if ch0_name == "vinc":
                eval_ds = VINC_GROUP
            elif ch0_name == "pfak":
                eval_ds = ["pfak"]
            elif ch0_name == "ppax":
                eval_ds = ["ppax"]
            else:
                eval_ds = ALL_DATASETS
            return dict(model_type="4ch", proteins=ch,
                        eval_datasets=eval_ds,
                        ch_suffixes=MULTICHANNEL_SUFFIXES)
        elif len(ch) == 3:
            # 3ch: ["pax", "zyx", "act"]
            return dict(model_type="3ch", proteins=ch,
                        eval_datasets=ALL_DATASETS,
                        ch_suffixes=["", "_ch2", "_ch3"])
        else:
            raise ValueError(f"Unexpected channel list length {len(ch)}: {ch}")
    else:
        # single-channel
        protein = str(ch) if ch else "pax"
        return dict(model_type="single", proteins=[protein],
                    eval_datasets=PROTEIN_DATASETS.get(protein, ALL_DATASETS),
                    ch_suffixes=[PROTEIN_SUFFIX.get(protein, "")])


def _get_training_conditions(cfg: dict) -> set[str]:
    """Return set of condition_names used in training."""
    conds = set()
    for entry in cfg.get("data", {}).get("patch_dirs", []):
        cname = entry.get("condition_name", "")
        if cname:
            conds.add(cname)
    return conds


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_model(run_dir: Path, device: str):
    """Load best or final model from run_dir."""
    for name in ("model_best.pt", "model_final.pt"):
        mp = run_dir / name
        if mp.exists():
            model = torch.load(mp, map_location=device, weights_only=False)
            model.eval()
            return model
    raise FileNotFoundError(f"No model found in {run_dir}")


def _infer_single(model, patch_dir: Path, device: str, input_divisor: float,
                  batch_size: int = 256) -> tuple[list, list]:
    """Single-channel inference. Returns (raws, recons) as numpy arrays."""
    if not patch_dir.exists():
        return [], []
    ds = PatchDataset(str(patch_dir), condition=0, condition_name="")
    if len(ds) == 0:
        return [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    raws, recons = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0]
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if input_divisor != 1.0:
                x = x / input_divisor
            x_in = x.to(device)
            out = model(x_in)
            x_hat = out[0] if isinstance(out, (tuple, list)) else out
            raws.extend(x.squeeze(1).cpu().numpy())
            recons.extend(x_hat.squeeze(1).cpu().numpy())
    return raws, recons


def _infer_multi(model, ch_dirs: list[Path], device: str, input_divisor: float,
                 batch_size: int = 128) -> tuple[list, list]:
    """Multi-channel inference using MultiChannelPatchDataset."""
    # Filter: all dirs must exist
    missing = [str(d) for d in ch_dirs if not d.exists()]
    if missing:
        return [], []
    ds = MultiChannelPatchDataset(
        channel_dirs=[str(d) for d in ch_dirs],
        condition=0, condition_name=""
    )
    if len(ds) == 0:
        return [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    raws, recons = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0]  # (B, C, H, W)
            if input_divisor != 1.0:
                x = x / input_divisor
            x_in = x.to(device)
            out = model(x_in)
            x_hat = out[0] if isinstance(out, (tuple, list)) else out
            raws.extend(x.cpu().numpy())
            recons.extend(x_hat.cpu().numpy())
    return raws, recons


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(raws: list, recons: list, ds: str, cond: str,
                     in_training: bool) -> list[dict]:
    rows = []
    split_tag = "train" if in_training else "test"
    group = f"{ds}_{cond}"
    for r, p in zip(raws, recons):
        r64 = np.asarray(r, dtype=np.float64)
        p64 = np.asarray(p, dtype=np.float64)
        diff = r64 - p64
        l1   = float(np.mean(np.abs(diff)))
        mse  = float(np.mean(diff ** 2))
        nl1  = l1 / (float(np.mean(np.abs(r64))) + 1e-8)
        row  = dict(dataset=ds, condition=cond, group=group,
                    split=split_tag, recon_l1=l1, recon_mse=mse, recon_nl1=nl1)
        # per-channel for multi-channel
        if r64.ndim == 3:
            for c in range(r64.shape[0]):
                cl1 = float(np.mean(np.abs(r64[c] - p64[c])))
                row[f"recon_l1_ch{c}"]  = cl1
                row[f"recon_nl1_ch{c}"] = cl1 / (float(np.mean(np.abs(r64[c]))) + 1e-8)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

def _make_violin(df: pd.DataFrame, metric: str, out_path: Path, title: str):
    import seaborn as sns

    # Determine group order: train groups first, then test
    train_groups = sorted(df[df["split"] == "train"]["group"].unique())
    test_groups  = sorted(df[df["split"] == "test"]["group"].unique())
    order = train_groups + test_groups

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9 + 2), 5))
    palette = {g: ("#4C72B0" if g in set(train_groups) else "#DD8452") for g in order}
    sns.violinplot(data=df, x="group", y=metric, order=order,
                   palette=palette, cut=0, inner="box", ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=40, labelsize=8)

    # Separator line between train and test
    if train_groups and test_groups:
        sep = len(train_groups) - 0.5
        ax.axvline(sep, color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main eval logic for one run
# ---------------------------------------------------------------------------

def eval_run(run_dir: Path, device: str = "auto") -> bool:
    """Evaluate one protein sweep run. Returns True if successful."""
    eval_dir = run_dir / "eval"
    metrics_csv = eval_dir / "protein_sweep_metrics.csv"

    if metrics_csv.exists():
        print(f"  [skip] already done: {run_dir.name}")
        return True

    cfg = _read_config(run_dir)
    if not cfg:
        print(f"  [skip] no config found: {run_dir.name}")
        return False

    try:
        ch_info = _detect_channel_type(cfg)
    except ValueError as e:
        print(f"  [skip] {e}: {run_dir.name}")
        return False

    try:
        model = _load_model(run_dir, device)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return False

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the input_divisor used at training time
    enlcrop = cfg.get("enlarged_crop", {})
    input_divisor = float(enlcrop.get("input_divisor", 1.0))

    training_conds = _get_training_conditions(cfg)
    all_rows: list[dict] = []

    model_type    = ch_info["model_type"]
    eval_datasets = ch_info["eval_datasets"]
    ch_suffixes   = ch_info["ch_suffixes"]

    for ds in eval_datasets:
        for cond in ALL_CONDITIONS:
            cond_name = f"{ds}_{cond}"
            in_training = cond_name in training_conds

            if model_type == "single":
                sfx = ch_suffixes[0]
                patch_dir = PATCH_BASE / f"{ds}{sfx}" / cond / "tiff_patches32_mr10"
                raws, recons = _infer_single(model, patch_dir, device, input_divisor)
            else:
                ch_dirs = [PATCH_BASE / f"{ds}{sfx}" / cond / "tiff_patches32_mr10"
                           for sfx in ch_suffixes]
                raws, recons = _infer_multi(model, ch_dirs, device, input_divisor)

            if not raws:
                print(f"    [warn] no patches for {cond_name}")
                continue

            rows = _compute_metrics(raws, recons, ds, cond, in_training)
            print(f"    {cond_name:25s} n={len(rows):5d}  "
                  f"nl1={np.mean([r['recon_nl1'] for r in rows]):.4f}")
            all_rows.extend(rows)

    if not all_rows:
        print(f"  [fail] no data for {run_dir.name}")
        return False

    eval_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(str(metrics_csv), index=False)
    print(f"  Saved: {metrics_csv}")

    # Violin plots
    run_name = run_dir.name
    for metric in ["recon_nl1", "recon_l1", "recon_mse"]:
        if metric not in df.columns:
            continue
        out_png = eval_dir / f"violin_{metric}.png"
        _make_violin(df, metric, out_png, f"{run_name}  |  {metric}")
        print(f"  Saved: {out_png}")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate protein-sweep model(s).")
    p.add_argument("run_dir", help="Model run directory (or sweep root with --all)")
    p.add_argument("--all", action="store_true",
                   help="Process all subdirectories under run_dir that contain a model")
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if metrics CSV already exists")
    return p.parse_args()


def main():
    args = _parse_args()
    root = Path(args.run_dir)

    if args.all:
        # Discover all subdirs that contain a trained model
        run_dirs = []
        for d in sorted(root.iterdir()):
            if d.is_dir() and any((d / m).exists()
                                  for m in ("model_best.pt", "model_final.pt")):
                run_dirs.append(d)
        print(f"Found {len(run_dirs)} run dirs under {root}")
    else:
        run_dirs = [root]

    if args.force:
        for rd in run_dirs:
            csv = rd / "eval" / "protein_sweep_metrics.csv"
            if csv.exists():
                csv.unlink()

    ok = total = 0
    for rd in run_dirs:
        print(f"\n[{rd.name}]")
        total += 1
        if eval_run(rd, device=args.device):
            ok += 1

    print(f"\nDone: {ok}/{total} runs completed.")


if __name__ == "__main__":
    main()
