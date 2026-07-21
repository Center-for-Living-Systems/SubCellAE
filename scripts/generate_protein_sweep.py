"""
generate_protein_sweep.py
=========================
Generates training configs and a submission script for the full protein sweep.

Three sets of experiments:
  Set 1: Single-channel — pax(ch1), zyx(ch2), act(ch3), vinc/pfak/ppax(ch0)
          All valid dataset combinations × 3 losses (mse/l1/nl1)
  Set 2: 4-channel — ch0+ch1+ch2+ch3 per dataset group × 3 losses
  Set 3: 3-channel — ch1+ch2+ch3, all 4 datasets × 3 losses

Training style: enlcrop sc2 (58×58→32×32, input_divisor=2.0), ConAE,
                latent_dim=12, proj_dim=8, 500 epochs, cosine LR

Run:
    python scripts/generate_protein_sweep.py [--dry_run]
    # then: bash scripts/submit_protein_sweep.sh
"""

from __future__ import annotations
import argparse
import os
from itertools import combinations
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = "/net/projects/CLS/lding/data/fa_data_analysis"
PATCH_BASE  = f"{ROOT}/ae_results/patches/cio_rb"
FRAME_BASE  = f"{ROOT}/ae_results/source_frames/cio_rb"
RESULT_BASE = f"{ROOT}/ae_results/protein_sweep"
LABEL_BASE  = f"{ROOT}/labelling"

REPO    = "/net/projects/CLS/lding/gitcode/SubCellAE"
PYTHON  = "/home/liyading/miniconda3/bin/python3"
PYTHONPATH = f"{REPO}:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

CONFIG_OUT = Path(REPO) / "config" / "protein_sweep"

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
# Each dataset: name, abbreviation, conditions, ch0 protein name
DATASETS = {
    "vinc":   dict(abbr="v", conditions=["control", "ycomp"], ch0="vinc"),
    "pfak":   dict(abbr="f", conditions=["control", "ycomp"], ch0="pfak"),
    "ppax":   dict(abbr="p", conditions=["control", "ycomp"], ch0="ppax"),
    "nih3t3": dict(abbr="n", conditions=["control", "ycomp"], ch0="vinc"),
}

# ---------------------------------------------------------------------------
# Protein / channel definitions
# ---------------------------------------------------------------------------
# protein_name -> dict with:
#   channel      : frame channel name (used in enlarged_crop.channel)
#   patch_suffix : patch dir suffix; "{ds}" expands to dataset name
#                  "" = use main ds patch dir; "_ch3" = use ds_ch3 patch dir
#   datasets     : which dataset keys are valid for this protein
PROTEINS = {
    "pax":  dict(channel="pax",  patch_suffix="",     datasets=["vinc","pfak","ppax","nih3t3"]),
    "zyx":  dict(channel="zyx",  patch_suffix="_ch2", datasets=["vinc","pfak","ppax","nih3t3"]),
    "act":  dict(channel="act",  patch_suffix="_ch3", datasets=["vinc","pfak","ppax","nih3t3"]),
    "vinc": dict(channel="vinc", patch_suffix="_ch0", datasets=["vinc","nih3t3"]),
    "pfak": dict(channel="pfak", patch_suffix="_ch0", datasets=["pfak"]),
    "ppax": dict(channel="ppax", patch_suffix="_ch0", datasets=["ppax"]),
}

# ---------------------------------------------------------------------------
# Loss / lambda_contrast pairs
# ---------------------------------------------------------------------------
LOSSES = {
    "mse": 0.20,
    "l1":  0.01,
    "nl1": 0.03,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_nonempty_subsets(items: list) -> list[tuple]:
    """Return all non-empty subsets of items."""
    result = []
    for r in range(1, len(items) + 1):
        result.extend(combinations(items, r))
    return result


def ds_abbr(ds_list: list[str]) -> str:
    """Short abbreviation for a list of datasets, e.g. ['vinc','nih3t3'] -> 'vn'."""
    return "".join(DATASETS[d]["abbr"] for d in sorted(ds_list, key=lambda d: DATASETS[d]["abbr"]))


def patch_dir(ds: str, protein: str, condition: str) -> str:
    suffix = PROTEINS[protein]["patch_suffix"]
    return f"{PATCH_BASE}/{ds}{suffix}/{condition}/tiff_patches32_mr10"


def frame_dir(ds: str, condition: str) -> str:
    return f"{FRAME_BASE}/{ds}/{condition}"


def condition_index(cond: str) -> int:
    return {"control": 0, "ycomp": 1}[cond]


def make_single_ch_config(protein: str, ds_list: list[str], loss: str) -> dict:
    """Build a YAML-serializable config dict for single-channel ConAE."""
    channel = PROTEINS[protein]["channel"]
    lc      = LOSSES[loss]
    abbr    = ds_abbr(ds_list)
    run_id  = f"conae_{protein}_{abbr}_{loss}"

    patch_dirs_cfg = []
    cond_offset = 0
    for ds in sorted(ds_list, key=lambda d: DATASETS[d]["abbr"]):
        for cond in DATASETS[ds]["conditions"]:
            patch_dirs_cfg.append({
                "path":           patch_dir(ds, protein, cond),
                "frame_dir":      frame_dir(ds, cond),
                "condition":      cond_offset,
                "condition_name": f"{ds}_{cond}",
            })
            cond_offset += 1

    return {
        "root_folder": ROOT,
        "data":        {"patch_dirs": patch_dirs_cfg},
        "enlarged_crop": {
            "enabled":       True,
            "channel":       channel,
            "context_size":  58,
            "max_shift_px":  4,
            "max_angle_deg": 15.0,
            "pad_size":      64,
            "input_divisor": 2.0,
        },
        "output":  {"result_dir": f"{RESULT_BASE}/{run_id}"},
        "model": {
            "model_type":      "contrastive",
            "latent_dim":      12,
            "input_ps":        32,
            "no_ch":           1,
            "BN_flag":         False,
            "dropout_flag":    False,
            "output_sigmoid":  True,
            "recon_loss_type": loss,
            "proj_dim":              8,
            "noise_prob":            0.05,
            "temperature":           0.5,
            "lambda_recon":          1.0,
            "lambda_contrast":       lc,
            "intensity_scale_range": [0.8, 1.2],
        },
        "training": {
            "epochs":                  500,
            "lr":                      0.001,
            "batch_size":              128,
            "num_workers":             6,
            "val_split":               0.2,
            "loss_norm_flag":          False,
            "group_split":             True,
            "weight_decay":            0.0001,
            "warmup_epochs":           0,
            "lr_scheduler":            "cosine",
            "early_stopping_patience": 0,
            "min_epochs_for_best":     0,
        },
        "reconstruction": {
            "save_recon":       True,
            "recon_pad_size":   64,
            "recon_image_size": 1024,
        },
        "misc": {"device": "auto", "log_level": "INFO"},
    }


def make_4ch_config(ds_list: list[str], loss: str) -> dict:
    """Build config for 4-channel ConAE: ch0+pax+zyx+act.
    Only valid when all datasets share the same ch0 protein (i.e. all vinc-type or single ds).
    """
    lc     = LOSSES[loss]
    abbr   = ds_abbr(ds_list)
    # Determine ch0 names (must all be the same for multi-ds combos)
    ch0_names = list({DATASETS[d]["ch0"] for d in ds_list})
    assert len(ch0_names) == 1, f"4ch combo {ds_list} has mixed ch0 proteins: {ch0_names}"
    ch0_name = ch0_names[0]

    run_id = f"conae_4ch_{ch0_name}_{abbr}_{loss}"
    channels = [ch0_name, "pax", "zyx", "act"]
    # patch suffixes per channel: ch0->_ch0, pax->main(""), zyx->_ch2, act->_ch3
    patch_suffixes = ["_ch0", "", "_ch2", "_ch3"]

    patch_dirs_cfg = []
    cond_offset = 0
    for ds in sorted(ds_list, key=lambda d: DATASETS[d]["abbr"]):
        for cond in DATASETS[ds]["conditions"]:
            ch_dirs = [patch_dir_raw(ds, sfx, cond) for sfx in patch_suffixes]
            patch_dirs_cfg.append({
                "channel_dirs":   ch_dirs,
                "frame_dir":      frame_dir(ds, cond),
                "condition":      cond_offset,
                "condition_name": f"{ds}_{cond}",
            })
            cond_offset += 1

    return {
        "root_folder": ROOT,
        "data":        {"patch_dirs": patch_dirs_cfg},
        "enlarged_crop": {
            "enabled":       True,
            "channel":       channels,
            "context_size":  58,
            "max_shift_px":  4,
            "max_angle_deg": 15.0,
            "pad_size":      64,
            "input_divisor": 2.0,
        },
        "output":  {"result_dir": f"{RESULT_BASE}/{run_id}"},
        "model": {
            "model_type":      "contrastive",
            "latent_dim":      12,
            "input_ps":        32,
            "no_ch":           4,
            "BN_flag":         False,
            "dropout_flag":    False,
            "output_sigmoid":  True,
            "recon_loss_type": loss,
            "proj_dim":              8,
            "noise_prob":            0.05,
            "temperature":           0.5,
            "lambda_recon":          1.0,
            "lambda_contrast":       lc,
            "intensity_scale_range": [0.8, 1.2],
        },
        "training": {
            "epochs":                  500,
            "lr":                      0.001,
            "batch_size":              128,
            "num_workers":             6,
            "val_split":               0.2,
            "loss_norm_flag":          False,
            "group_split":             True,
            "weight_decay":            0.0001,
            "warmup_epochs":           0,
            "lr_scheduler":            "cosine",
            "early_stopping_patience": 0,
            "min_epochs_for_best":     0,
        },
        "reconstruction": {
            "save_recon":       True,
            "recon_pad_size":   64,
            "recon_image_size": 1024,
        },
        "misc": {"device": "auto", "log_level": "INFO"},
    }


def make_3ch_config(ds_list: list[str], loss: str) -> dict:
    """Build config for 3-channel ConAE: pax+zyx+act (ch1+ch2+ch3)."""
    lc     = LOSSES[loss]
    abbr   = ds_abbr(ds_list)
    run_id = f"conae_3ch_pza_{abbr}_{loss}"
    channels       = ["pax", "zyx", "act"]
    patch_suffixes = ["",    "_ch2", "_ch3"]

    patch_dirs_cfg = []
    cond_offset = 0
    for ds in sorted(ds_list, key=lambda d: DATASETS[d]["abbr"]):
        for cond in DATASETS[ds]["conditions"]:
            ch_dirs = [patch_dir_raw(ds, sfx, cond) for sfx in patch_suffixes]
            patch_dirs_cfg.append({
                "channel_dirs":   ch_dirs,
                "frame_dir":      frame_dir(ds, cond),
                "condition":      cond_offset,
                "condition_name": f"{ds}_{cond}",
            })
            cond_offset += 1

    return {
        "root_folder": ROOT,
        "data":        {"patch_dirs": patch_dirs_cfg},
        "enlarged_crop": {
            "enabled":       True,
            "channel":       channels,
            "context_size":  58,
            "max_shift_px":  4,
            "max_angle_deg": 15.0,
            "pad_size":      64,
            "input_divisor": 2.0,
        },
        "output":  {"result_dir": f"{RESULT_BASE}/{run_id}"},
        "model": {
            "model_type":      "contrastive",
            "latent_dim":      12,
            "input_ps":        32,
            "no_ch":           3,
            "BN_flag":         False,
            "dropout_flag":    False,
            "output_sigmoid":  True,
            "recon_loss_type": loss,
            "proj_dim":              8,
            "noise_prob":            0.05,
            "temperature":           0.5,
            "lambda_recon":          1.0,
            "lambda_contrast":       lc,
            "intensity_scale_range": [0.8, 1.2],
        },
        "training": {
            "epochs":                  500,
            "lr":                      0.001,
            "batch_size":              128,
            "num_workers":             6,
            "val_split":               0.2,
            "loss_norm_flag":          False,
            "group_split":             True,
            "weight_decay":            0.0001,
            "warmup_epochs":           0,
            "lr_scheduler":            "cosine",
            "early_stopping_patience": 0,
            "min_epochs_for_best":     0,
        },
        "reconstruction": {
            "save_recon":       True,
            "recon_pad_size":   64,
            "recon_image_size": 1024,
        },
        "misc": {"device": "auto", "log_level": "INFO"},
    }


def patch_dir_raw(ds: str, suffix: str, condition: str) -> str:
    return f"{PATCH_BASE}/{ds}{suffix}/{condition}/tiff_patches32_mr10"


# ---------------------------------------------------------------------------
# Generate all configs
# ---------------------------------------------------------------------------

def generate_all(dry_run: bool = False):
    CONFIG_OUT.mkdir(parents=True, exist_ok=True)
    (CONFIG_OUT / "set1_single_ch").mkdir(exist_ok=True)
    (CONFIG_OUT / "set2_4ch").mkdir(exist_ok=True)
    (CONFIG_OUT / "set3_3ch").mkdir(exist_ok=True)

    all_configs: list[tuple[str, str]] = []  # (config_path, run_id)

    # ── Set 1: single-channel ────────────────────────────────────────────────
    for protein, pdef in PROTEINS.items():
        valid_ds = pdef["datasets"]
        for ds_combo in all_nonempty_subsets(valid_ds):
            ds_list = list(ds_combo)
            for loss in LOSSES:
                abbr   = ds_abbr(ds_list)
                run_id = f"conae_{protein}_{abbr}_{loss}"
                cfg    = make_single_ch_config(protein, ds_list, loss)
                out    = CONFIG_OUT / "set1_single_ch" / f"{run_id}.yaml"
                if not dry_run:
                    with open(out, "w") as f:
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                all_configs.append((str(out), run_id))

    # ── Set 2: 4-channel ─────────────────────────────────────────────────────
    # vinc+nih3t3 share ch0=vinc → can combine
    vinc_like = ["vinc", "nih3t3"]
    for ds_combo in all_nonempty_subsets(vinc_like):
        ds_list = list(ds_combo)
        for loss in LOSSES:
            abbr   = ds_abbr(ds_list)
            run_id = f"conae_4ch_vinc_{abbr}_{loss}"
            cfg    = make_4ch_config(ds_list, loss)
            # override run_id in result_dir to use the explicit name
            cfg["output"]["result_dir"] = f"{RESULT_BASE}/{run_id}"
            out    = CONFIG_OUT / "set2_4ch" / f"{run_id}.yaml"
            if not dry_run:
                with open(out, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            all_configs.append((str(out), run_id))

    # pfak and ppax: single-dataset 4ch
    for ds in ["pfak", "ppax"]:
        for loss in LOSSES:
            abbr   = DATASETS[ds]["abbr"]
            ch0    = DATASETS[ds]["ch0"]
            run_id = f"conae_4ch_{ch0}_{abbr}_{loss}"
            cfg    = make_4ch_config([ds], loss)
            cfg["output"]["result_dir"] = f"{RESULT_BASE}/{run_id}"
            out    = CONFIG_OUT / "set2_4ch" / f"{run_id}.yaml"
            if not dry_run:
                with open(out, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            all_configs.append((str(out), run_id))

    # ── Set 3: 3-channel (pax+zyx+act) ───────────────────────────────────────
    all_ds = ["vinc", "pfak", "ppax", "nih3t3"]
    for ds_combo in all_nonempty_subsets(all_ds):
        ds_list = list(ds_combo)
        for loss in LOSSES:
            abbr   = ds_abbr(ds_list)
            run_id = f"conae_3ch_pza_{abbr}_{loss}"
            cfg    = make_3ch_config(ds_list, loss)
            out    = CONFIG_OUT / "set3_3ch" / f"{run_id}.yaml"
            if not dry_run:
                with open(out, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            all_configs.append((str(out), run_id))

    print(f"Total configs: {len(all_configs)}")
    by_set = {
        "set1": sum(1 for _, r in all_configs if "4ch" not in r and "3ch" not in r),
        "set2": sum(1 for _, r in all_configs if "4ch" in r),
        "set3": sum(1 for _, r in all_configs if "3ch" in r),
    }
    for k, v in by_set.items():
        print(f"  {k}: {v}")

    # ── Write submission script ───────────────────────────────────────────────
    submit_path = Path(REPO) / "scripts" / "submit_protein_sweep.sh"
    if not dry_run:
        _write_submit_script(submit_path, all_configs)
        print(f"\nConfigs written to:    {CONFIG_OUT}/")
        print(f"Submit script written: {submit_path}")
    else:
        print("\n[dry_run] No files written.")

    return all_configs


def _write_submit_script(path: Path, configs: list[tuple[str, str]]):
    lines = [
        "#!/usr/bin/env bash",
        "# submit_protein_sweep.sh — auto-generated by generate_protein_sweep.py",
        "# Submits all protein sweep training jobs to Slurm",
        "#",
        f"# Total jobs: {len(configs)}",
        "set -eo pipefail",
        "mkdir -p logs/slurm",
        "",
        f"PYTHON={PYTHON}",
        f'export PYTHONPATH="{PYTHONPATH}"',
        f"RUNNER={REPO}/scripts/run_ae_from_config.py",
        "",
        f'echo "Submitting {len(configs)} protein sweep jobs..."',
        'echo ""',
        "",
    ]

    for cfg_path, run_id in configs:
        lines += [
            f'JOB=$(sbatch --parsable \\',
            f'    --job-name="{run_id[:40]}" \\',
            f'    --partition=general \\',
            f'    --gres=gpu:a40:1 \\',
            f'    --cpus-per-task=8 \\',
            f'    --mem=32G \\',
            f'    --time=08:00:00 \\',
            f'    --output="logs/slurm/{run_id}_%j.out" \\',
            f'    --wrap="exec 2>&1',
            f"export PYTHONPATH='{PYTHONPATH}'",
            f'echo Node: $(hostname)',
            f'echo GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)',
            f'echo Start: $(date)',
            f'$PYTHON $RUNNER {cfg_path}',
            f'echo End: $(date)")',
            f'echo "  {run_id} -> job $JOB"',
            "",
        ]

    lines += [
        'echo ""',
        'echo "All jobs submitted. Monitor: squeue -u $USER"',
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry_run", action="store_true",
                   help="Count configs and print summary without writing files.")
    args = p.parse_args()
    generate_all(dry_run=args.dry_run)
