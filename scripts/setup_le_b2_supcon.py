#!/usr/bin/env python3
"""
setup_le_b2_supcon.py

Generate annotation CSV subsets, YAML training configs, and job list for the
label-efficiency benchmark (Batch 2 labels, patch-level 5-fold CV).

Design
------
Datasets:
  DS1 (vinc, control + ycomp)  —  1,224 labeled patches / 18 images
  DS2 (pfak, control + ycomp)  —    211 labeled patches /  4 images
  DS3 (ppax, control + ycomp)  —    243 labeled patches /  5 images

5-fold stratified patch CV × variable budgets × 5 repeats
                              + "all" budget × 1 repeat per fold

Label budgets:
  DS1: [10, 20, 25, 50, 75, 100, 150, 200, 300, 500, 750, "all"]
  DS2: [10, 20, 25, 50, 75, 100, 150, "all"]
  DS3: [10, 20, 25, 50, 75, 100, 150, "all"]
Total jobs: DS1: 5×(11×5+1)=280  DS2: 5×(7×5+1)=180  DS3: 5×(7×5+1)=180  total=640

Each fold:
  - 80 % of labeled patches → training pool  (all images included)
  - 20 % of labeled patches → test set       (labels withheld from SupCon + classifier)

For each budget, stratified subsample from training pool (~50 % adhesion).
Subsampling is repeated 5 times (different RNG seeds) per budget per fold.
"All" uses the entire training pool (1 repeat only, no subsampling noise).

Outputs per job:
  labelling/le_b2_supcon/{name}.csv      — annotation CSV (train labels only)
  config/le_b2_supcon/{name}.yaml        — SupCon-AE training config

Also writes per-dataset fold-split metadata:
  labelling/le_b2_supcon/fold_splits_ds1.csv
  labelling/le_b2_supcon/fold_splits_ds2.csv
  labelling/le_b2_supcon/fold_splits_ds3.csv

Usage
-----
  python scripts/setup_le_b2_supcon.py [--dry-run] [--dataset ds1|ds2|ds3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR  = DATA_ROOT / "labelling"
CONFIG_DIR = REPO_ROOT / "config" / "le_b2_supcon"
OUT_ANN    = LABEL_DIR / "le_b2_supcon"
RUN_ROOT   = DATA_ROOT / "ae_results" / "contrastive_run" / "le_b2_supcon"

ANN_FILES = {
    "ds1": LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv",
    "ds2": LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv",
    "ds3": LABEL_DIR / "ppax_control_label_Ernest_20260825_1433.csv",
}

# patch_dirs entries per dataset (relative to DATA_ROOT)
PATCH_DIRS = {
    "ds1": [
        dict(path="ae_results/patches/cio/vinc/control/tiff_patches32_mr10",
             frame_dir="ae_results/source_frames/cio_mode_prt/vinc/control",
             condition=0, condition_name="control"),
        dict(path="ae_results/patches/cio/vinc/ycomp/tiff_patches32_mr10",
             frame_dir="ae_results/source_frames/cio_mode_prt/vinc/ycomp",
             condition=1, condition_name="ycomp"),
    ],
    "ds2": [
        dict(path="ae_results/patches/cio/pfak/control/tiff_patches32_mr10",
             frame_dir="ae_results/source_frames/cio_mode_prt/pfak/control",
             condition=0, condition_name="control"),
        dict(path="ae_results/patches/cio/pfak/ycomp/tiff_patches32_mr10",
             frame_dir="ae_results/source_frames/cio_mode_prt/pfak/ycomp",
             condition=1, condition_name="ycomp"),
    ],
    "ds3": [
        dict(path="ae_results/patches/cio/ppax/control/tiff_patches32_mr10",
             frame_dir="ae_results/source_frames/cio_mode_prt/ppax/control",
             condition=0, condition_name="control"),
        dict(path="ae_results/patches/cio/ppax/ycomp/tiff_patches32_mr10",
             frame_dir="ae_results/source_frames/cio_mode_prt/ppax/ycomp",
             condition=1, condition_name="ycomp"),
    ],
}

N_FOLDS   = 5
BUDGETS = {
    "ds1": [10, 20, 25, 50, 75, 100, 150, 200, 300, 500, 750],
    "ds2": [10, 20, 25, 50, 75, 100, 150],
    "ds3": [10, 20, 25, 50, 75, 100, 150],
}
N_REPEATS = 5
CV_SEED   = 42


# ---------------------------------------------------------------------------
# Helpers

def _binarize(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def _underscore_to_hyphen(filename: str) -> str:
    """Convert `control_f0000x...tif` to `control-f0000x...tif` (pipeline key format)."""
    return filename.replace("_f", "-f", 1)


def stratified_subsample(
    df: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Return ≤ n rows: ceil(n/2) adhesion + floor(n/2) No adhesion."""
    adh  = df[df["label"] == "adhesion"]
    noad = df[df["label"] == "No adhesion"]
    n_a  = min(n - n // 2, len(adh))
    n_b  = min(n // 2,     len(noad))
    parts = []
    if n_a > 0:
        parts.append(adh.iloc[rng.choice(len(adh), n_a, replace=False)])
    if n_b > 0:
        parts.append(noad.iloc[rng.choice(len(noad), n_b, replace=False)])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def _patch_dirs_block(ds: str) -> str:
    lines = []
    for d in PATCH_DIRS[ds]:
        lines.append(
            f'    - path           : root_folder + "/{d["path"]}"\n'
            f'      frame_dir      : root_folder + "/{d["frame_dir"]}"\n'
            f'      condition      : {d["condition"]}\n'
            f'      condition_name : "{d["condition_name"]}"'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# YAML template

YAML_TEMPLATE = """\
# =============================================================================
# LE B2 SupCon benchmark — {label}
# =============================================================================
root_folder : "/net/projects/CLS/lding/data/fa_data_analysis"

data:
  patch_dirs:
{patch_dirs_block}

enlarged_crop:
  enabled       : true
  channel       : "pax"
  context_size  : 58
  max_shift_px  : 4
  max_angle_deg : 15.0
  pad_size      : 64
  input_divisor : 2.0

output:
  result_dir : root_folder + "/ae_results/contrastive_run/{run_tag}/{name}"

model:
  model_type      : "supcon"
  latent_dim      : 32
  input_ps        : 32
  no_ch           : 1
  BN_flag         : false
  dropout_flag    : false
  output_sigmoid  : false
  recon_loss_type : "nl1"

  proj_dim              : {proj_dim}
  noise_prob            : 0.0
  temperature           : 0.5
  lambda_recon          : 1.0
  lambda_contrast       : 0.5
  lambda_supcon         : 5.0
  intensity_scale_range : [0.8, 1.2]

annotation:
  annotation_file : root_folder + "/labelling/le_b2_supcon/{name}.csv"
  label_col       : "label"
  filename_col    : "unique_ID"
  label_order:
    - "No adhesion"
    - "adhesion"

training:
  epochs                  : 500
  lr                      : 0.001
  batch_size              : 128
  n_labeled_per_class     : 2
  num_workers             : 6
  val_split               : 0.0
  group_split             : false
  loss_norm_flag          : false
  weight_decay            : 0.0001
  warmup_epochs           : 0
  lr_scheduler            : "none"
  early_stopping_patience : 0
  min_epochs_for_best     : 501

reconstruction:
  save_recon : false

misc:
  device    : "auto"
  log_level : "INFO"
"""


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be created without writing files")
    ap.add_argument("--dataset", choices=list(ANN_FILES.keys()),
                    help="Generate configs for one dataset only (default: all)")
    ap.add_argument("--proj-dim", type=int, default=8,
                    help="Projection head dim (default 8; use 16 for high-dim variant)")
    args = ap.parse_args()
    dry = args.dry_run

    proj_dim = args.proj_dim
    run_tag  = "le_b2_supcon" if proj_dim == 8 else f"le_b2_supcon_pd{proj_dim}"

    global CONFIG_DIR, RUN_ROOT
    CONFIG_DIR = REPO_ROOT / "config" / run_tag
    RUN_ROOT   = DATA_ROOT / "ae_results" / "contrastive_run" / run_tag

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)

    datasets = {args.dataset: ANN_FILES[args.dataset]} if args.dataset else ANN_FILES

    job_list    = []
    rng_master  = np.random.default_rng(0)
    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    for ds, ann_file in datasets.items():
        df = pd.read_csv(ann_file)
        df["label"]     = df["label"].apply(_binarize)
        df["unique_ID"] = df["filename"].apply(_underscore_to_hyphen)
        df = df.reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  total={len(df)}  "
              f"adh={(df['label']=='adhesion').sum()}  "
              f"noad={(df['label']=='No adhesion').sum()}")

        # Save fold-split metadata for eval script
        fold_labels = np.empty(len(df), dtype=int)
        for fold, (train_idx, test_idx) in enumerate(
                skf.split(np.arange(len(df)), df["label"].values)):
            fold_labels[test_idx] = fold

        splits_df = df[["unique_ID", "label"]].copy()
        splits_df["fold"] = fold_labels
        if not dry:
            splits_path = OUT_ANN / f"fold_splits_{ds}.csv"
            splits_df.to_csv(splits_path, index=False)
            print(f"  fold splits → {splits_path}")

        # Generate jobs per fold
        for fold, (train_idx, test_idx) in enumerate(
                skf.split(np.arange(len(df)), df["label"].values)):
            train_pool = df.iloc[train_idx].copy().reset_index(drop=True)
            print(f"\n  Fold {fold}: train={len(train_pool)}  test={len(test_idx)}")

            # Variable budgets × N_REPEATS
            for budget in BUDGETS[ds]:
                for rep in range(N_REPEATS):
                    rng  = np.random.default_rng(rng_master.integers(1 << 31))
                    name = f"le_b2_{ds}_fv{fold}_nb{budget}_r{rep}"

                    if len(train_pool) <= budget:
                        subset = train_pool.copy()
                    else:
                        subset = stratified_subsample(train_pool, budget, rng)

                    _write_job(name, ds, fold, budget, rep, subset, dry, job_list,
                               proj_dim=proj_dim, run_tag=run_tag)

            # "all" budget — 1 repeat, full training pool
            name = f"le_b2_{ds}_fv{fold}_nball_r0"
            _write_job(name, ds, fold, "all", 0, train_pool.copy(), dry, job_list,
                       proj_dim=proj_dim, run_tag=run_tag)

    # Write job list
    suffix = f"_{args.dataset}" if args.dataset else ""
    job_list_path = CONFIG_DIR / f"job_list{suffix}.txt"
    if not dry:
        with open(job_list_path, "w") as fh:
            fh.write("\n".join(job_list) + "\n")
        print(f"\nJob list ({len(job_list)} entries) → {job_list_path}")
    else:
        print(f"\n[dry-run] Would write {len(job_list)} jobs to {job_list_path}")

    print(f"Total training jobs: {len(job_list)}")


def _write_job(name, ds, fold, budget, rep, subset, dry, job_list,
               proj_dim=8, run_tag="le_b2_supcon"):
    n_labels = len(subset)
    n_adh    = (subset["label"] == "adhesion").sum()
    n_noad   = (subset["label"] == "No adhesion").sum()
    label    = (f"{ds}  fold={fold}  budget={budget}  repeat={rep}  "
                f"n_labels={n_labels} (adh={n_adh} noad={n_noad})")
    print(f"    {name}: n={n_labels}  adh={n_adh}  noad={n_noad}")

    ann_path    = OUT_ANN    / f"{name}.csv"
    config_path = CONFIG_DIR / f"{name}.yaml"

    if not dry:
        # Only write annotation CSV for the base (pd8) run; reuse for variants
        if proj_dim == 8:
            out_df = subset[["unique_ID", "label"]].copy()
            out_df.to_csv(ann_path, index=False)

        yaml_text = YAML_TEMPLATE.format(
            label=label,
            name=name,
            patch_dirs_block=_patch_dirs_block(ds),
            proj_dim=proj_dim,
            run_tag=run_tag,
        )
        config_path.write_text(yaml_text)

    job_list.append(str(config_path.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
