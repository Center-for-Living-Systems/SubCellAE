#!/usr/bin/env python3
"""
setup_le_b2_src_split.py

Domain-split label efficiency experiment — DS1 (vinc ctrl+ycomp), B2 labels.

Three training-label variants, each with budget=50:
  ctrl  : 25 ctrl-adhesion  + 25 ctrl-no-adhesion
  mix   : 12 ctrl-ad + 13 ctrl-noad + 12 ycomp-ad + 13 ycomp-noad  (25+25)
  ycomp : 25 ycomp-adhesion + 25 ycomp-no-adhesion

Evaluation reports per-condition test accuracy (ctrl / ycomp / combined).
5 folds × 5 repeats × 3 variants = 75 jobs.

Run tag : le_b2_src_split
Ann dir : DATA/labelling/le_b2_src_split/
Config  : config/le_b2_src_split/

Usage:
  python scripts/setup_le_b2_src_split.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")

RUN_TAG    = "le_b2_src_split"
N_FOLDS    = 5
N_REPEATS  = 5
VARIANTS   = ["ctrl", "mix", "ycomp"]
N_PER_SIDE = 25   # labels per class for ctrl/ycomp variants (total 50)
N_MIX_SRC  = 25  # labels per source for mix (25 ctrl + 25 ycomp)

FOLD_SPLITS_SRC = DATA_ROOT / "labelling" / "le_b2_supcon" / "fold_splits_ds1.csv"
ANN_DIR  = DATA_ROOT / "labelling" / RUN_TAG
CONFIG_DIR = REPO_ROOT / "config" / RUN_TAG

YAML_TEMPLATE = """\
# =============================================================================
# LE B2 source-split — {label}
# Budget=50: {variant_desc}
# =============================================================================
root_folder : "/net/projects/CLS/lding/data/fa_data_analysis"

data:
  patch_dirs:
    - path           : root_folder + "/ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/vinc/control"
      condition      : 0
      condition_name : "control"
    - path           : root_folder + "/ae_results/patches/cio/vinc/ycomp/tiff_patches32_mr10"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/vinc/ycomp"
      condition      : 1
      condition_name : "ycomp"

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
  latent_dim      : 12
  input_ps        : 32
  no_ch           : 1
  BN_flag         : false
  dropout_flag    : false
  output_sigmoid  : false
  recon_loss_type : "nl1"

  proj_dim              : 8
  noise_prob            : 0.0
  temperature           : 0.5
  lambda_recon          : 1.0
  lambda_contrast       : 0.5
  lambda_supcon         : 5.0
  intensity_scale_range : [0.8, 1.2]

annotation:
  annotation_file : root_folder + "/labelling/{run_tag}/{ann_name}.csv"
  label_col       : "label"
  filename_col    : "unique_ID"
  label_order:
    - "No adhesion"
    - "adhesion"

training:
  epochs                  : 500
  lr                      : 0.001
  batch_size              : 128
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

VARIANT_DESC = {
    "ctrl":  "50 ctrl labels (25 ad + 25 no-ad)",
    "mix":   "25 ctrl + 25 ycomp labels (class-balanced per source)",
    "ycomp": "50 ycomp labels (25 ad + 25 no-ad)",
}


def _sample_balanced(pool: pd.DataFrame, n_per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n_per_class from each label class in pool."""
    parts = []
    for lbl in ["adhesion", "No adhesion"]:
        sub = pool[pool["label"] == lbl]
        n = min(n_per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=int(rng.integers(1e6))))
    return pd.concat(parts, ignore_index=True)


def generate_annotations(fold_splits: pd.DataFrame, dry: bool):
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    fold_splits = fold_splits.copy()
    fold_splits["src"] = fold_splits["unique_ID"].apply(
        lambda x: "ctrl" if x.startswith("control") else "ycomp"
    )

    for fold in range(N_FOLDS):
        pool = fold_splits[fold_splits["fold"] != fold].copy()
        pool_ctrl  = pool[pool["src"] == "ctrl"]
        pool_ycomp = pool[pool["src"] == "ycomp"]

        for rep in range(N_REPEATS):
            rng = np.random.default_rng(seed=fold * 100 + rep)

            for variant in VARIANTS:
                ann_name = f"le_b2_src_{variant}_fv{fold}_r{rep}"
                out_path = ANN_DIR / f"{ann_name}.csv"

                if variant == "ctrl":
                    samp = _sample_balanced(pool_ctrl, N_PER_SIDE, rng)
                elif variant == "ycomp":
                    samp = _sample_balanced(pool_ycomp, N_PER_SIDE, rng)
                else:  # mix: 12 ad + 13 noad per source = 25+25 = 50 total
                    def _sample_src(pool_src, rng):
                        ad   = pool_src[pool_src["label"] == "adhesion"].sample(
                                   n=12, random_state=int(rng.integers(1e6)))
                        noad = pool_src[pool_src["label"] == "No adhesion"].sample(
                                   n=13, random_state=int(rng.integers(1e6)))
                        return pd.concat([ad, noad], ignore_index=True)
                    samp = pd.concat([_sample_src(pool_ctrl, rng),
                                      _sample_src(pool_ycomp, rng)], ignore_index=True)

                result = samp[["unique_ID", "label"]].copy()
                if not dry:
                    result.to_csv(out_path, index=False)

                print(f"  {'[dry] ' if dry else ''}{ann_name}: n={len(result)} "
                      f"(ad={( result['label']=='adhesion').sum()}, "
                      f"noad={(result['label']=='No adhesion').sum()})")


def generate_configs(dry: bool) -> list[str]:
    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    job_list = []
    for fold in range(N_FOLDS):
        for rep in range(N_REPEATS):
            for variant in VARIANTS:
                ann_name = f"le_b2_src_{variant}_fv{fold}_r{rep}"
                name     = ann_name
                cfg_path = CONFIG_DIR / f"{name}.yaml"

                if not dry:
                    cfg_path.write_text(YAML_TEMPLATE.format(
                        label        = f"{variant} fv{fold} r{rep}",
                        variant_desc = VARIANT_DESC[variant],
                        run_tag      = RUN_TAG,
                        name         = name,
                        ann_name     = ann_name,
                    ))
                job_list.append(str(cfg_path.relative_to(REPO_ROOT)))

    job_list_path = CONFIG_DIR / "job_list.txt"
    if not dry:
        job_list_path.write_text("\n".join(job_list) + "\n")

    print(f"\n{'[dry] ' if dry else ''}{RUN_TAG}: {len(job_list)} configs → {CONFIG_DIR}")
    return job_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    fold_splits = pd.read_csv(FOLD_SPLITS_SRC)
    print(f"Loaded fold splits: {len(fold_splits)} patches")

    print("\nGenerating annotation CSVs ...")
    generate_annotations(fold_splits, dry)

    print("\nGenerating YAML configs ...")
    generate_configs(dry)

    # Copy fold_splits to ann dir for eval script
    dst = ANN_DIR / "fold_splits_ds1.csv"
    if not dry and not dst.exists():
        import shutil
        shutil.copy(FOLD_SPLITS_SRC, dst)
        print(f"Copied fold_splits_ds1.csv → {dst}")

    print("\nDone. Submit with:")
    n_jobs = N_FOLDS * N_REPEATS * len(VARIANTS) - 1
    print(f"  sbatch --array=0-{n_jobs} scripts/sbatch_le_b2_src_split.sh")


if __name__ == "__main__":
    main()
