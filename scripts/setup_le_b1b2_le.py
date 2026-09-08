#!/usr/bin/env python3
"""
setup_le_b1b2_le.py

Generate label-efficiency annotation CSVs and YAML configs for the
B1/B2 matched fair-comparison experiment at multiple budgets.

Design
------
- Reuses fold_splits_b1.csv / fold_splits_b2.csv from labelling/le_b1b2_matched/
- Budgets: 10, 20, 25, 50, 75, 100, 150, 200, all
- 5 folds × 5 repeats × 8 numeric budgets + 1 all = 205 jobs per (batch × latent)
- 4 config dirs: le_b1b2_b1_lat12p8, le_b1b2_b2_lat12p8,
                  le_b1b2_b1_lat64p32, le_b1b2_b2_lat64p32
- Annotation CSVs shared: labelling/le_b1b2_le/

Usage
-----
  python scripts/setup_le_b1b2_le.py [--dry-run]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")

SRC_ANN_DIR = DATA_ROOT / "labelling" / "le_b1b2_matched"
OUT_ANN_DIR = DATA_ROOT / "labelling" / "le_b1b2_le"

N_FOLDS  = 5
N_REPEATS = 5
BUDGETS  = [10, 20, 25, 50, 75, 100, 150, 200]
BATCHES  = ["b1", "b2"]
LATENTS  = [
    ("lat12p8", 12, 8),
    ("lat64p32", 64, 32),
]

YAML_TEMPLATE = """\
# =============================================================================
# LE B1B2 {batch_upper} {lat_tag} — {label}
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
  latent_dim      : {latent_dim}
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
  annotation_file : root_folder + "/labelling/le_b1b2_le/{ann_name}.csv"
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


def _subsample(train_df: pd.DataFrame, budget: int, rng: np.random.Generator) -> pd.DataFrame:
    adh   = train_df[train_df["label"] == "adhesion"]
    noad  = train_df[train_df["label"] == "No adhesion"]
    n_adh  = min(math.ceil(budget / 2), len(adh))
    n_noad = min(budget - n_adh, len(noad))
    sel = pd.concat([
        adh.sample(n=n_adh,  random_state=int(rng.integers(1e6))),
        noad.sample(n=n_noad, random_state=int(rng.integers(1e6))),
    ])
    return sel.sample(frac=1, random_state=int(rng.integers(1e6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    if not dry:
        OUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

    # Load fold splits for both batches
    fold_splits = {}
    for batch in BATCHES:
        fs = pd.read_csv(SRC_ANN_DIR / f"fold_splits_{batch}.csv")
        fold_splits[batch] = fs

    # --- Generate annotation CSVs (shared across latent configs) ---
    ann_counts = {"b1": 0, "b2": 0}
    for batch in BATCHES:
        fs = fold_splits[batch]
        rng = np.random.default_rng(42)

        for fold in range(N_FOLDS):
            train_df = fs[fs["fold"] != fold].copy()

            # Budget-subsampled
            for budget in BUDGETS:
                for rep in range(N_REPEATS):
                    sub = _subsample(train_df, budget, rng)
                    ann_name = f"le_b1b2_{batch}_fv{fold}_nb{budget}_r{rep}"
                    if not dry:
                        sub[["unique_ID", "label"]].to_csv(
                            OUT_ANN_DIR / f"{ann_name}.csv", index=False)
                    ann_counts[batch] += 1

            # All labels
            ann_name = f"le_b1b2_{batch}_fv{fold}_nball_r0"
            if not dry:
                train_df[["unique_ID", "label"]].to_csv(
                    OUT_ANN_DIR / f"{ann_name}.csv", index=False)
            ann_counts[batch] += 1

    print(f"Annotation CSVs: b1={ann_counts['b1']}, b2={ann_counts['b2']} "
          f"({'dry' if dry else 'written'} → {OUT_ANN_DIR})")

    # --- Generate YAML configs per (batch × latent) ---
    for batch in BATCHES:
        for lat_tag, latent_dim, proj_dim in LATENTS:
            run_tag  = f"le_b1b2_{batch}_{lat_tag}"
            cfg_dir  = REPO_ROOT / "config" / run_tag
            if not dry:
                cfg_dir.mkdir(parents=True, exist_ok=True)

            job_list = []
            for fold in range(N_FOLDS):
                for budget in BUDGETS:
                    for rep in range(N_REPEATS):
                        ann_name = f"le_b1b2_{batch}_fv{fold}_nb{budget}_r{rep}"
                        name     = f"le_b1b2_{batch}_{lat_tag}_fv{fold}_nb{budget}_r{rep}"
                        cfg_path = cfg_dir / f"{name}.yaml"
                        if not dry:
                            cfg_path.write_text(YAML_TEMPLATE.format(
                                batch_upper = batch.upper(),
                                lat_tag     = lat_tag,
                                label       = f"fv{fold} nb{budget} r{rep}",
                                run_tag     = run_tag,
                                name        = name,
                                ann_name    = ann_name,
                                latent_dim  = latent_dim,
                                proj_dim    = proj_dim,
                            ))
                        job_list.append(str(cfg_path.relative_to(REPO_ROOT)))

                # all budget
                ann_name = f"le_b1b2_{batch}_fv{fold}_nball_r0"
                name     = f"le_b1b2_{batch}_{lat_tag}_fv{fold}_nball_r0"
                cfg_path = cfg_dir / f"{name}.yaml"
                if not dry:
                    cfg_path.write_text(YAML_TEMPLATE.format(
                        batch_upper = batch.upper(),
                        lat_tag     = lat_tag,
                        label       = f"fv{fold} nb=all r0",
                        run_tag     = run_tag,
                        name        = name,
                        ann_name    = ann_name,
                        latent_dim  = latent_dim,
                        proj_dim    = proj_dim,
                    ))
                job_list.append(str(cfg_path.relative_to(REPO_ROOT)))

            jl_path = cfg_dir / "job_list.txt"
            if not dry:
                jl_path.write_text("\n".join(job_list) + "\n")
            print(f"{'[dry] ' if dry else ''}{run_tag}: {len(job_list)} jobs → {jl_path}")


if __name__ == "__main__":
    main()
