#!/usr/bin/env python3
"""
setup_le_b12_ds2_lat12p8.py

Generate YAML configs for DS2 B12 label-efficiency benchmark with:
  - lat=12, proj=8
  - No n_labeled_per_class (bug-fixed version of le_b12_supcon)

Reuses annotation CSVs from le_b12_supcon (same folds, same budget subsets).
DS2 only (pfak ctrl+ycomp).

Budgets: 10, 20, 25, 50, 75, 100, 150, all
Jobs: 5 folds × (7 budgets × 5 repeats + 1 all) = 180

Usage:
  python scripts/setup_le_b12_ds2_lat12p8.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[1]
DATA_ROOT    = Path("/net/projects/CLS/lding/data/fa_data_analysis")

RUN_TAG      = "le_b12_ds2_lat12p8"
SRC_ANN_TAG  = "le_b12_supcon"
DS           = "ds2"
N_FOLDS      = 5
N_REPEATS    = 5
BUDGETS      = [10, 20, 25, 50, 75, 100, 150]

CONFIG_DIR   = REPO_ROOT / "config" / RUN_TAG
ANN_DIR      = DATA_ROOT / "labelling" / SRC_ANN_TAG

YAML_TEMPLATE = """\
# =============================================================================
# LE B12 DS2 lat12/proj8 — {label}
# All budget labels go to SupCon loss (no n_labeled_per_class cap).
# =============================================================================
root_folder : "/net/projects/CLS/lding/data/fa_data_analysis"

data:
  patch_dirs:
    - path           : root_folder + "/ae_results/patches/cio/pfak/control/tiff_patches32_mr10"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/pfak/control"
      condition      : 0
      condition_name : "control"
    - path           : root_folder + "/ae_results/patches/cio/pfak/ycomp/tiff_patches32_mr10"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/pfak/ycomp"
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
  annotation_file : root_folder + "/labelling/{src_ann_tag}/{ann_name}.csv"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    job_list = []
    missing  = []

    for fold in range(N_FOLDS):
        for budget in BUDGETS:
            for rep in range(N_REPEATS):
                ann_name = f"le_b12_{DS}_fv{fold}_nb{budget}_r{rep}"
                ann_path = ANN_DIR / f"{ann_name}.csv"
                if not ann_path.exists():
                    missing.append(str(ann_path))
                    continue
                name = f"le_b12_{DS}_lat12p8_fv{fold}_nb{budget}_r{rep}"
                _write(name, ann_name, fold, budget, rep, dry, job_list)

        # "all" budget — 1 repeat
        ann_name = f"le_b12_{DS}_fv{fold}_nball_r0"
        ann_path = ANN_DIR / f"{ann_name}.csv"
        if ann_path.exists():
            name = f"le_b12_{DS}_lat12p8_fv{fold}_nball_r0"
            _write(name, ann_name, fold, "all", 0, dry, job_list)
        else:
            missing.append(str(ann_path))

    job_list_path = CONFIG_DIR / "job_list.txt"
    if not dry:
        job_list_path.write_text("\n".join(job_list) + "\n")

    print(f"{'[dry] ' if dry else ''}{RUN_TAG}: {len(job_list)} jobs → {job_list_path}")
    if missing:
        print(f"  WARNING: {len(missing)} annotation files not found")
        for m in missing[:5]:
            print(f"    {m}")


def _write(name, ann_name, fold, budget, rep, dry, job_list):
    cfg_path = CONFIG_DIR / f"{name}.yaml"
    if not dry:
        cfg_path.write_text(YAML_TEMPLATE.format(
            label       = f"fv{fold} nb{budget} r{rep}",
            name        = name,
            run_tag     = RUN_TAG,
            src_ann_tag = SRC_ANN_TAG,
            ann_name    = ann_name,
        ))
    job_list.append(str(cfg_path.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
