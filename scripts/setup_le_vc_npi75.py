#!/usr/bin/env python3
"""
setup_le_vc_npi75.py

Generate YAML configs for a quick npi=75 pilot comparing two architectures
on vinc/ctrl-only B2 labels, WITHOUT n_labeled_per_class (all 75 labels go
to the SupCon loss directly, matching the le_combined protocol).

Reuses annotation CSVs from le_b2_vinc_ctrl (same folds, nb75 subsets).

Variants:
  le_vc_lat12p8   — latent_dim=12, proj_dim=8
  le_vc_lat32p16  — latent_dim=32, proj_dim=16

Jobs: 5 folds × 5 repeats = 25 per variant, 50 total.

Usage:
  python scripts/setup_le_vc_npi75.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")

# Reuse annotation files from le_b2_vinc_ctrl (nb75 subsets)
SRC_ANN_TAG = "le_b2_vinc_ctrl"
NPI         = 75
N_FOLDS     = 5
N_REPEATS   = 5

PATCH_DIR = "ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
FRAME_DIR = "ae_results/source_frames/cio_mode_prt/vinc/control"

VARIANTS = [
    dict(run_tag="le_vc_lat12p8",  latent_dim=12, proj_dim=8),
    dict(run_tag="le_vc_lat32p16", latent_dim=32, proj_dim=16),
]

YAML_TEMPLATE = """\
# =============================================================================
# LE vc {variant} — fv{fold} nb{npi} r{rep}
# All {npi} labels go to SupCon loss (no n_labeled_per_class cap).
# =============================================================================
root_folder : "/net/projects/CLS/lding/data/fa_data_analysis"

data:
  patch_dirs:
    - path           : root_folder + "/{patch_dir}"
      frame_dir      : root_folder + "/{frame_dir}"
      condition      : 0
      condition_name : "control"

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

    for var in VARIANTS:
        run_tag   = var["run_tag"]
        cfg_dir   = REPO_ROOT / "config" / run_tag
        if not dry:
            cfg_dir.mkdir(parents=True, exist_ok=True)

        job_list = []
        for fold in range(N_FOLDS):
            for rep in range(N_REPEATS):
                name     = f"{run_tag}_fv{fold}_nb{NPI}_r{rep}"
                ann_name = f"{SRC_ANN_TAG}_fv{fold}_nb{NPI}_r{rep}"

                # verify annotation file exists
                ann_path = DATA_ROOT / "labelling" / SRC_ANN_TAG / f"{ann_name}.csv"
                if not ann_path.exists():
                    print(f"  WARNING: annotation not found: {ann_path}")

                yaml_text = YAML_TEMPLATE.format(
                    variant    = run_tag,
                    fold       = fold,
                    npi        = NPI,
                    rep        = rep,
                    name       = name,
                    run_tag    = run_tag,
                    src_ann_tag= SRC_ANN_TAG,
                    ann_name   = ann_name,
                    patch_dir  = PATCH_DIR,
                    frame_dir  = FRAME_DIR,
                    latent_dim = var["latent_dim"],
                    proj_dim   = var["proj_dim"],
                )

                cfg_path = cfg_dir / f"{name}.yaml"
                if not dry:
                    cfg_path.write_text(yaml_text)
                job_list.append(str(cfg_path.relative_to(REPO_ROOT)))

        job_list_path = cfg_dir / "job_list.txt"
        if not dry:
            job_list_path.write_text("\n".join(job_list) + "\n")
        print(f"{'[dry] ' if dry else ''}{run_tag}: {len(job_list)} jobs → {job_list_path}")

    print("Done.")


if __name__ == "__main__":
    main()
