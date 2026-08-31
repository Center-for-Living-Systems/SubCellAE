#!/usr/bin/env python3
"""
setup_le_b2_ds1c.py

DS1 B2 ctrl-only label-efficiency benchmark (fixed).
  - Train and test on vinc-control patches only
  - Fold splits derived from fold_splits_ds1.csv ctrl subset
  - lat=12, proj=8, no n_labeled_per_class
  - Budgets: 10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 750 + all
  - 5 folds × (11 budgets × 5 repeats + 1 all) = 276 jobs

Usage:
  python scripts/setup_le_b2_ds1c.py [--dry-run]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")

RUN_TAG  = "le_b2_ds1c"
N_FOLDS  = 5
N_REPEATS = 5
BUDGETS  = [10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 750]

ANN_DIR    = DATA_ROOT / "labelling" / RUN_TAG
CONFIG_DIR = REPO_ROOT / "config" / RUN_TAG
FS_PATH    = ANN_DIR / "fold_splits_ds1c.csv"

YAML_TEMPLATE = """\
# =============================================================================
# LE B2 DS1 ctrl-only lat12/proj8 — {label}
# Train and test on vinc-control patches only.
# =============================================================================
root_folder : "/net/projects/CLS/lding/data/fa_data_analysis"

data:
  patch_dirs:
    - path           : root_folder + "/ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/vinc/control"
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


def sample_ann(pool: pd.DataFrame, budget: int, rng: random.Random) -> pd.DataFrame:
    """Class-balanced sample of `budget` patches from pool."""
    classes = pool["label"].unique().tolist()
    per_class = budget // len(classes)
    remainder = budget % len(classes)
    rows = []
    for i, cls in enumerate(classes):
        sub = pool[pool["label"] == cls]
        n = per_class + (1 if i < remainder else 0)
        n = min(n, len(sub))
        rows.append(sub.sample(n, random_state=rng.randint(0, 2**31)))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    fs = pd.read_csv(FS_PATH)

    if not dry:
        ANN_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    job_list = []

    for fold in range(N_FOLDS):
        test_ids  = set(fs[fs["fold"] == fold]["unique_ID"])
        train_pool = fs[~fs["unique_ID"].isin(test_ids)].copy()

        # budgeted repeats
        for budget in BUDGETS:
            for rep in range(N_REPEATS):
                rng = random.Random(fold * 10000 + budget * 100 + rep)
                ann = sample_ann(train_pool, budget, rng)
                if len(ann) < 2:
                    continue
                ann_name = f"le_b2_ds1c_fv{fold}_nb{budget}_r{rep}"
                name     = ann_name
                if not dry:
                    ann[["unique_ID", "label"]].to_csv(
                        ANN_DIR / f"{ann_name}.csv", index=False)
                    (CONFIG_DIR / f"{name}.yaml").write_text(
                        YAML_TEMPLATE.format(
                            label=f"fv{fold} nb{budget} r{rep}",
                            name=name, run_tag=RUN_TAG, ann_name=ann_name))
                job_list.append(str((CONFIG_DIR / f"{name}.yaml").relative_to(REPO_ROOT)))

        # "all" budget — 1 repeat
        ann_name = f"le_b2_ds1c_fv{fold}_nball_r0"
        name     = ann_name
        if not dry:
            train_pool[["unique_ID", "label"]].to_csv(
                ANN_DIR / f"{ann_name}.csv", index=False)
            (CONFIG_DIR / f"{name}.yaml").write_text(
                YAML_TEMPLATE.format(
                    label=f"fv{fold} nball r0",
                    name=name, run_tag=RUN_TAG, ann_name=ann_name))
        job_list.append(str((CONFIG_DIR / f"{name}.yaml").relative_to(REPO_ROOT)))

    job_list_path = CONFIG_DIR / "job_list.txt"
    if not dry:
        job_list_path.write_text("\n".join(job_list) + "\n")

    print(f"{'[dry] ' if dry else ''}{RUN_TAG}: {len(job_list)} jobs → {job_list_path}")


if __name__ == "__main__":
    main()
