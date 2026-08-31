#!/usr/bin/env python3
"""
setup_le_b2_vinc_ctrl.py

Generate annotation CSVs, YAML configs, and job list for the
label-efficiency benchmark using ONLY vinc/control patches from B2 labels.

Mirrors setup_le_b2_supcon.py but restricts DS1 to control condition only.
Used to compare against old s1v3/s2v2 results (which also used ctrl-only).

Dataset: vinc/control only — 539 patches (197 adhesion, 342 no-adhesion)
Budgets: [10, 20, 25, 50, 75, 100, 150, "all"]  (same as DS2/DS3 scale)
5-fold stratified CV × 5 repeats per budget + "all" × 1 repeat
Total jobs: 5 × (7×5 + 1) = 180

Outputs:
  labelling/le_b2_vinc_ctrl/fold_splits.csv
  labelling/le_b2_vinc_ctrl/{name}.csv       (annotation subsets)
  config/le_b2_vinc_ctrl/{name}.yaml
  config/le_b2_vinc_ctrl/job_list.txt

Usage:
  python scripts/setup_le_b2_vinc_ctrl.py [--dry-run]
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
RUN_TAG    = "le_b2_vinc_ctrl"
CONFIG_DIR = REPO_ROOT / "config" / RUN_TAG
OUT_ANN    = LABEL_DIR / RUN_TAG

ANN_FILE   = LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv"
PATCH_DIR  = "ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
FRAME_DIR  = "ae_results/source_frames/cio_mode_prt/vinc/control"

BUDGETS   = [10, 20, 25, 50, 75, 100, 150]
N_FOLDS   = 5
N_REPEATS = 5
CV_SEED   = 42


def _binarize(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def _u2h(filename: str) -> str:
    return filename.replace("_f", "-f", 1)


def stratified_subsample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    adh  = df[df["label"] == "adhesion"]
    noad = df[df["label"] == "No adhesion"]
    n_a  = min(n - n // 2, len(adh))
    n_b  = min(n // 2,     len(noad))
    parts = []
    if n_a > 0: parts.append(adh.iloc[rng.choice(len(adh),  n_a, replace=False)])
    if n_b > 0: parts.append(noad.iloc[rng.choice(len(noad), n_b, replace=False)])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


YAML_TEMPLATE = """\
# =============================================================================
# LE B2 vinc/ctrl only — {label}
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
  latent_dim      : 32
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
  annotation_file : root_folder + "/labelling/{run_tag}/{name}.csv"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry  = args.dry_run

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)

    # Load B2 labels, filter to ctrl only, binarize
    df = pd.read_csv(ANN_FILE)
    df = df[df["filename"].str.startswith("control_")].copy()
    df["label"]     = df["label"].apply(_binarize)
    df["unique_ID"] = df["filename"].apply(_u2h)
    df = df.reset_index(drop=True)

    print(f"vinc/ctrl B2 patches: {len(df)}  "
          f"adh={(df['label']=='adhesion').sum()}  "
          f"noad={(df['label']=='No adhesion').sum()}")

    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    rng_master = np.random.default_rng(0)

    # Save fold-split metadata
    fold_labels = np.empty(len(df), dtype=int)
    for fold, (_, test_idx) in enumerate(skf.split(np.arange(len(df)), df["label"].values)):
        fold_labels[test_idx] = fold
    splits_df = df[["unique_ID", "label"]].copy()
    splits_df["fold"] = fold_labels
    if not dry:
        splits_df.to_csv(OUT_ANN / "fold_splits.csv", index=False)
        print(f"Fold splits → {OUT_ANN}/fold_splits.csv")

    job_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(df)), df["label"].values)):
        train_pool = df.iloc[train_idx].copy().reset_index(drop=True)
        print(f"\nFold {fold}: train={len(train_pool)}  test={len(test_idx)}")

        for budget in BUDGETS:
            for rep in range(N_REPEATS):
                rng  = np.random.default_rng(rng_master.integers(1 << 31))
                name = f"le_b2_vinc_ctrl_fv{fold}_nb{budget}_r{rep}"

                subset = train_pool if len(train_pool) <= budget \
                         else stratified_subsample(train_pool, budget, rng)

                _write_job(name, subset, dry, job_list,
                           f"fv{fold} nb{budget} r{rep} n={len(subset)}")

        # "all" budget
        name = f"le_b2_vinc_ctrl_fv{fold}_nball_r0"
        _write_job(name, train_pool.copy(), dry, job_list, f"fv{fold} nb=all")

    job_list_path = CONFIG_DIR / "job_list.txt"
    if not dry:
        job_list_path.write_text("\n".join(job_list) + "\n")
        print(f"\nJob list ({len(job_list)} jobs) → {job_list_path}")
    else:
        print(f"\n[dry-run] {len(job_list)} jobs")


def _write_job(name, subset, dry, job_list, label):
    ann_path    = OUT_ANN    / f"{name}.csv"
    config_path = CONFIG_DIR / f"{name}.yaml"

    if not dry:
        subset[["unique_ID", "label"]].to_csv(ann_path, index=False)
        config_path.write_text(YAML_TEMPLATE.format(
            label=label, name=name,
            patch_dir=PATCH_DIR, frame_dir=FRAME_DIR,
            run_tag=RUN_TAG,
        ))

    job_list.append(str(config_path.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
