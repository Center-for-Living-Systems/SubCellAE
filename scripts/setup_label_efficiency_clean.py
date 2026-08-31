#!/usr/bin/env python3
"""
setup_label_efficiency_clean.py

Generate annotation CSV subsets, YAML training configs, and job list for the
clean label-efficiency experiment.

Design
------
3 frame splits × 5 n_per_img (10,25,50,75,100) × 3 repeats
                + 3 × 1 (n_per_img="all", deterministic) = 48 jobs

For each job:
  - Subsample K labels (stratified per train frame) from Annabel's 539-label CSV
  - Save annotation CSV  →  labelling/le_clean/le_c{c}_npi{npi}_r{r}.csv
  - Save YAML config     →  config/label_efficiency/le_c{c}_npi{npi}_r{r}.yaml
  - Add config path to   →  config/label_efficiency/job_list.txt

SupCon and classifier see exactly the same K labels; test-frame labels
never enter the SupCon loss.

Usage
-----
  python scripts/setup_label_efficiency_clean.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR  = DATA_ROOT / "labelling"
ANN_FILE   = LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"
CONFIG_DIR = REPO_ROOT / "config" / "label_efficiency"
OUT_ANN    = LABEL_DIR / "le_clean"
RUN_ROOT   = DATA_ROOT / "ae_results" / "contrastive_run" / "le_clean"

CONFIGS = [
    dict(cfg_id=0, train_frames=[0],    test_frames=[1, 2, 3]),
    dict(cfg_id=1, train_frames=[0, 1], test_frames=[2, 3]),
    dict(cfg_id=2, train_frames=[0, 1, 2], test_frames=[3]),
]

N_PER_IMG_VALS  = [10, 25, 50, 75, 100]   # repeats × 3 each
N_PER_IMG_ALL   = "all"                    # deterministic → 1 repeat only
N_REPEATS       = 3


# ── Stratified subsample ──────────────────────────────────────────────────────

def stratified_subsample(
    df: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Return up to n rows: ceil(n/2) adhesion + floor(n/2) No adhesion."""
    adh  = df[df["label"] == "adhesion"]
    noad = df[df["label"] == "No adhesion"]
    n_a  = min(n // 2,     len(adh))
    n_b  = min(n - n // 2, len(noad))
    parts = []
    if n_a > 0:
        parts.append(adh.iloc[rng.choice(len(adh), n_a, replace=False)])
    if n_b > 0:
        parts.append(noad.iloc[rng.choice(len(noad), n_b, replace=False)])
    return pd.concat(parts) if parts else df.iloc[:0]


# ── YAML template ─────────────────────────────────────────────────────────────

YAML_TEMPLATE = """\
# =============================================================================
# Label-efficiency clean experiment — {label}
# train_frames={train_frames}  test_frames={test_frames}
# n_per_img={npi}  repeat={repeat}
# SupCon and classifier see exactly the same {n_labels} labels.
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
  result_dir : root_folder + "/ae_results/contrastive_run/le_clean/{name}"

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
  intensity_scale_range : [0.8, 1.2]

annotation:
  annotation_file : root_folder + "/labelling/le_clean/{name}.csv"
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be created without writing files")
    args = ap.parse_args()
    dry = args.dry_run

    # Load Annabel's 539-label file
    ann = pd.read_csv(ANN_FILE)
    ann["frame"] = ann["unique_ID"].apply(
        lambda u: int(re.search(r"f(\d+)", u).group(1))
    )
    print(f"Loaded {len(ann)} labels from {ANN_FILE.name}")

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)

    job_list = []
    rng_master = np.random.default_rng(0)

    for cfg in CONFIGS:
        c          = cfg["cfg_id"]
        train_frms = cfg["train_frames"]
        test_frms  = cfg["test_frames"]
        ann_train  = ann[ann["frame"].isin(train_frms)].copy()

        # Variable n_per_img with 3 repeats
        for npi in N_PER_IMG_VALS:
            for rep in range(N_REPEATS):
                name = f"le_c{c}_npi{npi}_r{rep}"
                rng  = np.random.default_rng(rng_master.integers(1 << 31))

                # Subsample K labels from each train frame (stratified)
                parts = []
                for f in train_frms:
                    df_f = ann_train[ann_train["frame"] == f]
                    parts.append(stratified_subsample(df_f, npi, rng))
                subset = pd.concat(parts, ignore_index=True)

                _write_job(name, subset, cfg, npi, rep, dry, job_list)

        # n_per_img="all" — deterministic, 1 repeat
        name   = f"le_c{c}_npiall_r0"
        subset = ann_train.copy()
        _write_job(name, subset, cfg, "all", 0, dry, job_list)

    # Write job list
    job_list_path = CONFIG_DIR / "job_list.txt"
    if not dry:
        with open(job_list_path, "w") as fh:
            fh.write("\n".join(job_list) + "\n")
        print(f"\nJob list ({len(job_list)} entries) → {job_list_path}")
    else:
        print(f"\n[dry-run] Would write {len(job_list)} jobs to {job_list_path}")

    print("\nPer-config totals:")
    for cfg in CONFIGS:
        c = cfg["cfg_id"]
        n = sum(1 for j in job_list if f"le_c{c}_" in j)
        print(f"  cfg{c}  train={cfg['train_frames']}  test={cfg['test_frames']}  → {n} jobs")
    print(f"\nTotal: {len(job_list)} training jobs")


def _write_job(name, subset, cfg, npi, rep, dry, job_list):
    c          = cfg["cfg_id"]
    train_frms = cfg["train_frames"]
    test_frms  = cfg["test_frames"]
    n_labels   = len(subset)

    ann_path    = OUT_ANN    / f"{name}.csv"
    config_path = CONFIG_DIR / f"{name}.yaml"

    label = (f"cfg{c}  train={train_frms}  test={test_frms}  "
             f"n_per_img={npi}  repeat={rep}  n_labels={n_labels}")
    print(f"  {name}: {n_labels} labels  "
          f"adh={( subset['label']=='adhesion').sum()}  "
          f"noad={(subset['label']=='No adhesion').sum()}")

    if not dry:
        subset.to_csv(ann_path, index=False)
        yaml_text = YAML_TEMPLATE.format(
            label=label,
            train_frames=train_frms,
            test_frames=test_frms,
            npi=npi,
            repeat=rep,
            n_labels=n_labels,
            name=name,
        )
        config_path.write_text(yaml_text)

    job_list.append(str(config_path.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
