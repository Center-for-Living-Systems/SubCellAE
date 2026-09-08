#!/usr/bin/env python3
"""
setup_le_b1b2_matched.py

Generate 5-fold annotation CSVs and SupCon-AE training configs for the
B1/B2 matched-label fair-comparison experiment.

Design
------
- Input: vinc_b1_matched_ds1.csv and vinc_b2_matched_ds1.csv (327 patches each)
- 5-fold stratified split (seed=42) per batch
- Per fold: training-set annotation CSV (80% of patches) written to:
    labelling/le_b1b2_matched/le_b1b2_{b1|b2}_fv{fold}.csv
- Full fold split metadata:
    labelling/le_b1b2_matched/fold_splits_b1.csv
    labelling/le_b1b2_matched/fold_splits_b2.csv
- YAML config per job:
    config/le_b1b2_matched/le_b1b2_{b1|b2}_fv{fold}.yaml
- Job list:
    config/le_b1b2_matched/job_list.txt
- SLURM script:
    scripts/sbatch_le_b1b2_matched.sh

Training covers ALL DS1 vinc patches for reconstruction; SupCon loss uses
only training-fold labels.  10 total jobs (5 folds × 2 batches).

Usage
-----
  python scripts/setup_le_b1b2_matched.py [--dry-run]
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
RUN_TAG    = "le_b1b2_matched"
CONFIG_DIR = REPO_ROOT / "config" / RUN_TAG
OUT_ANN    = LABEL_DIR / RUN_TAG

ANN_FILES = {
    "b1": LABEL_DIR / "vinc_b1_matched_ds1.csv",
    "b2": LABEL_DIR / "vinc_b2_matched_ds1.csv",
}

N_FOLDS  = 5
CV_SEED  = 42

YAML_TEMPLATE = """\
# =============================================================================
# LE B1/B2 matched fair-comparison — {label}
# All training-fold labels used for SupCon loss; all DS1 patches reconstructed.
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

SBATCH_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --job-name=le_b1b2
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/le_b1b2_matched_%A_%a.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUN_TAG="{run_tag}"
JOB_LIST="config/${{RUN_TAG}}/job_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")
NAME=$(basename "$CONFIG" .yaml)

FOLD=$(echo "$NAME"  | grep -oP '(?<=_fv)\\d+')
BATCH=$(echo "$NAME" | grep -oP '(?<=le_b1b2_)(b1|b2)(?=_fv)')

DATA="/net/projects/CLS/lding/data/fa_data_analysis"
ANN_CSV="${{DATA}}/labelling/${{RUN_TAG}}/${{NAME}}.csv"
FOLD_SPLITS="${{DATA}}/labelling/${{RUN_TAG}}/fold_splits_${{BATCH}}.csv"
RUN_DIR="${{DATA}}/ae_results/contrastive_run/${{RUN_TAG}}/${{NAME}}"

echo "task $SLURM_ARRAY_TASK_ID -> $NAME"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

$PYTHON scripts/run_ae_from_config.py "$CONFIG"

echo "--- LGBM eval ---"
$PYTHON scripts/eval_one_supcon_run.py \\
    --run-dir     "$RUN_DIR" \\
    --ann-csv     "$ANN_CSV" \\
    --fold-splits "$FOLD_SPLITS" \\
    --fold        "$FOLD" \\
    --budget      "all" \\
    --repeat      0

echo "End: $(date)"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / "logs" / "slurm").mkdir(parents=True, exist_ok=True)

    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    job_list = []

    for batch, ann_file in ANN_FILES.items():
        df = pd.read_csv(ann_file)
        print(f"\n{'='*60}")
        print(f"Batch: {batch}  total={len(df)}  "
              f"adh={(df['label']=='adhesion').sum()}  "
              f"noad={(df['label']=='No adhesion').sum()}")

        fold_labels = np.empty(len(df), dtype=int)
        for fold, (_, test_idx) in enumerate(skf.split(np.arange(len(df)), df["label"].values)):
            fold_labels[test_idx] = fold

        splits_df = df[["unique_ID", "label"]].copy()
        splits_df["fold"] = fold_labels
        splits_path = OUT_ANN / f"fold_splits_{batch}.csv"
        if not dry:
            splits_df.to_csv(splits_path, index=False)
        print(f"  fold splits → {splits_path}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(df)), df["label"].values)):
            train_pool = df.iloc[train_idx].copy()
            n_adh  = (train_pool["label"] == "adhesion").sum()
            n_noad = (train_pool["label"] == "No adhesion").sum()
            name   = f"le_b1b2_{batch}_fv{fold}"
            label  = (f"{batch}  fold={fold}  n_train={len(train_pool)} "
                      f"(adh={n_adh} noad={n_noad})  n_test={len(test_idx)}")
            print(f"  {name}: train={len(train_pool)}  test={len(test_idx)}"
                  f"  adh={n_adh}  noad={n_noad}")

            ann_path    = OUT_ANN    / f"{name}.csv"
            config_path = CONFIG_DIR / f"{name}.yaml"

            if not dry:
                train_pool[["unique_ID", "label"]].to_csv(ann_path, index=False)
                config_path.write_text(YAML_TEMPLATE.format(
                    label=label, name=name, run_tag=RUN_TAG))

            job_list.append(str(config_path.relative_to(REPO_ROOT)))

    job_list_path = CONFIG_DIR / "job_list.txt"
    sbatch_path   = REPO_ROOT / "scripts" / "sbatch_le_b1b2_matched.sh"

    if not dry:
        with open(job_list_path, "w") as fh:
            fh.write("\n".join(job_list) + "\n")
        sbatch_path.write_text(SBATCH_TEMPLATE.format(run_tag=RUN_TAG))
        sbatch_path.chmod(0o755)

    print(f"\nJob list ({len(job_list)} entries) → {job_list_path}")
    print(f"SLURM script → {sbatch_path}")
    print(f"Submit with: sbatch --array=0-{len(job_list)-1} scripts/sbatch_le_b1b2_matched.sh")


if __name__ == "__main__":
    main()
