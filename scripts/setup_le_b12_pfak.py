#!/usr/bin/env python3
"""
setup_le_b12_pfak.py

Generate annotation CSVs, YAML configs and job list for the DS2 (pfak)
B1+B2 combined label-efficiency benchmark.

Label merging:
  B1: labels_pfak_20260521.csv (Margaret, 54 patches, ctrl only)
  B2: pfak_combined_label_Annabel_aug2026.csv (Annabel, 211 patches)
  B2 takes priority on overlap (21 patches overlap).
  Combined: 244 patches  (175 adhesion, 69 no-adhesion)

Outputs into the le_b12_supcon infrastructure (same ann dir, same run tag):
  labelling/le_b12_supcon/fold_splits_ds2.csv
  labelling/le_b12_supcon/le_b12_ds2_fv{f}_nb{n}_r{r}.csv
  config/le_b12_supcon/le_b12_ds2_fv{f}_nb{n}_r{r}.yaml
  config/le_b12_supcon/job_list_ds2.txt

Budgets: [10, 20, 25, 50, 75, 100, 150, "all"]  (DS2/DS3 scale, 180 jobs)

Usage:
  python scripts/setup_le_b12_pfak.py [--dry-run]
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
RUN_TAG    = "le_b12_supcon"
CONFIG_DIR = REPO_ROOT / "config" / RUN_TAG
OUT_ANN    = LABEL_DIR / RUN_TAG

B1_FILE = LABEL_DIR / "labels_pfak_20260521.csv"
B2_FILE = LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv"

PATCH_DIRS = [
    dict(path="ae_results/patches/cio/pfak/control/tiff_patches32_mr10",
         frame_dir="ae_results/source_frames/cio_mode_prt/pfak/control",
         condition=0, condition_name="control"),
    dict(path="ae_results/patches/cio/pfak/ycomp/tiff_patches32_mr10",
         frame_dir="ae_results/source_frames/cio_mode_prt/pfak/ycomp",
         condition=1, condition_name="ycomp"),
]

BUDGETS   = [10, 20, 25, 50, 75, 100, 150]
N_FOLDS   = 5
N_REPEATS = 5
CV_SEED   = 42


def _binarize(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def _u2h(fn: str) -> str:
    return fn.replace("_f", "-f", 1)


def stratified_subsample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    adh  = df[df["label"] == "adhesion"]
    noad = df[df["label"] == "No adhesion"]
    n_a  = min(n - n // 2, len(adh))
    n_b  = min(n // 2,     len(noad))
    parts = []
    if n_a > 0: parts.append(adh.iloc[rng.choice(len(adh),  n_a, replace=False)])
    if n_b > 0: parts.append(noad.iloc[rng.choice(len(noad), n_b, replace=False)])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def _patch_dirs_block() -> str:
    lines = []
    for d in PATCH_DIRS:
        lines.append(
            f'    - path           : root_folder + "/{d["path"]}"\n'
            f'      frame_dir      : root_folder + "/{d["frame_dir"]}"\n'
            f'      condition      : {d["condition"]}\n'
            f'      condition_name : "{d["condition_name"]}"'
        )
    return "\n".join(lines)


YAML_TEMPLATE = """\
# =============================================================================
# LE B12 pfak DS2 — {label}
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


def build_combined_labels() -> pd.DataFrame:
    b1 = pd.read_csv(B1_FILE)
    b1["fn"]    = b1["unique_ID"].str.replace("-f", "_f", n=1)
    b1["label"] = b1["classification"].apply(_binarize)
    b1_out = b1[["fn", "label"]].copy()

    b2 = pd.read_csv(B2_FILE)
    b2["fn"]    = b2["filename"]
    b2["label"] = b2["label"].apply(_binarize)
    b2_out = b2[["fn", "label"]].copy()

    b2_ids = set(b2_out["fn"])
    combined = pd.concat([b2_out, b1_out[~b1_out["fn"].isin(b2_ids)]], ignore_index=True)
    combined["unique_ID"] = combined["fn"].apply(_u2h)
    return combined.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry  = args.dry_run

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)

    df = build_combined_labels()
    print(f"pfak B1+B2 combined: {len(df)}  "
          f"adh={(df['label']=='adhesion').sum()}  "
          f"noad={(df['label']=='No adhesion').sum()}")

    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    rng_master = np.random.default_rng(0)

    # Fold-split metadata
    fold_labels = np.empty(len(df), dtype=int)
    for fold, (_, test_idx) in enumerate(skf.split(np.arange(len(df)), df["label"].values)):
        fold_labels[test_idx] = fold
    splits_df = df[["unique_ID", "label"]].copy()
    splits_df["fold"] = fold_labels
    if not dry:
        splits_df.to_csv(OUT_ANN / "fold_splits_ds2.csv", index=False)
        print(f"Fold splits → {OUT_ANN}/fold_splits_ds2.csv")

    job_list = []
    patch_block = _patch_dirs_block()

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(df)), df["label"].values)):
        train_pool = df.iloc[train_idx].copy().reset_index(drop=True)
        print(f"Fold {fold}: train={len(train_pool)}  test={len(test_idx)}")

        for budget in BUDGETS:
            for rep in range(N_REPEATS):
                rng    = np.random.default_rng(rng_master.integers(1 << 31))
                name   = f"le_b12_ds2_fv{fold}_nb{budget}_r{rep}"
                subset = train_pool if len(train_pool) <= budget \
                         else stratified_subsample(train_pool, budget, rng)
                _write_job(name, subset, patch_block, dry, job_list,
                           f"ds2 fv{fold} nb{budget} r{rep} n={len(subset)}")

        name = f"le_b12_ds2_fv{fold}_nball_r0"
        _write_job(name, train_pool.copy(), patch_block, dry, job_list,
                   f"ds2 fv{fold} nb=all")

    job_list_path = CONFIG_DIR / "job_list_ds2.txt"
    if not dry:
        job_list_path.write_text("\n".join(job_list) + "\n")
        print(f"\nJob list ({len(job_list)} jobs) → {job_list_path}")
    else:
        print(f"\n[dry-run] {len(job_list)} jobs")


def _write_job(name, subset, patch_block, dry, job_list, label):
    ann_path    = OUT_ANN    / f"{name}.csv"
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not dry:
        subset[["unique_ID", "label"]].to_csv(ann_path, index=False)
        config_path.write_text(YAML_TEMPLATE.format(
            label=label, name=name,
            patch_dirs_block=patch_block,
            run_tag=RUN_TAG,
        ))
    job_list.append(str(config_path.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
