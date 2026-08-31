#!/usr/bin/env python3
"""
setup_le_b12_supcon.py

Generate annotation CSVs, YAML configs and job list for the DS1 B1+B2
combined label-efficiency benchmark.

Label merging rules
-------------------
- Load B1 (labels_vinc_20260521.csv, Margaret) and B2 (vinc_combined_label_Annabel_20260816.csv, Annabel)
- Normalize filenames to underscore format for matching
- Patches labeled in BOTH -> drop (conflict)
- B1 patches labeled "Uncertain" -> drop
- Remaining patches from both -> combine, binarize

Result: ~2,481 labeled patches (vs 1,224 in B2 alone)

Same structure as le_b2_supcon DS1:
  5 folds x (11 budgets x 5 repeats + 1 all) = 280 jobs
  Budgets: [10, 20, 25, 50, 75, 100, 150, 200, 300, 500, 750, all]

Usage
-----
  python scripts/setup_le_b12_supcon.py [--dry-run]
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
CONFIG_DIR = REPO_ROOT / "config" / "le_b12_supcon"
OUT_ANN    = LABEL_DIR / "le_b12_supcon"

B1_FILE = LABEL_DIR / "labels_vinc_20260521.csv"
B2_FILE = LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv"

PATCH_DIRS_DS1 = [
    dict(path="ae_results/patches/cio/vinc/control/tiff_patches32_mr10",
         frame_dir="ae_results/source_frames/cio_mode_prt/vinc/control",
         condition=0, condition_name="control"),
    dict(path="ae_results/patches/cio/vinc/ycomp/tiff_patches32_mr10",
         frame_dir="ae_results/source_frames/cio_mode_prt/vinc/ycomp",
         condition=1, condition_name="ycomp"),
]

N_FOLDS   = 5
BUDGETS   = [10, 25, 50, 75, 150, 400, 750]
N_REPEATS = 5
CV_SEED   = 42


def _to_underscore(fn: str) -> str:
    return fn.replace("-f", "_f", 1)

def _to_hyphen(fn: str) -> str:
    return fn.replace("_f", "-f", 1)

def _binarize(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def load_combined_labels() -> pd.DataFrame:
    b2 = pd.read_csv(B2_FILE)
    b2["fn_norm"] = b2["filename"].apply(_to_underscore)
    b2["label"]   = b2["label"].apply(_binarize)

    b1 = pd.read_csv(B1_FILE)
    b1 = b1[b1["classification"] != "Uncertain"].copy()
    b1["fn_norm"] = b1["unique_ID"].apply(_to_underscore)
    b1["label"]   = b1["classification"].apply(_binarize)

    conflict = set(b2["fn_norm"]) & set(b1["fn_norm"])

    b2_keep = b2[~b2["fn_norm"].isin(conflict)][["fn_norm", "label"]]
    b1_keep = b1[~b1["fn_norm"].isin(conflict)][["fn_norm", "label"]]

    combined = pd.concat([b2_keep, b1_keep], ignore_index=True)
    combined = combined.rename(columns={"fn_norm": "filename"})

    print(f"  B2: {len(b2)} total, {len(b2_keep)} kept")
    print(f"  B1: {len(b1)} total (excl Uncertain), {len(b1_keep)} kept")
    print(f"  Conflicts dropped: {len(conflict)}")
    print(f"  Combined: {len(combined)}  "
          f"adh={(combined['label']=='adhesion').sum()}  "
          f"noad={(combined['label']=='No adhesion').sum()}")
    return combined


def stratified_subsample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
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


def _patch_dirs_block() -> str:
    lines = []
    for d in PATCH_DIRS_DS1:
        lines.append(
            f'    - path           : root_folder + "/{d["path"]}"\n'
            f'      frame_dir      : root_folder + "/{d["frame_dir"]}"\n'
            f'      condition      : {d["condition"]}\n'
            f'      condition_name : "{d["condition_name"]}"'
        )
    return "\n".join(lines)


YAML_TEMPLATE = """\
# =============================================================================
# LE B12 SupCon benchmark (DS1 B1+B2 combined) — {label}
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
  result_dir : root_folder + "/ae_results/contrastive_run/le_b12_supcon/{name}"

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
  annotation_file : root_folder + "/labelling/le_b12_supcon/{name}.csv"
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


def _write_job(name, fold, budget, rep, subset, dry, job_list):
    n_labels = len(subset)
    n_adh    = (subset["label"] == "adhesion").sum()
    n_noad   = (subset["label"] == "No adhesion").sum()
    label    = (f"ds1-b12  fold={fold}  budget={budget}  repeat={rep}  "
                f"n={n_labels} (adh={n_adh} noad={n_noad})")
    print(f"    {name}: n={n_labels}  adh={n_adh}  noad={n_noad}")

    ann_path    = OUT_ANN    / f"{name}.csv"
    config_path = CONFIG_DIR / f"{name}.yaml"

    if not dry:
        out_df = subset[["unique_ID", "label"]].copy()
        out_df.to_csv(ann_path, index=False)
        config_path.write_text(YAML_TEMPLATE.format(
            label=label, name=name,
            patch_dirs_block=_patch_dirs_block(),
        ))

    job_list.append(str(config_path.relative_to(REPO_ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry  = args.dry_run

    print("Loading combined B1+B2 labels for DS1...")
    df = load_combined_labels()
    df["unique_ID"] = df["filename"].apply(_to_hyphen)
    df = df.reset_index(drop=True)

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)

    job_list   = []
    rng_master = np.random.default_rng(0)
    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    fold_labels = np.empty(len(df), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(
            skf.split(np.arange(len(df)), df["label"].values)):
        fold_labels[test_idx] = fold

    splits_df = df[["unique_ID", "label"]].copy()
    splits_df["fold"] = fold_labels
    if not dry:
        (OUT_ANN / "fold_splits_ds1.csv").parent.mkdir(parents=True, exist_ok=True)
        splits_df.to_csv(OUT_ANN / "fold_splits_ds1.csv", index=False)

    for fold, (train_idx, test_idx) in enumerate(
            skf.split(np.arange(len(df)), df["label"].values)):
        train_pool = df.iloc[train_idx].copy().reset_index(drop=True)
        n_adh  = (train_pool["label"] == "adhesion").sum()
        n_noad = (train_pool["label"] == "No adhesion").sum()
        print(f"\n  Fold {fold}: train={len(train_pool)} (adh={n_adh} noad={n_noad})  test={len(test_idx)}")

        for budget in BUDGETS:
            for rep in range(N_REPEATS):
                rng  = np.random.default_rng(rng_master.integers(1 << 31))
                name = f"le_b12_ds1_fv{fold}_nb{budget}_r{rep}"
                subset = (train_pool.copy() if len(train_pool) <= budget
                          else stratified_subsample(train_pool, budget, rng))
                _write_job(name, fold, budget, rep, subset, dry, job_list)

        name = f"le_b12_ds1_fv{fold}_nball_r0"
        _write_job(name, fold, "all", 0, train_pool.copy(), dry, job_list)

    job_list_path = CONFIG_DIR / "job_list.txt"
    if not dry:
        job_list_path.write_text("\n".join(job_list) + "\n")
        print(f"\nJob list ({len(job_list)}) -> {job_list_path}")
    else:
        print(f"\n[dry-run] {len(job_list)} jobs")

    print(f"Total: {len(job_list)} jobs")


if __name__ == "__main__":
    main()
