#!/usr/bin/env python3
"""
setup_ms_b2_ds1_ps64.py

Set up 5-fold CV training configs for DS1 B2 (Annabel vinc, ctrl+ycomp) at
patch size 64, using CoordCropDataset (online crop from source frames).

Produces:
  labelling/ms_b2_ds1_ps64/fold_splits.csv        -- all patches with fold 0-4
  labelling/ms_b2_ds1_ps64/ms_b2_ds1_ps64_fv{k}_train.csv  (5 files)
  config/ms_b2_ds1_ps64/ms_b2_ds1_ps64_fv{k}.yaml          (5 files)
  config/ms_b2_ds1_ps64/job_list.txt

Usage:
    python scripts/setup_ms_b2_ds1_ps64.py [--dry-run] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")

RUN_TAG   = "ms_b2_ds1_ps64_lat64p32"
N_FOLDS   = 5
PAD_SIZE  = 64   # patchprep pads coordinates by this amount

B2_LABEL  = DATA_ROOT / "labelling" / "vinc_combined_label_Annabel_20260816.csv"
ANN_DIR   = DATA_ROOT / "labelling" / RUN_TAG
CONFIG_DIR = REPO_ROOT / "config" / RUN_TAG

FNAME_PAT = re.compile(r'^(control|ycomp)_f(\d{4})x(\d{4})y(\d{4})ps32\.tif$')

COND_MAP = {"control": 0, "ycomp": 1}

# Remap 5-class FA labels → 2-class for SupCon training
LABEL_2CLS = {
    "No adhesion":        "No adhesion",
    "Nascent Adhesion":   "adhesion",
    "focal complex":      "adhesion",
    "focal adhesion":     "adhesion",
    "fibrillar adhesion": "adhesion",
    "Uncertain":          "Uncertain",   # → -1 (ignored in SupCon)
}

YAML_TEMPLATE = """\
# =============================================================================
# Multiscale B2 DS1 ps64 — {label}
# AE trained on DS1 B2 (Annabel, ctrl+ycomp) with patch_size=64.
# Patches cropped online from source frames via CoordCropDataset.
# =============================================================================
root_folder : "/net/projects/CLS/lding/data/fa_data_analysis"

data:
  coord_dirs:
    - coord_csv      : root_folder + "/labelling/{run_tag}/{name}_train.csv"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/vinc/control"
      channel        : "pax"
      condition      : 0
      condition_name : "control"
      annotation_label_col : "label"
      label_order    :
        - "No adhesion"
        - "adhesion"

    - coord_csv      : root_folder + "/labelling/{run_tag}/{name}_train.csv"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/vinc/ycomp"
      channel        : "pax"
      condition      : 1
      condition_name : "ycomp"
      annotation_label_col : "label"
      label_order    :
        - "No adhesion"
        - "adhesion"

  patch_dirs: []

output:
  result_dir : root_folder + "/ae_results/multiscale/{run_tag}/{name}"

model:
  model_type      : "supcon"
  latent_dim      : 64
  input_ps        : 64
  no_ch           : 1
  BN_flag         : false
  dropout_flag    : false
  output_sigmoid  : false
  recon_loss_type : "nl1"

  proj_dim              : 32
  noise_prob            : 0.0
  temperature           : 0.5
  lambda_recon          : 1.0
  lambda_contrast       : 0.5
  lambda_supcon         : 5.0
  intensity_scale_range : [0.8, 1.2]

annotation:
  annotation_file : ""
  label_col       : "label"
  label_order:
    - "No adhesion"
    - "adhesion"

training:
  epochs                  : 500
  lr                      : 0.001
  batch_size              : 128
  num_workers             : 0
  val_split               : 0.2
  loss_norm_flag          : false
  group_split             : true
  weight_decay            : 0.0001
  warmup_epochs           : 0
  lr_scheduler            : "none"
  early_stopping_patience : 0
  min_epochs_for_best     : 0

reconstruction:
  save_recon       : false
  recon_pad_size   : 80
  recon_image_size : 1024

misc:
  device    : "auto"
  log_level : "INFO"
"""


def parse_b2_labels(csv_path: Path) -> pd.DataFrame:
    """Parse B2 label CSV into coord DataFrame."""
    df = pd.read_csv(csv_path)
    parsed = []
    for _, row in df.iterrows():
        m = FNAME_PAT.match(row["filename"])
        if m is None:
            continue
        cond_name = m.group(1)
        frame_idx = int(m.group(2))
        cx = int(m.group(3)) - PAD_SIZE
        cy = int(m.group(4)) - PAD_SIZE
        raw_label = str(row["label"])
        parsed.append({
            "dataset":        "vinc",
            "condition":      COND_MAP[cond_name],
            "condition_name": cond_name,
            "frame_idx":      frame_idx,
            "cx":             cx,
            "cy":             cy,
            "source_ps":      32,
            "label":          LABEL_2CLS.get(raw_label, raw_label),
            "label_5cls":     raw_label,
            "annotator":      row.get("annotator", "Annabel"),
            "original_fn":    row["filename"],
        })
    return pd.DataFrame(parsed)


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Assign fold 0..n_folds-1 to each row (patch-level random)."""
    rng = random.Random(seed)
    folds = list(range(n_folds)) * (len(df) // n_folds + 1)
    rng.shuffle(folds)
    df = df.copy()
    df["fold"] = folds[:len(df)]
    return df


def make_train_csv(df: pd.DataFrame, test_fold: int) -> pd.DataFrame:
    """Return rows where fold != test_fold, dropping fold/original_fn columns."""
    keep = df[df["fold"] != test_fold].copy()
    return keep.drop(columns=["fold", "original_fn", "label_5cls"], errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    df = parse_b2_labels(B2_LABEL)
    print(f"Parsed {len(df)} patches from B2 DS1 label CSV")
    print(f"  ctrl: {(df['condition_name'] == 'control').sum()}")
    print(f"  ycomp: {(df['condition_name'] == 'ycomp').sum()}")

    df = assign_folds(df, N_FOLDS, args.seed)

    # Fold size summary
    for k in range(N_FOLDS):
        n_test  = (df["fold"] == k).sum()
        n_train = len(df) - n_test
        print(f"  fold {k}: train={n_train}  test={n_test}")

    if args.dry_run:
        print("DRY RUN — no files written.")
        return

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Save fold splits
    splits_path = ANN_DIR / "fold_splits.csv"
    df.to_csv(splits_path, index=False)
    print(f"Saved fold splits → {splits_path}")

    job_list = []
    for k in range(N_FOLDS):
        name  = f"{RUN_TAG}_fv{k}"
        label = f"fv{k} — test fold {k}"

        # Train coord CSV
        train_df = make_train_csv(df, test_fold=k)
        train_csv = ANN_DIR / f"{name}_train.csv"
        train_df.to_csv(train_csv, index=False)

        # YAML config
        yaml_str = YAML_TEMPLATE.format(
            label=label, run_tag=RUN_TAG, name=name)
        cfg_path = CONFIG_DIR / f"{name}.yaml"
        cfg_path.write_text(yaml_str)
        job_list.append(str(cfg_path.relative_to(REPO_ROOT)))

    # Job list
    jl_path = CONFIG_DIR / "job_list.txt"
    jl_path.write_text("\n".join(job_list) + "\n")
    print(f"Wrote {N_FOLDS} configs → {CONFIG_DIR}")
    print(f"Job list → {jl_path}")
    print(f"\nTo submit:\n  sbatch --array=0-{N_FOLDS - 1} scripts/sbatch_ms_b2_ds1_ps64.sh")


if __name__ == "__main__":
    main()
