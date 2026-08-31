#!/usr/bin/env python3
"""
setup_le_combined_s3v1.py
=========================
Generate configs for the combined image-count × label-count efficiency
experiment — s3v1 split (train: frames 0+1+2 · test: frame 3).

Axes
----
* N_images : AE training image count (frames 0,1,2 + unlabeled from {4,...})
* npi      : labels PER TRAINING FRAME (cumulative: 10 ⊂ 25 ⊂ 50 ⊂ 75 ⊂ 100)
             → total labels = npi × 3 frames

N_images values (15): [3,4,5,6,7,8,9,10,11,12,17,22,32,42,49]
  min=3 (frames 0,1,2 only), max=49 (frames 0,1,2,4,...,49)

Outputs
-------
  labelling/le_combined_s3v1/le_comb_s3_npi{npi}_r{series}.csv
  config/le_combined_s3v1/le_comb_s3_n{N:03d}_npi{npi}_r{series}.yaml
  config/le_combined_s3v1_job_list.txt   (225 entries)

Submission
----------
  sbatch --array=0-74   scripts/sbatch_le_combined_s3v1.sh   # series 0
  sbatch --array=75-149 scripts/sbatch_le_combined_s3v1.sh   # series 1
  sbatch --array=150-224 scripts/sbatch_le_combined_s3v1.sh  # series 2
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
PATCH_DIR  = DATA_ROOT / "ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
FRAME_DIR  = DATA_ROOT / "ae_results/source_frames/cio_mode_prt/vinc/control"
RUN_ROOT   = DATA_ROOT / "ae_results/contrastive_run/le_combined_s3v1"
CONFIG_DIR = REPO_ROOT / "config" / "le_combined_s3v1"
OUT_ANN    = LABEL_DIR / "le_combined_s3v1"

TRAIN_FRAMES = [0, 1, 2]
TEST_FRAMES  = [3]

N_VALUES   = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 22, 32, 42, 49]
NPI_LEVELS = [10, 25, 50, 75, 100]
N_SERIES   = 3
RNG_SEED   = 7777


def training_frames(n: int) -> list[int]:
    """Frames 0,1,2 (labeled) + n-3 unlabeled frames from {4,5,...}."""
    extra = list(range(4, 4 + (n - 3)))
    return sorted([0, 1, 2] + extra)


def cumulative_subsets_per_frame(
    df_frame: pd.DataFrame,
    npi_levels: list[int],
    rng: np.random.Generator,
) -> dict[int, pd.DataFrame]:
    """Nested subsets for a single frame: 10 ⊂ 25 ⊂ 50 ⊂ 75 ⊂ 100."""
    selected_idx: set[int] = set()
    subsets: dict[int, pd.DataFrame] = {}
    prev_n = 0
    for n in npi_levels:
        n_add     = n - prev_n
        remaining = df_frame[~df_frame.index.isin(selected_idx)]
        adh_pool  = remaining[remaining["label"] == "adhesion"]
        noad_pool = remaining[remaining["label"] == "No adhesion"]
        n_adh  = min(n_add // 2,         len(adh_pool))
        n_noad = min(n_add - n_add // 2, len(noad_pool))
        new_idx: list[int] = []
        if n_adh > 0:
            new_idx += adh_pool.index[
                rng.choice(len(adh_pool), n_adh, replace=False)
            ].tolist()
        if n_noad > 0:
            new_idx += noad_pool.index[
                rng.choice(len(noad_pool), n_noad, replace=False)
            ].tolist()
        selected_idx.update(new_idx)
        subsets[n] = df_frame.loc[sorted(selected_idx)].copy()
        prev_n = n
    return subsets


YAML_TEMPLATE = """\
# =============================================================================
# Combined efficiency s3v1 — N_images={n_images}  npi={npi}  series={series}
# AE trains on {n_frames} frame(s): {frames}
# LGBM labels: {n_labels} total ({npi} per frame × 3 frames, cumulative series {series})
# Test: frame 3  (always held out)
# =============================================================================

data:
  patch_dirs:
    - path           : "{patch_dir}"
      frame_dir      : "{frame_dir}"
      include_frames : {include_str}
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
  result_dir : "{result_dir}"

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
  annotation_file : "{ann_csv}"
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

    ann = pd.read_csv(ANN_FILE)
    ann["frame"] = ann["unique_ID"].apply(
        lambda u: int(re.search(r"f(\d+)", u).group(1))
    )

    for fr in TRAIN_FRAMES:
        sub = ann[ann["frame"] == fr]
        print(f"Frame {fr} labels: {len(sub)}  "
              f"(adh={(sub['label']=='adhesion').sum()}  "
              f"noad={(sub['label']=='No adhesion').sum()})")

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(RNG_SEED)

    # Step 1: annotation CSVs — per-frame cumulative, merged across frames
    ann_csvs: dict[tuple[int, int], Path] = {}
    for series in range(N_SERIES):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        frame_subsets: dict[int, dict[int, pd.DataFrame]] = {}
        for fr in TRAIN_FRAMES:
            df_fr = ann[ann["frame"] == fr].reset_index(drop=False)
            frame_subsets[fr] = cumulative_subsets_per_frame(df_fr, NPI_LEVELS, rng)

        for npi in NPI_LEVELS:
            parts = []
            for fr in TRAIN_FRAMES:
                part = frame_subsets[fr][npi].drop(columns=["index"], errors="ignore")
                parts.append(part)
            combined = pd.concat(parts, ignore_index=True)
            ann_name = f"le_comb_s3_npi{npi}_r{series}.csv"
            ann_path = OUT_ANN / ann_name
            ann_csvs[(npi, series)] = ann_path
            n_adh  = (combined["label"] == "adhesion").sum()
            n_noad = (combined["label"] == "No adhesion").sum()
            if not dry:
                combined.to_csv(ann_path, index=False)
            print(f"  ann  npi={npi:3d}  s{series}: {len(combined)} labels  "
                  f"adh={n_adh}  noad={n_noad}")

    # Step 2: YAML configs — ordered series → N → npi
    job_list: list[str] = []
    for series in range(N_SERIES):
        for n_images in N_VALUES:
            frames      = training_frames(n_images)
            include_str = "[" + ", ".join(str(f) for f in frames) + "]"
            for npi in NPI_LEVELS:
                run_name   = f"le_comb_s3_n{n_images:03d}_npi{npi}_r{series}"
                result_dir = RUN_ROOT / run_name
                ann_path   = ann_csvs[(npi, series)]
                cfg_path   = CONFIG_DIR / f"{run_name}.yaml"

                n_labels = len(pd.read_csv(ann_path)) if not dry else "?"
                yaml_text = YAML_TEMPLATE.format(
                    n_images=n_images, npi=npi, series=series,
                    n_frames=len(frames), frames=frames,
                    include_str=include_str,
                    patch_dir=PATCH_DIR, frame_dir=FRAME_DIR,
                    result_dir=result_dir, ann_csv=ann_path,
                    n_labels=n_labels,
                )
                if not dry:
                    cfg_path.write_text(yaml_text)
                job_list.append(str(cfg_path.relative_to(REPO_ROOT)))

    # Step 3: job list
    job_list_path = REPO_ROOT / "config" / "le_combined_s3v1_job_list.txt"
    if not dry:
        with open(job_list_path, "w") as fh:
            fh.write("\n".join(job_list) + "\n")

    n_per_series = len(N_VALUES) * len(NPI_LEVELS)
    print(f"\n[done] {len(job_list)} configs → {CONFIG_DIR}")
    print(f"       job list → {job_list_path}")
    print(f"  Series 0: --array=0-{n_per_series-1}")
    print(f"  Series 1: --array={n_per_series}-{2*n_per_series-1}")
    print(f"  Series 2: --array={2*n_per_series}-{3*n_per_series-1}")
    print(f"\nTotal: {len(job_list)} jobs")


if __name__ == "__main__":
    main()
