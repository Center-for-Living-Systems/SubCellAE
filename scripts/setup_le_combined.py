#!/usr/bin/env python3
"""
setup_le_combined.py
====================
Generate configs for the combined image-count × label-count efficiency
experiment.

Two axes
--------
* N_images : how many frames the AE trains on
             frame 0 (labeled) + N-1 unlabeled frames from {4,5,…,49}
             frames 1,2,3 are always held out (they provide the test labels)

* npi      : how many labels per image the SupCon AE and classifier see
             labels are CUMULATIVE within a series: 10 ⊂ 25 ⊂ 50 ⊂ 75 ⊂ 100

Fixed split: s1v3 — frame 0 trains, frames 1,2,3 test.
             (only valid split when N=1; kept constant for comparability)

Outputs
-------
  labelling/le_combined/le_comb_npi{npi}_r{series}.csv   (shared annotation)
  config/le_combined/le_comb_n{N:03d}_npi{npi}_r{series}.yaml
  config/le_combined_job_list.txt   (225 entries, ordered series→N→npi)

Submission
----------
  All 225 jobs : sbatch scripts/sbatch_le_combined.sh
  Single series: sbatch --array=1-75  scripts/sbatch_le_combined.sh
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
RUN_ROOT   = DATA_ROOT / "ae_results/contrastive_run/le_combined"
CONFIG_DIR = REPO_ROOT / "config" / "le_combined"
OUT_ANN    = LABEL_DIR / "le_combined"

N_VALUES   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 47]
NPI_LEVELS = [10, 25, 50, 75, 100]
N_SERIES   = 3

TEST_FRAMES        = {1, 2, 3}
TRAIN_LABEL_FRAME  = 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def training_frames(n: int) -> list[int]:
    """Frame 0 (labeled) + n-1 unlabeled frames from {4,5,...}."""
    if n == 1:
        return [TRAIN_LABEL_FRAME]
    return sorted([TRAIN_LABEL_FRAME] + list(range(4, n + 3)))


def cumulative_subsets(
    df: pd.DataFrame,
    npi_levels: list[int],
    rng: np.random.Generator,
) -> dict[int, pd.DataFrame]:
    """Return {npi: subset_df} with 10 ⊂ 25 ⊂ 50 ⊂ 75 ⊂ 100."""
    selected_idx: set[int] = set()
    subsets: dict[int, pd.DataFrame] = {}
    prev_n = 0
    for n in npi_levels:
        n_add     = n - prev_n
        remaining = df[~df.index.isin(selected_idx)]
        adh_pool  = remaining[remaining["label"] == "adhesion"]
        noad_pool = remaining[remaining["label"] == "No adhesion"]
        n_adh  = min(n_add // 2,        len(adh_pool))
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
        subsets[n] = df.loc[sorted(selected_idx)].copy()
        prev_n = n
    return subsets


YAML_TEMPLATE = """\
# =============================================================================
# Combined efficiency — N_images={n_images}  npi={npi}  series={series}
# AE trains on {n_frames} frame(s): {frames}
# SupCon + classifier: {n_labels} labels (cumulative series {series})
# Test: frames 1, 2, 3  (always held out)
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    ann = pd.read_csv(ANN_FILE)
    ann["frame"] = ann["unique_ID"].apply(
        lambda u: int(re.search(r"f(\d+)", u).group(1))
    )
    df_f0 = ann[ann["frame"] == TRAIN_LABEL_FRAME].reset_index(drop=False)
    print(f"Frame-0 labels: {len(df_f0)}  "
          f"(adh={( df_f0['label']=='adhesion').sum()}  "
          f"noad={(df_f0['label']=='No adhesion').sum()})")

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(9999)

    # ── Step 1: generate per-series cumulative label CSVs (shared across N) ──
    # 5 npi × 3 series = 15 annotation files
    ann_csvs: dict[tuple[int, int], Path] = {}   # (npi, series) → path
    for series in range(N_SERIES):
        rng     = np.random.default_rng(rng_master.integers(1 << 31))
        subsets = cumulative_subsets(df_f0, NPI_LEVELS, rng)
        for npi, subset in subsets.items():
            subset = subset.drop(columns=["index"], errors="ignore")
            ann_name = f"le_comb_npi{npi}_r{series}.csv"
            ann_path = OUT_ANN / ann_name
            ann_csvs[(npi, series)] = ann_path
            n_adh  = (subset["label"] == "adhesion").sum()
            n_noad = (subset["label"] == "No adhesion").sum()
            if not dry:
                subset.to_csv(ann_path, index=False)
            print(f"  ann  npi={npi:3d}  s{series}: {len(subset)} labels  "
                  f"adh={n_adh}  noad={n_noad}")

    # ── Step 2: generate YAML configs for each (N, npi, series) ─────────────
    # Ordered by series → N → npi for grouped submission
    job_list: list[str] = []
    for series in range(N_SERIES):
        for n_images in N_VALUES:
            frames      = training_frames(n_images)
            include_str = "[" + ", ".join(str(f) for f in frames) + "]"
            for npi in NPI_LEVELS:
                run_name   = f"le_comb_n{n_images:03d}_npi{npi}_r{series}"
                result_dir = RUN_ROOT / run_name
                ann_path   = ann_csvs[(npi, series)]
                cfg_path   = CONFIG_DIR / f"{run_name}.yaml"

                yaml_text = YAML_TEMPLATE.format(
                    n_images=n_images,
                    npi=npi,
                    series=series,
                    n_frames=len(frames),
                    frames=frames,
                    include_str=include_str,
                    patch_dir=PATCH_DIR,
                    frame_dir=FRAME_DIR,
                    result_dir=result_dir,
                    ann_csv=ann_path,
                    n_labels=(len(pd.read_csv(ann_path)) if not dry else "?"),
                )
                if not dry:
                    cfg_path.write_text(yaml_text)
                job_list.append(str(cfg_path.relative_to(REPO_ROOT)))

    # ── Step 3: write job list ────────────────────────────────────────────────
    job_list_path = REPO_ROOT / "config" / "le_combined_job_list.txt"
    if not dry:
        with open(job_list_path, "w") as fh:
            fh.write("\n".join(job_list) + "\n")

    n_per_series = len(N_VALUES) * len(NPI_LEVELS)
    print(f"\n[done] {len(job_list)} configs → {CONFIG_DIR}")
    print(f"       job list → {job_list_path}")
    print(f"  Series 0: --array=1-{n_per_series}")
    print(f"  Series 1: --array={n_per_series+1}-{2*n_per_series}")
    print(f"  Series 2: --array={2*n_per_series+1}-{3*n_per_series}")
    print(f"\nTotal: {len(job_list)} jobs  "
          f"({N_SERIES} series × {len(N_VALUES)} N_images × {len(NPI_LEVELS)} npi)")


if __name__ == "__main__":
    main()
