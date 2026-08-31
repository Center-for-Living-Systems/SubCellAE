#!/usr/bin/env python3
"""
setup_le_cumulative.py
======================
Generate annotation CSVs and YAML configs for the cumulative label-efficiency
experiment.

Design
------
Same 3 frame splits and 3 series as le_clean, but label sets are NESTED:
  10 ⊂ 25 ⊂ 50 ⊂ 75 ⊂ 100  within each (cfg, series)

This isolates whether accuracy gains come from better coverage or simply more
data: comparing le_clean vs le_cumulative shows the effect of label choice.

Nesting is applied per training frame: frame 0's 10 labels are a subset of
frame 0's 25 labels, and likewise for frames 1 and 2 in multi-frame cfgs.

Outputs
-------
  labelling/le_cumulative/le_c{c}_npi{npi}_r{s}.csv
  config/le_cumulative/le_c{c}_npi{npi}_r{s}.yaml
  config/le_cumulative_job_list.txt

Job list is ordered by series so a single series can be submitted with
  --array=1-15   (series 0)
  --array=16-30  (series 1)
  --array=31-45  (series 2)

Usage
-----
  python scripts/setup_le_cumulative.py [--dry-run]
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
CONFIG_DIR = REPO_ROOT / "config" / "le_cumulative"
OUT_ANN    = LABEL_DIR / "le_cumulative"
RUN_ROOT   = DATA_ROOT / "ae_results" / "contrastive_run" / "le_cumulative"

CONFIGS = [
    dict(cfg_id=0, train_frames=[0],       test_frames=[1, 2, 3]),
    dict(cfg_id=1, train_frames=[0, 1],    test_frames=[2, 3]),
    dict(cfg_id=2, train_frames=[0, 1, 2], test_frames=[3]),
]

NPI_LEVELS = [10, 25, 50, 75, 100]   # must be ascending; each is a superset
N_SERIES   = 3                        # independent starting seeds

YAML_TEMPLATE = """\
# =============================================================================
# Label-efficiency CUMULATIVE — {label}
# train_frames={train_frames}  test_frames={test_frames}
# npi={npi}  series={series}  (labels nested: 10 ⊂ 25 ⊂ 50 ⊂ 75 ⊂ 100)
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
  result_dir : root_folder + "/ae_results/contrastive_run/le_cumulative/{name}"

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
  annotation_file : root_folder + "/labelling/le_cumulative/{name}.csv"
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


# ── Cumulative (nested) subsample ────────────────────────────────────────────

def cumulative_subsets(df: pd.DataFrame, npi_levels: list[int],
                       rng: np.random.Generator) -> dict[int, pd.DataFrame]:
    """Return {npi: subset_df} where smaller sets are subsets of larger ones.

    At each level we add stratified samples from the labels not yet chosen.
    """
    selected_idx: set[int] = set()
    subsets: dict[int, pd.DataFrame] = {}

    prev_n = 0
    for n in npi_levels:
        n_add = n - prev_n
        remaining = df[~df.index.isin(selected_idx)]

        adh_pool  = remaining[remaining["label"] == "adhesion"]
        noad_pool = remaining[remaining["label"] == "No adhesion"]

        n_adh  = min(n_add // 2,       len(adh_pool))
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
    print(f"Loaded {len(ann)} labels from {ANN_FILE.name}")

    if not dry:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ANN.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(1234)   # different seed from le_clean (seed=0)

    # Job list ordered by series so --array=1-15 submits one full series
    # Order: series 0 all cfgs/npis → series 1 → series 2
    job_list: list[str] = []

    for series in range(N_SERIES):
        for cfg in CONFIGS:
            c          = cfg["cfg_id"]
            train_frms = cfg["train_frames"]
            test_frms  = cfg["test_frames"]
            ann_train  = ann[ann["frame"].isin(train_frms)].copy()

            # Build per-frame cumulative subsets, then merge across frames
            per_frame_subsets: dict[int, dict[int, pd.DataFrame]] = {}
            for f in train_frms:
                df_f = ann_train[ann_train["frame"] == f].reset_index(drop=False)
                rng  = np.random.default_rng(rng_master.integers(1 << 31))
                per_frame_subsets[f] = cumulative_subsets(df_f, NPI_LEVELS, rng)

            for npi in NPI_LEVELS:
                name = f"le_c{c}_npi{npi}_r{series}"

                # Merge subsets from all training frames
                subset = pd.concat(
                    [per_frame_subsets[f][npi] for f in train_frms],
                    ignore_index=True,
                )
                # Drop the helper 'index' column added by reset_index
                subset = subset.drop(columns=["index"], errors="ignore")

                n_adh  = (subset["label"] == "adhesion").sum()
                n_noad = (subset["label"] == "No adhesion").sum()
                print(f"  {name}: {len(subset)} labels  "
                      f"adh={n_adh}  noad={n_noad}")

                ann_path    = OUT_ANN    / f"{name}.csv"
                config_path = CONFIG_DIR / f"{name}.yaml"
                label_str   = (f"cfg{c}  train={train_frms}  test={test_frms}  "
                               f"npi={npi}  series={series}  n_labels={len(subset)}")

                if not dry:
                    subset.to_csv(ann_path, index=False)
                    yaml_text = YAML_TEMPLATE.format(
                        label=label_str,
                        train_frames=train_frms,
                        test_frames=test_frms,
                        npi=npi,
                        series=series,
                        n_labels=len(subset),
                        name=name,
                    )
                    config_path.write_text(yaml_text)

                job_list.append(str(config_path.relative_to(REPO_ROOT)))

    job_list_path = REPO_ROOT / "config" / "le_cumulative_job_list.txt"
    if not dry:
        with open(job_list_path, "w") as fh:
            fh.write("\n".join(job_list) + "\n")
        print(f"\nJob list ({len(job_list)} entries) → {job_list_path}")
        n_per_series = len(NPI_LEVELS) * len(CONFIGS)
        print(f"  Series 0: --array=1-{n_per_series}")
        print(f"  Series 1: --array={n_per_series+1}-{2*n_per_series}")
        print(f"  Series 2: --array={2*n_per_series+1}-{3*n_per_series}")
    else:
        print(f"\n[dry-run] Would write {len(job_list)} jobs")

    print(f"\nTotal: {len(job_list)} training jobs  "
          f"({N_SERIES} series × {len(CONFIGS)} cfgs × {len(NPI_LEVELS)} npi)")


if __name__ == "__main__":
    main()
