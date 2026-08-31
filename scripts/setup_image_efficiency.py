#!/usr/bin/env python3
"""
setup_image_efficiency.py
=========================
Generate YAML configs and annotation CSVs for the image-count efficiency
experiment.

Design
------
* Fixed test set : frames 1, 2, 3 — Annabel-labeled, always held out from
                   both AE training and classifier.
* Training frame with labels : frame 0  (npi = all labels, 145 patches).
* Additional AE training frames : unlabeled frames 4, 5, … (skip 1–3).
* N = total images seen by AE : 1, 2, 3, … 10, 15, 20, 30, 40, 47.
  N=1  → only frame 0.
  N=k  → frame 0 + frames 4, 5, …, k+2   (k−1 unlabeled frames).
  Max N = 47  (frame 0 + frames 4–49 = 1 + 46).
* 3 repeats per N  (different AE random seeds via training stochasticity).

Outputs
-------
  config/img_eff/ie_n{N:03d}_r{repeat}.yaml   — AE training config
  <lab_dir>/img_eff/ie_frame0_all.csv          — shared annotation (frame 0)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PATCH_DIR  = DATA / "ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
FRAME_DIR  = DATA / "ae_results/source_frames/cio_mode_prt/vinc/control"
LE_DIR     = DATA / "ae_results/contrastive_run/img_eff"
LAB_DIR    = DATA / "labelling"
FULL_ANN   = LAB_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"
CFG_OUT    = _REPO / "config" / "img_eff"

TEST_FRAMES  = {1, 2, 3}   # always held out
TRAIN_LABEL_FRAME = 0       # only this frame provides classifier labels

N_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 47]
REPEATS  = [0, 1, 2]

# ---------------------------------------------------------------------------

def _extract_frame(fn: str) -> int:
    m = re.search(r"_f(\d+)", fn)
    return int(m.group(1)) if m else -1


def make_frame0_annotation(lab_dir: Path, full_ann: pd.DataFrame) -> Path:
    """Write frame-0-only annotation CSV (npi=all) shared across all N runs."""
    out_dir = lab_dir / "img_eff"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ie_frame0_all.csv"
    if out.exists():
        return out
    full_ann["frame"] = full_ann["filename"].apply(_extract_frame)
    f0 = full_ann[full_ann["frame"] == TRAIN_LABEL_FRAME].copy()
    f0 = f0.drop(columns=["frame"])
    f0.to_csv(out, index=False)
    print(f"[ann] wrote {len(f0)} frame-0 labels → {out}")
    return out


def training_frames(n: int) -> list[int]:
    """Return sorted list of frame indices the AE trains on for a given N."""
    if n == 1:
        return [TRAIN_LABEL_FRAME]
    # frame 0 (labeled) + n−1 unlabeled frames starting at 4 (skip test frames 1,2,3)
    unlabeled = list(range(4, n + 3))   # frames 4, 5, …, n+2  → n−1 frames
    return sorted([TRAIN_LABEL_FRAME] + unlabeled)


def write_config(n: int, repeat: int, ann_csv: Path) -> Path:
    frames = training_frames(n)
    run_name = f"ie_n{n:03d}_r{repeat}"
    result_dir = LE_DIR / run_name
    cfg_path   = CFG_OUT / f"{run_name}.yaml"

    include_str = "[" + ", ".join(str(f) for f in frames) + "]"

    yaml_text = f"""\
# =============================================================================
# Image-efficiency experiment — N={n} images  repeat={repeat}
# AE trains on {len(frames)} frame(s): {frames}
# Classifier trains on frame-0 labels only  (npi=all, {ann_csv.name})
# Test : frames 1, 2, 3  (held out from AE and classifier)
# =============================================================================

data:
  patch_dirs:
    - path           : "{PATCH_DIR}"
      frame_dir      : "{FRAME_DIR}"
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
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml_text)
    return cfg_path


def main():
    full_ann = pd.read_csv(FULL_ANN)
    ann_csv  = make_frame0_annotation(LAB_DIR, full_ann)

    cfg_paths = []
    for n in N_VALUES:
        frames = training_frames(n)
        # Validate: no test frames in training
        overlap = set(frames) & TEST_FRAMES
        if overlap:
            raise ValueError(f"N={n}: training frames {frames} overlap test frames {overlap}")
        for repeat in REPEATS:
            cfg_path = write_config(n, repeat, ann_csv)
            cfg_paths.append(cfg_path)
            print(f"  N={n:3d}  r{repeat}  frames={frames}  → {cfg_path.name}")

    # Write a flat job list for sbatch
    job_list = _REPO / "config" / "img_eff_job_list.txt"
    with open(job_list, "w") as fh:
        for p in cfg_paths:
            fh.write(str(p) + "\n")
    print(f"\n[done] {len(cfg_paths)} configs → {CFG_OUT}")
    print(f"       job list → {job_list}")


if __name__ == "__main__":
    main()
