#!/usr/bin/env python3
"""
setup_le_supcon.py

Generate YAML configs for the SupCon label-efficiency experiment:
  cfg0 (train=[0], test=[1,2,3]) × 6 npi × (3 or 1) repeats = 16 jobs per tag

Usage
-----
  # single tag
  python scripts/setup_le_supcon.py --lambda-supcon 2.0 --tag ls2

  # all three at once (48 jobs combined)
  python scripts/setup_le_supcon.py --all
  python scripts/setup_le_supcon.py --dry-run --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")

N_PER_CLASS = 2
NPI_VALS    = [10, 25, 50, 75, 100]
N_REPEATS   = 3

YAML_TEMPLATE = """\
# =============================================================================
# SupCon label-efficiency — cfg0  train=[0]  test=[1, 2, 3]
# n_per_img={npi}  repeat={repeat}  n_labels={n_labels}  tag={tag}
# LabeledAwareBatchSampler: n_per_class={n_per_class} (guaranteed per batch)
# lambda_supcon={lambda_supcon}  lambda_contrast=0.5
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
  result_dir : root_folder + "/ae_results/contrastive_run/le_supcon_{tag}/{name}"

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
  lambda_supcon         : {lambda_supcon}
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
  n_labeled_per_class     : {n_per_class}
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


def generate_tag(tag: str, lambda_supcon: float, dry: bool) -> list[str]:
    config_dir = REPO_ROOT / "config" / f"le_supcon_{tag}"
    if not dry:
        config_dir.mkdir(parents=True, exist_ok=True)

    job_list: list[str] = []

    def write_job(name: str, npi, repeat: int, n_labels):
        ann_csv = DATA_ROOT / "labelling" / "le_clean" / f"{name}.csv"
        if not ann_csv.exists():
            print(f"  [warn] annotation CSV missing: {ann_csv}")

        yaml = YAML_TEMPLATE.format(
            name=name,
            npi=npi,
            repeat=repeat,
            n_labels=n_labels,
            n_per_class=N_PER_CLASS,
            tag=tag,
            lambda_supcon=lambda_supcon,
        )
        cfg_path = config_dir / f"{name}.yaml"
        if not dry:
            cfg_path.write_text(yaml)
        else:
            print(f"  [dry-run] {cfg_path}")
        job_list.append(str(cfg_path))

    for npi in NPI_VALS:
        for rep in range(N_REPEATS):
            write_job(f"le_c0_npi{npi}_r{rep}", npi, rep, npi)

    write_job("le_c0_npiall_r0", "all", 0, 145)
    return job_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-supcon", type=float, default=None)
    ap.add_argument("--tag",           type=str,   default=None)
    ap.add_argument("--all",           action="store_true",
                    help="Generate ls1/ls15/ls2 configs and a combined job list")
    ap.add_argument("--dry-run",       action="store_true")
    args = ap.parse_args()
    dry  = args.dry_run

    if args.all:
        sweep = [("ls1", 1.0), ("ls15", 1.5), ("ls2", 2.0)]
    else:
        if args.lambda_supcon is None or args.tag is None:
            ap.error("Provide --lambda-supcon and --tag, or use --all")
        sweep = [(args.tag, args.lambda_supcon)]

    all_jobs: list[str] = []
    for tag, lam in sweep:
        jobs = generate_tag(tag, lam, dry)
        all_jobs.extend(jobs)
        print(f"{'[dry-run] ' if dry else ''}tag={tag}  lambda_supcon={lam}  → {len(jobs)} configs")

    # write combined job list for the array sbatch
    combined_path = REPO_ROOT / "config" / "le_supcon_sweep_job_list.txt"
    if not dry:
        combined_path.write_text("\n".join(all_jobs) + "\n")
    print(f"{'[dry-run] ' if dry else ''}Combined job list ({len(all_jobs)} jobs) → {combined_path}")
    print(f"  n_labeled_per_class = {N_PER_CLASS}  (guaranteed {N_PER_CLASS * 2} labeled/batch)")


if __name__ == "__main__":
    main()
