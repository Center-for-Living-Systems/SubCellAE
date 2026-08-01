#!/usr/bin/env python3
"""
generate_ds_combo_v4_configs.py

Generate ConAE training configs for all 15 dataset combinations using
cio_mode_prt normalisation — no clipping, no output sigmoid.

Two variants (analogous to v3 clip01/sc2_clip02 but without clip):
  prt      : input_divisor=1.0  (raw cio_mode_prt values, ~[−0.002, 1.5])
  prt_sc2  : input_divisor=2.0  (halved, ~[−0.001, 0.75])

Both use: cio_mode_prt frames, L1 loss, lambda_contrast=0.10, output_sigmoid=false,
          no input_clip_max, balanced repeats (same as v3).

Configs written to:
  config/contrastive_config/ds_combo_v4/ae_conae_enlcrop_prt_l1_lc010_bal_{combo}.yaml
  config/contrastive_config/ds_combo_v4/ae_conae_enlcrop_prt_sc2_l1_lc010_bal_{combo}.yaml

Usage:
  python scripts/generate_ds_combo_v4_configs.py
  python scripts/generate_ds_combo_v4_configs.py --dry-run
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

DATASETS = ["vinc", "nih3t3", "ppax", "pfak"]
REPEAT   = {"vinc": 1, "nih3t3": 2, "ppax": 4, "pfak": 8}
COUNTS   = {"vinc": 27637, "nih3t3": 7257, "ppax": 6667, "pfak": 3400}

# cio_mode_prt frames live here (populated by export_source_frames.py)
FRAME_ROOT = 'root_folder + "/ae_results/source_frames/cio_mode_prt'
# TIF patch directories — same FA detections, just used for patch location info
PATCH_ROOT = 'root_folder + "/ae_results/patches/cio'

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR   = REPO_ROOT / "config" / "contrastive_config" / "ds_combo_v4"


def _patch_dir_entries(combo: list[str]) -> str:
    lines = []
    cond_id = 0
    for ds in combo:
        for rep in range(REPEAT[ds]):
            rep_note = f" (repeat {rep+1}/{REPEAT[ds]})" if REPEAT[ds] > 1 else ""
            for cond in ("control", "ycomp"):
                lines.append(f"    # {ds} {cond}{rep_note}")
                lines.append(f'    - path           : {PATCH_ROOT}/{ds}/{cond}/tiff_patches32_mr10"')
                lines.append(f'      frame_dir      : {FRAME_ROOT}/{ds}/{cond}"')
                lines.append(f"      condition      : {cond_id if cond == 'control' else cond_id + 1}")
                lines.append(f'      condition_name : "{ds}_{cond}"')
                if ds == "vinc":
                    lines.append(f"      val_split      : 0.6")
        cond_id += 2
    return "\n".join(lines)


def _comment(combo: list[str]) -> str:
    parts = []
    for ds in combo:
        n = REPEAT[ds]
        total = COUNTS[ds]
        parts.append(f"{ds}×{n} (~{n*total//1000}k)")
    return " + ".join(parts)


def generate_config(combo: list[str], variant: str) -> str:
    combo_name  = "_".join(combo)
    patch_block = _patch_dir_entries(combo)

    if variant == "prt":
        input_divisor = 1.0
        sc2_note      = "cio_mode_prt / no clip / no sigmoid"
        run_subdir    = f"ds_combo_enlcrop_prt_l1/{combo_name}"
        cfg_prefix    = "ae_conae_enlcrop_prt_l1_lc010_bal"
    else:  # prt_sc2
        input_divisor = 2.0
        sc2_note      = "cio_mode_prt / ÷2 / no clip / no sigmoid"
        run_subdir    = f"ds_combo_enlcrop_prt_sc2_l1/{combo_name}"
        cfg_prefix    = "ae_conae_enlcrop_prt_sc2_l1_lc010_bal"

    return (
f"# =============================================================================\n"
f"# ConAE  |  {sc2_note}  |  L1  |  lambda_contrast=0.10\n"
f"# Training data: {_comment(combo)}  |  cio_mode_prt frames\n"
f"# Results: ae_results/contrastive_run/{run_subdir}/\n"
f"# =============================================================================\n"
f'root_folder : "/net/projects/CLS/lding/data/fa_data_analysis"\n'
f"\n"
f"data:\n"
f"  patch_dirs:\n"
f"{patch_block}\n"
f"\n"
f"enlarged_crop:\n"
f"  enabled       : true\n"
f'  channel       : "pax"\n'
f"  context_size  : 58\n"
f"  max_shift_px  : 4\n"
f"  max_angle_deg : 15.0\n"
f"  pad_size      : 64\n"
f"  input_divisor : {input_divisor}\n"
f"\n"
f"output:\n"
f'  result_dir : root_folder + "/ae_results/contrastive_run/{run_subdir}"\n'
f"\n"
f"model:\n"
f'  model_type      : "contrastive"\n'
f"  latent_dim      : 12\n"
f"  input_ps        : 32\n"
f"  no_ch           : 1\n"
f"  BN_flag         : false\n"
f"  dropout_flag    : false\n"
f"  output_sigmoid  : false\n"
f'  recon_loss_type : "l1"\n'
f"\n"
f"  proj_dim              : 8\n"
f"  noise_prob            : 0.05\n"
f"  temperature           : 0.5\n"
f"  lambda_recon          : 1.0\n"
f"  lambda_contrast       : 0.10\n"
f"  intensity_scale_range : [0.8, 1.2]\n"
f"\n"
f"training:\n"
f"  epochs                  : 500\n"
f"  lr                      : 0.001\n"
f"  batch_size              : 128\n"
f"  num_workers             : 6\n"
f"  val_split               : 0.2\n"
f"  loss_norm_flag          : false\n"
f"  group_split             : true\n"
f"  weight_decay            : 0.0001\n"
f"  warmup_epochs           : 0\n"
f'  lr_scheduler            : "cosine"\n'
f"  early_stopping_patience : 0\n"
f"  min_epochs_for_best     : 0\n"
f"\n"
f"reconstruction:\n"
f"  save_recon      : true\n"
f"  recon_pad_size  : 64\n"
f"  recon_image_size: 1024\n"
f"\n"
f"misc:\n"
f'  device    : "auto"\n'
f'  log_level : "INFO"\n'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_combos = []
    for r in range(1, len(DATASETS) + 1):
        for combo in combinations(DATASETS, r):
            all_combos.append(list(combo))

    print(f"Generating {len(all_combos)*2} configs ({len(all_combos)} combos × 2 variants) → {CFG_DIR}/")

    if not args.dry_run:
        CFG_DIR.mkdir(parents=True, exist_ok=True)

    combo_names = []
    for combo in all_combos:
        combo_name = "_".join(combo)
        combo_names.append(combo_name)
        for variant in ("prt", "prt_sc2"):
            cfg_text = generate_config(combo, variant)
            if variant == "prt":
                cfg_path = CFG_DIR / f"ae_conae_enlcrop_prt_l1_lc010_bal_{combo_name}.yaml"
            else:
                cfg_path = CFG_DIR / f"ae_conae_enlcrop_prt_sc2_l1_lc010_bal_{combo_name}.yaml"

            if args.dry_run:
                print(f"\n{'='*70}\n# {cfg_path.name}")
                print(cfg_text[:300] + "…")
            else:
                cfg_path.write_text(cfg_text)
                print(f"  wrote {cfg_path.name}")

    if not args.dry_run:
        list_path = CFG_DIR / "combo_list.txt"
        list_path.write_text("\n".join(combo_names) + "\n")
        print(f"\nCombo list → {list_path}")


if __name__ == "__main__":
    main()
