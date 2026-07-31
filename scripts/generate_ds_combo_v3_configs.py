#!/usr/bin/env python3
"""
generate_ds_combo_v3_configs.py

Generate ConAE training configs for all 15 dataset combinations of
{vinc, nih3t3, ppax, pfak}, in two clip variants:

  clip_01  : input clipped to [0,1], no sc2 division (input_divisor=1.0)
             → model input and reconstruction target are both in [0,1]
  clip_02  : input clipped to [0,2], then sc2 division by 2 (input_divisor=2.0)
             → model input and reconstruction target are in [0,1] after division

Both use: CIO (no RB) patches, L1 loss, lambda_contrast=0.10, balanced repeats.

Balancing — each ds repeated to contribute ~27k patches:
  vinc   : ×1  (27.6k)   val_split=0.6 (40% train → ~11k)
  nih3t3 : ×2  (2×7.3k ≈ 14.6k)
  ppax   : ×4  (4×6.7k ≈ 27k)
  pfak   : ×8  (8×3.4k ≈ 27k)

Configs written to:
  config/contrastive_config/ds_combo_v3/ae_conae_enlcrop_clip01_l1_lc010_bal_{combo}.yaml
  config/contrastive_config/ds_combo_v3/ae_conae_enlcrop_sc2_clip02_l1_lc010_bal_{combo}.yaml

Also writes:
  config/contrastive_config/ds_combo_v3/combo_list.txt

Usage:
  python scripts/generate_ds_combo_v3_configs.py
  python scripts/generate_ds_combo_v3_configs.py --dry-run
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

# ── dataset metadata ──────────────────────────────────────────────────────────

DATASETS = ["vinc", "nih3t3", "ppax", "pfak"]

REPEAT = {"vinc": 1, "nih3t3": 2, "ppax": 4, "pfak": 8}

PATCH_ROOT = 'root_folder + "/ae_results/patches/cio'
FRAME_ROOT  = 'root_folder + "/ae_results/source_frames/cio'

COUNTS = {"vinc": 27637, "nih3t3": 7257, "ppax": 6667, "pfak": 3400}

# ── output dirs ───────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR   = REPO_ROOT / "config" / "contrastive_config" / "ds_combo_v3"

# ── helpers ───────────────────────────────────────────────────────────────────

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
    """
    variant: "clip_01" (clip [0,1], no sc2) or "clip_02" (clip [0,2], sc2 ÷2)
    """
    combo_name  = "_".join(combo)
    patch_block = _patch_dir_entries(combo)

    if variant == "clip_01":
        clip_max      = 1.0
        input_divisor = 1.0
        sc2_note      = "clip [0,1] → input ∈ [0,1] (no divisor)"
        run_subdir    = f"ds_combo_enlcrop_clip01_l1/{combo_name}"
        cfg_prefix    = "ae_conae_enlcrop_clip01_l1_lc010_bal"
    else:  # clip_02
        clip_max      = 2.0
        input_divisor = 2.0
        sc2_note      = "clip [0,2] → ÷2 (sc2) → input ∈ [0,1]"
        run_subdir    = f"ds_combo_enlcrop_sc2_clip02_l1/{combo_name}"
        cfg_prefix    = "ae_conae_enlcrop_sc2_clip02_l1_lc010_bal"

    return (
f"# =============================================================================\n"
f"# ConAE  |  {sc2_note}  |  L1  |  lambda_contrast=0.10\n"
f"# Training data: {_comment(combo)}  |  CIO (no RB)\n"
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
f"  input_clip_max: {clip_max}\n"
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
f"  output_sigmoid  : true\n"
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

    print(f"Generating {len(all_combos)*2} configs ({len(all_combos)} combos × 2 clip variants) → {CFG_DIR}/")

    if not args.dry_run:
        CFG_DIR.mkdir(parents=True, exist_ok=True)

    combo_names = []
    for combo in all_combos:
        combo_name = "_".join(combo)
        combo_names.append(combo_name)
        for variant in ("clip_01", "clip_02"):
            cfg_text = generate_config(combo, variant)
            if variant == "clip_01":
                cfg_path = CFG_DIR / f"ae_conae_enlcrop_clip01_l1_lc010_bal_{combo_name}.yaml"
            else:
                cfg_path = CFG_DIR / f"ae_conae_enlcrop_sc2_clip02_l1_lc010_bal_{combo_name}.yaml"

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
        print(f"\nAll combos ({len(combo_names)}):")
        for i, name in enumerate(combo_names):
            print(f"  [{i:2d}] {name}")


if __name__ == "__main__":
    main()
