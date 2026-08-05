#!/usr/bin/env python3
"""
generate_annabel_sweep_configs.py

Generates all configs for the Annabel vinc control label sweep:
  • 2-class remapped label CSV (all adhesion types → "adhesion")
  • 9 training YAML configs  (conae / supcon2 / supcon5) × (s1v3 / s2v2 / s3v1)
  • 18 classification YAML configs  (9 models × z_recon / z_proj)
  • 9 analysis YAML configs  (UMAP + clustering per result dir)

Usage:
  python scripts/generate_annabel_sweep_configs.py
  python scripts/generate_annabel_sweep_configs.py --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = "/net/projects/CLS/lding/data/fa_data_analysis"
LABEL_DIR  = Path(DATA_ROOT) / "labelling"
CFG_DIR    = REPO_ROOT / "config" / "annabel_sweep"

LABEL_SRC   = "vinc_control_label_Annabel_20260715_1554.csv"
LABEL_2CLS  = "vinc_control_label_Annabel_20260715_1554_2cls.csv"
LABEL_5CLS  = LABEL_SRC

ADHESION_TYPES = {"focal adhesion", "Nascent Adhesion", "fibrillar adhesion", "focal complex"}

LABEL_ORDER_5 = [
    "No adhesion",
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]
LABEL_ORDER_2 = ["No adhesion", "adhesion"]

# model_key → (model_type, label_file, label_order)
MODELS = {
    "conae":   ("contrastive", LABEL_5CLS, LABEL_ORDER_5),
    "supcon2": ("supcon",      LABEL_2CLS, LABEL_ORDER_2),
    "supcon5": ("supcon",      LABEL_5CLS, LABEL_ORDER_5),
}

# split_key → val_split fraction
SPLITS = {
    "s1v3": 0.75,   # 1 train image, 3 val images
    "s2v2": 0.50,   # 2 / 2
    "s3v1": 0.25,   # 3 train, 1 val
}

MODEL_ORDER = ["conae", "supcon2", "supcon5"]
SPLIT_ORDER = ["s1v3", "s2v2", "s3v1"]


def _lo_yaml(order: list[str], indent: int = 4) -> str:
    pad = " " * indent
    return "\n".join(f'{pad}- "{x}"' for x in order)


# ---------------------------------------------------------------------------
# 2-class label CSV
# ---------------------------------------------------------------------------

def make_2cls_csv(dry_run: bool):
    src = LABEL_DIR / LABEL_SRC
    dst = LABEL_DIR / LABEL_2CLS
    df = pd.read_csv(src)
    df["label"] = df["label"].apply(
        lambda x: "adhesion" if x in ADHESION_TYPES else x
    )
    if dry_run:
        print(f"[dry] Would write 2-class CSV → {dst}")
        print(df["label"].value_counts().to_string())
    else:
        df.to_csv(dst, index=False)
        print(f"  wrote 2-class CSV → {dst.name}")
        print("  ", df["label"].value_counts().to_dict())


# ---------------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------------

def make_train_config(model: str, split: str) -> str:
    model_type, label_file, lo = MODELS[model]
    val_split   = SPLITS[split]
    result_name = f"annabel_vinc_{model}_{split}"

    annotation = f"""
annotation:
  annotation_file : root_folder + "/labelling/{label_file}"
  label_col       : "label"
  filename_col    : "unique_ID"
  label_order:
{_lo_yaml(lo)}
"""
    return f"""\
# =============================================================================
# Annabel vinc control — {model} {split}
# model_type={model_type}  latent=12  proj=8  val_split={val_split}
# split: {split.replace('s','').replace('v',' train / ')} val (per-image)
# =============================================================================
root_folder : "{DATA_ROOT}"

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
  result_dir : root_folder + "/ae_results/contrastive_run/{result_name}"

model:
  model_type      : "{model_type}"
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
{annotation}
training:
  epochs                  : 500
  lr                      : 0.001
  batch_size              : 128
  num_workers             : 6
  val_split               : {val_split}
  loss_norm_flag          : false
  group_split             : true
  weight_decay            : 0.0001
  warmup_epochs           : 0
  lr_scheduler            : "none"
  early_stopping_patience : 0
  min_epochs_for_best     : 0

reconstruction:
  save_recon       : true
  recon_pad_size   : 64
  recon_image_size : 1024

misc:
  device    : "auto"
  log_level : "INFO"
"""


# ---------------------------------------------------------------------------
# Classification configs
# ---------------------------------------------------------------------------

def make_cls_config(model: str, split: str, feat: str) -> str:
    _, _, lo = MODELS[model]
    result_name = f"annabel_vinc_{model}_{split}"
    prefix      = "z_" if feat == "zrecon" else "p_"
    feat_label  = "z_recon (latent)" if feat == "zrecon" else "z_proj (projection)"
    # Use fa_cls_* naming so pack_model_h5.py picks up predictions automatically
    out_subdir  = f"fa_cls_{feat}"

    return f"""\
# =============================================================================
# Classification — {result_name}  |  features: {feat_label}
# =============================================================================
root_folder : "{DATA_ROOT}"

input:
  latents_csv : root_folder + "/ae_results/contrastive_run/{result_name}/latents.csv"

output:
  out_dir : root_folder + "/ae_results/contrastive_run/{result_name}/{out_subdir}"

labels:
  label_col    : "annotation_label_name"
  filename_col : "filename"
  label_order:
{_lo_yaml(lo)}

features:
  feature_prefix        : "{prefix}"
  include_mean_intensity: false

split:
  strategy     : "from_csv"

classifier:
  type : "lgbm"

lightgbm:
  n_estimators      : 500
  learning_rate     : 0.05
  num_leaves        : 31
  min_child_samples : 10
  class_weight      : "balanced"
"""


# ---------------------------------------------------------------------------
# Analysis configs
# ---------------------------------------------------------------------------

def make_analysis_config(model: str, split: str) -> str:
    _, _, lo = MODELS[model]
    result_name = f"annabel_vinc_{model}_{split}"

    return f"""\
# =============================================================================
# Analysis — {result_name}
# =============================================================================
root_folder : "{DATA_ROOT}"

input:
  latents_csv  : root_folder + "/ae_results/contrastive_run/{result_name}/latents.csv"
  split_filter : "all"

output:
  out_dir : root_folder + "/ae_results/contrastive_run/{result_name}/analysis"

embedding:
  methods:
    - UMAP
  umap_n_neighbors  : 15
  umap_min_dist     : 0.1
  umap_random_state : 42

clustering:
  kmeans_enabled    : true
  kmeans_n_clusters : 5
  dbscan_enabled    : false
  boxplot_kind      : "violin"

label_orders:
  annotation_label_name:
{_lo_yaml(lo, indent=4)}
  condition_name:
    - "control"

misc:
  log_level : "INFO"
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        (CFG_DIR / "cls").mkdir(parents=True, exist_ok=True)
        (CFG_DIR / "analysis").mkdir(parents=True, exist_ok=True)

    print("=== 2-class label CSV ===")
    make_2cls_csv(args.dry_run)

    print("\n=== Training configs ===")
    train_cfgs = []
    for model in MODEL_ORDER:
        for split in SPLIT_ORDER:
            name = f"ae_annabel_vinc_{model}_{split}.yaml"
            path = CFG_DIR / name
            content = make_train_config(model, split)
            train_cfgs.append(str(path.relative_to(REPO_ROOT)))
            if args.dry_run:
                print(f"  [dry] {name}")
            else:
                path.write_text(content)
                print(f"  wrote {name}")

    print("\n=== Classification configs ===")
    cls_cfgs = []
    for model in MODEL_ORDER:
        for split in SPLIT_ORDER:
            for feat in ["zrecon", "zproj"]:
                name = f"cls_annabel_vinc_{model}_{split}_{feat}.yaml"
                path = CFG_DIR / "cls" / name
                content = make_cls_config(model, split, feat)
                cls_cfgs.append(str(path.relative_to(REPO_ROOT)))
                if args.dry_run:
                    print(f"  [dry] {name}")
                else:
                    path.write_text(content)
                    print(f"  wrote {name}")

    print("\n=== Analysis configs ===")
    analysis_cfgs = []
    for model in MODEL_ORDER:
        for split in SPLIT_ORDER:
            name = f"analysis_annabel_vinc_{model}_{split}.yaml"
            path = CFG_DIR / "analysis" / name
            content = make_analysis_config(model, split)
            analysis_cfgs.append(str(path.relative_to(REPO_ROOT)))
            if args.dry_run:
                print(f"  [dry] {name}")
            else:
                path.write_text(content)
                print(f"  wrote {name}")

    if not args.dry_run:
        # Write config lists for sbatch scripts
        (CFG_DIR / "train_configs.txt").write_text("\n".join(train_cfgs) + "\n")
        (CFG_DIR / "cls_configs.txt").write_text("\n".join(cls_cfgs) + "\n")
        (CFG_DIR / "analysis_configs.txt").write_text("\n".join(analysis_cfgs) + "\n")
        print(f"\n  wrote config lists → {CFG_DIR.relative_to(REPO_ROOT)}/{{train,cls,analysis}}_configs.txt")

    print(f"\nDone.  {len(train_cfgs)} train | {len(cls_cfgs)} cls | {len(analysis_cfgs)} analysis configs")


if __name__ == "__main__":
    main()
