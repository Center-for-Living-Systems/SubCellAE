#!/usr/bin/env python3
"""
Generate conAE strategy sweep configs for lat12proj8 × all 11 strategies.
Skips 0322 (already exists). For each strategy:
  ae_contrastive_cio_rb_vinc_lat12proj8_{strategy}.yaml
  analysis_contrastive_cio_rb_vinc_lat12proj8_{strategy}.yaml
  cls_contrastive_cio_rb_vinc_lat12proj8_{strategy}_fa_zrecon.yaml
  cls_contrastive_cio_rb_vinc_lat12proj8_{strategy}_fa_zproj.yaml
  cls_contrastive_cio_rb_vinc_lat12proj8_{strategy}_pos_zrecon.yaml
  cls_contrastive_cio_rb_vinc_lat12proj8_{strategy}_pos_zproj.yaml
  vis_contrastive_cio_rb_vinc_lat12proj8_{strategy}_zrecon.yaml
  vis_contrastive_cio_rb_vinc_lat12proj8_{strategy}_zproj.yaml
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "config" / "contrastive_config"

STRATEGIES = {
    "0322":         dict(epochs=200,  weight_decay=0.0,    warmup_epochs=0,   min_epochs_for_best=0,   lr_scheduler="none"),
    "0324":         dict(epochs=500,  weight_decay=0.0001, warmup_epochs=0,   min_epochs_for_best=200, lr_scheduler="none"),
    "mar30":        dict(epochs=500,  weight_decay=0.0001, warmup_epochs=200, min_epochs_for_best=200, lr_scheduler="none"),
    "apr08":        dict(epochs=500,  weight_decay=0.0001, warmup_epochs=200, min_epochs_for_best=200, lr_scheduler="cosine"),
    "warmup50":     dict(epochs=500,  weight_decay=0.0001, warmup_epochs=50,  min_epochs_for_best=50,  lr_scheduler="none"),
    "warmup100":    dict(epochs=500,  weight_decay=0.0001, warmup_epochs=100, min_epochs_for_best=100, lr_scheduler="none"),
    "0324_nowd":    dict(epochs=500,  weight_decay=0.0,    warmup_epochs=0,   min_epochs_for_best=200, lr_scheduler="none"),
    "mar30_nowd":   dict(epochs=500,  weight_decay=0.0,    warmup_epochs=200, min_epochs_for_best=200, lr_scheduler="none"),
    "apr08_nowd":   dict(epochs=500,  weight_decay=0.0,    warmup_epochs=200, min_epochs_for_best=200, lr_scheduler="cosine"),
    "warmup50_nowd":  dict(epochs=500, weight_decay=0.0,   warmup_epochs=50,  min_epochs_for_best=50,  lr_scheduler="none"),
    "warmup100_nowd": dict(epochs=500, weight_decay=0.0,   warmup_epochs=100, min_epochs_for_best=100, lr_scheduler="none"),
}

LATENT_DIM = 12
PROJ_DIM   = 8
COMBO      = f"lat{LATENT_DIM}proj{PROJ_DIM}"
ROOT_FOLDER = "/net/projects/CLS/lding/data/fa_data_analysis"
LABEL_CSV   = f"{ROOT_FOLDER}/labelling/labels_vinc_20260521.csv"


def ae_config(strategy: str, p: dict) -> str:
    run_dir = f"contrastive_cio_rb_vinc_{COMBO}_{strategy}"
    warmup_note = (f"warmup {p['warmup_epochs']} epochs (recon-only) then contrastive, "
                   if p["warmup_epochs"] > 0 else "no warmup, ")
    wd_note = f"weight_decay={p['weight_decay']}"
    ckpt_note = (f"best ckpt after ep {p['min_epochs_for_best']}"
                 if p["min_epochs_for_best"] > 0 else "final model")
    sched_note = f"lr_scheduler={p['lr_scheduler']}"
    return f"""\
# =============================================================================
# AE variant: contrastive  (NT-Xent contrastive AE)
# Normalization: CIO-RB  |  Dataset: vinc (vinculin channel) control + ycomp
# latent_dim={LATENT_DIM}, proj_dim={PROJ_DIM}  →  projector: {LATENT_DIM}→{LATENT_DIM*4}→{PROJ_DIM}
# Training strategy: {strategy} — {p['epochs']} epochs, {wd_note},
#   {warmup_note}{ckpt_note}, {sched_note}
# Results: ae_results/contrastive_run/{run_dir}/
# =============================================================================
root_folder : "{ROOT_FOLDER}"

data:
  patch_dirs:
    - path           : root_folder + "/ae_results/pax_ch_patch/cio_rb/vinc/control/tiff_patches32"
      condition      : 0
      condition_name : "control"
    - path           : root_folder + "/ae_results/pax_ch_patch/cio_rb/vinc/ycomp/tiff_patches32"
      condition      : 1
      condition_name : "ycomp"

output:
  result_dir : root_folder + "/ae_results/contrastive_run/{run_dir}"

model:
  model_type   : "contrastive"
  latent_dim   : {LATENT_DIM}
  input_ps     : 32
  no_ch        : 1
  BN_flag      : false
  dropout_flag : false

  proj_dim              : {PROJ_DIM}
  noise_prob            : 0.05
  temperature           : 0.5
  lambda_recon          : 1.0
  lambda_contrast       : 0.5
  intensity_scale_range : [0.8, 1.2]

training:
  epochs                  : {p['epochs']}
  lr                      : 0.001
  batch_size              : 128
  val_split               : 0.2
  loss_norm_flag          : false
  group_split             : true
  weight_decay            : {p['weight_decay']}
  warmup_epochs           : {p['warmup_epochs']}
  lr_scheduler            : "{p['lr_scheduler']}"
  early_stopping_patience : 0
  min_epochs_for_best     : {p['min_epochs_for_best']}

reconstruction:
  save_recon      : true
  recon_pad_size  : 64
  recon_image_size: 1024

misc:
  device    : "auto"
  log_level : "INFO"
"""


def analysis_config(strategy: str) -> str:
    run_dir = f"contrastive_cio_rb_vinc_{COMBO}_{strategy}"
    return f"""\
# =============================================================================
# Analysis: contrastive AE, CIO-RB, vinc dataset, latent_dim={LATENT_DIM}, proj_dim={PROJ_DIM}
# Training strategy: {strategy}
# =============================================================================
root_folder : "{ROOT_FOLDER}"

input:
  latents_csv  : root_folder + "/ae_results/contrastive_run/{run_dir}/latents.csv"
  split_filter : "all"

output:
  out_dir : root_folder + "/ae_results/contrastive_run/{run_dir}/analysis"

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
  dbscan_eps        : 0.5
  dbscan_min_samples: 10
  boxplot_kind      : box

label_orders:
  annotation_label_name:
    - "Nascent Adhesion"
    - "focal complex"
    - "focal adhesion"
    - "fibrillar adhesion"
    - "No adhesion"
  condition_name:
    - "control"
    - "ycomp"

misc:
  log_level : "INFO"
"""


def cls_fa_config(strategy: str, z_type: str) -> str:
    """z_type: 'zrecon' or 'zproj'"""
    run_dir    = f"contrastive_cio_rb_vinc_{COMBO}_{strategy}"
    prefix     = "z_" if z_type == "zrecon" else "p_"
    out_subdir = f"fa_cls_{z_type}"
    return f"""\
root_folder : "{ROOT_FOLDER}"

input:
  latents_csv : root_folder + "/ae_results/contrastive_run/{run_dir}/latents.csv"

output:
  out_dir : root_folder + "/ae_results/contrastive_run/{run_dir}/{out_subdir}"

labels:
  label_col    : "classification"
  label_csv    : "{LABEL_CSV}"
  filename_col : "unique_ID"
  label_order  :
    - "Nascent Adhesion"
    - "focal complex"
    - "focal adhesion"
    - "fibrillar adhesion"
    - "No adhesion"
  exclude_labels:
    - "Uncertain"
  metrics_exclude_labels:
    - "No adhesion"

features:
  feature_cols   : null
  feature_prefix : "{prefix}"
  include_mean_intensity: false

split:
  strategy    : "from_csv"
  test_size   : 0.2
  random_state: 42

classifier:
  type : "svm"

svm:
  C     : 10.0
  gamma : "scale"

dist_features:
  patch_prep_dirs: null
  feature_weight : 20.0

patch_sort:
  sort_labelled  : true
  sort_unlabelled: false

misc:
  log_level : "INFO"
"""


def cls_pos_config(strategy: str, z_type: str) -> str:
    run_dir    = f"contrastive_cio_rb_vinc_{COMBO}_{strategy}"
    prefix     = "z_" if z_type == "zrecon" else "p_"
    out_subdir = f"pos_cls_{z_type}"
    return f"""\
root_folder : "{ROOT_FOLDER}"

input:
  latents_csv : root_folder + "/ae_results/contrastive_run/{run_dir}/latents.csv"

output:
  out_dir : root_folder + "/ae_results/contrastive_run/{run_dir}/{out_subdir}"

labels:
  label_col    : "Position"
  label_csv    : "{LABEL_CSV}"
  filename_col : "unique_ID"
  label_order  :
    - "Cell Protruding Edge"
    - "Cell Periphery/other"
    - "Lamella"
    - "Cell Body"
  exclude_labels:
    - "No Category/uncertain"
  metrics_exclude_labels: null

features:
  feature_cols   : null
  feature_prefix : "{prefix}"
  include_mean_intensity: false

split:
  strategy    : "from_csv"
  test_size   : 0.2
  random_state: 42

classifier:
  type : "svm"

svm:
  C     : 10.0
  gamma : "scale"

dist_features:
  patch_prep_dirs: null
  feature_weight : 20.0

patch_sort:
  sort_labelled  : true
  sort_unlabelled: false

misc:
  log_level : "INFO"
"""


def vis_config(strategy: str, z_type: str) -> str:
    run_dir    = f"contrastive_cio_rb_vinc_{COMBO}_{strategy}"
    fa_subdir  = f"fa_cls_{z_type}"
    pos_subdir = f"pos_cls_{z_type}"
    vis_subdir = f"vis_{z_type}"
    return f"""\
root_folder : "{ROOT_FOLDER}"

input:
  latents_csv          : root_folder + "/ae_results/contrastive_run/{run_dir}/latents.csv"
  fa_type_results_csv  : root_folder + "/ae_results/contrastive_run/{run_dir}/{fa_subdir}/predictions_all.csv"
  position_results_csv : root_folder + "/ae_results/contrastive_run/{run_dir}/{pos_subdir}/predictions_all.csv"
  umap_model_pkl       : root_folder + "/ae_results/contrastive_run/{run_dir}/{fa_subdir}/umap_all_model.pkl"

output:
  out_dir : root_folder + "/ae_results/contrastive_run/{run_dir}/{vis_subdir}"

labels:
  fa_type_label_col  : "classification"
  position_label_col : "Position"
  fa_type_order:
    - "Nascent Adhesion"
    - "focal complex"
    - "focal adhesion"
    - "fibrillar adhesion"
    - "No adhesion"
  position_order:
    - "Cell Protruding Edge"
    - "Cell Periphery/other"
    - "Lamella"
    - "Cell Body"

misc:
  random_state : 42
  log_level    : "INFO"
"""


def main():
    generated = []
    for strategy, params in STRATEGIES.items():
        if strategy == "0322":
            print(f"  skip {strategy} (already exists)")
            continue

        base = f"contrastive_cio_rb_vinc_{COMBO}_{strategy}"

        files = {
            f"ae_{base}.yaml":                    ae_config(strategy, params),
            f"analysis_{base}.yaml":              analysis_config(strategy),
            f"cls_{base}_fa_zrecon.yaml":         cls_fa_config(strategy, "zrecon"),
            f"cls_{base}_fa_zproj.yaml":          cls_fa_config(strategy, "zproj"),
            f"cls_{base}_pos_zrecon.yaml":        cls_pos_config(strategy, "zrecon"),
            f"cls_{base}_pos_zproj.yaml":         cls_pos_config(strategy, "zproj"),
            f"vis_{base}_zrecon.yaml":            vis_config(strategy, "zrecon"),
            f"vis_{base}_zproj.yaml":             vis_config(strategy, "zproj"),
        }

        for fname, content in files.items():
            path = OUT / fname
            path.write_text(content)
            generated.append(fname)

    print(f"\nGenerated {len(generated)} configs in {OUT}")
    for f in generated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
