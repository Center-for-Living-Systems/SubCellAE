"""
Generate config files for the contrastive AE latent/proj dim sweep.

Combos: latent_dim × {12, 16, 24}  ×  proj_dim × {8, 12}
Baseline (lat12proj8) already exists — skipped here.

Output: config/contrastive_config/  (ae, analysis, cls ×4, vis ×2 per combo)
"""
from pathlib import Path

ROOT_FOLDER = "/net/projects/CLS/lding/data/fa_data_analysis"
REPO_ROOT = Path(__file__).parent.parent
OUT_DIR = REPO_ROOT / "config" / "contrastive_config"

LATENT_DIMS = [12, 16, 24, 32]
PROJ_DIMS = [8, 12]
BASELINE = (12, 8)  # already exists


def run_name(lat, proj):
    return f"contrastive_cio_rb_vinc_lat{lat}proj{proj}"


def write(path, content):
    path.write_text(content)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


for lat in LATENT_DIMS:
    for proj in PROJ_DIMS:
        if (lat, proj) == BASELINE:
            continue

        name = run_name(lat, proj)
        run_dir = f"ae_results/contrastive_run/{name}"

        # ── AE training ──────────────────────────────────────────────────────
        write(OUT_DIR / f"ae_{name}.yaml", f"""\
# =============================================================================
# AE variant: contrastive  (NT-Xent contrastive AE)
# Normalization: CIO-RB  |  Dataset: vinc (vinculin channel) control + ycomp
# latent_dim={lat}, proj_dim={proj}  →  projector: {lat}→{lat*4}→{proj}
# Results: ae_results/contrastive_run/{name}/
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
  result_dir : root_folder + "/{run_dir}"

model:
  model_type   : "contrastive"
  latent_dim   : {lat}
  input_ps     : 32
  no_ch        : 1
  BN_flag      : false
  dropout_flag : false

  proj_dim              : {proj}
  noise_prob            : 0.05
  temperature           : 0.5
  lambda_recon          : 1.0
  lambda_contrast       : 0.5
  intensity_scale_range : [0.8, 1.2]

training:
  epochs         : 500
  lr             : 0.001
  batch_size     : 128
  val_split      : 0.2
  loss_norm_flag : false
  group_split    : true

reconstruction:
  save_recon      : true
  recon_pad_size  : 64
  recon_image_size: 1024

misc:
  device    : "auto"
  log_level : "INFO"
""")

        # ── Analysis ─────────────────────────────────────────────────────────
        write(OUT_DIR / f"analysis_{name}.yaml", f"""\
# =============================================================================
# Analysis: contrastive AE, CIO-RB, vinc dataset, latent_dim={lat}, proj_dim={proj}
# =============================================================================
root_folder : "{ROOT_FOLDER}"

input:
  latents_csv  : root_folder + "/{run_dir}/latents.csv"
  split_filter : "all"

output:
  out_dir : root_folder + "/{run_dir}/analysis"

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
""")

        # ── Classification (4 configs: fa/pos × zrecon/zproj) ────────────────
        for target, label_col, label_order in [
            ("fa", "classification", [
                "Nascent Adhesion", "focal complex", "focal adhesion",
                "fibrillar adhesion", "No adhesion"]),
            ("pos", "Position", [
                "Cell Protruding Edge", "Cell Periphery/other",
                "Lamella", "Cell Body"]),
        ]:
            label_order_yaml = "\n".join(f'    - "{l}"' for l in label_order)
            for feat, prefix in [("zrecon", "z_"), ("zproj", "p_")]:
                write(OUT_DIR / f"cls_{name}_{target}_{feat}.yaml", f"""\
root_folder : "{ROOT_FOLDER}"

input:
  latents_csv : root_folder + "/{run_dir}/latents.csv"

output:
  out_dir : root_folder + "/{run_dir}/{target}_cls_{feat}"

labels:
  label_col    : "{label_col}"
  label_csv    : "{ROOT_FOLDER}/labelling/labels_vinc_20260521.csv"
  filename_col : "unique_ID"
  label_order  :
{label_order_yaml}
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
""")

        # ── Visualization (2 configs: zrecon / zproj) ────────────────────────
        fa_order = "\n".join(f'    - "{l}"' for l in [
            "Nascent Adhesion", "focal complex", "focal adhesion",
            "fibrillar adhesion", "No adhesion"])
        pos_order = "\n".join(f'    - "{l}"' for l in [
            "Cell Protruding Edge", "Cell Periphery/other",
            "Lamella", "Cell Body"])

        for feat in ["zrecon", "zproj"]:
            write(OUT_DIR / f"vis_{name}_{feat}.yaml", f"""\
root_folder : "{ROOT_FOLDER}"

input:
  latents_csv          : root_folder + "/{run_dir}/latents.csv"
  fa_type_results_csv  : root_folder + "/{run_dir}/fa_cls_{feat}/predictions_all.csv"
  position_results_csv : root_folder + "/{run_dir}/pos_cls_{feat}/predictions_all.csv"
  umap_model_pkl       : root_folder + "/{run_dir}/fa_cls_{feat}/umap_all_model.pkl"

output:
  out_dir : root_folder + "/{run_dir}/vis_{feat}"

labels:
  fa_type_label_col  : "classification"
  position_label_col : "Position"
  fa_type_order:
{fa_order}
  position_order:
{pos_order}

misc:
  random_state : 42
  log_level    : "INFO"
""")

print("Done.")
