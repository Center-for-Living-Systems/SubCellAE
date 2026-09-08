# Multiscale B2 DS1 — Patch Size & Latent Dim Sweep

Experiment comparing 4 SupCon-AE configurations on DS1 B2 focal adhesion data,
varying patch size (32 vs 64 px) and latent dimensionality (lat=12/proj=8 vs lat=64/proj=32).

---

## Goal

Determine whether larger input patches and/or higher-dimensional latent spaces improve
binary No-adhesion vs adhesion classification on the DS1 B2 (Annabel, vinc ctrl+ycomp)
dataset under 5-fold cross-validation.

---

## Dataset

- **Annotator:** Annabel (vinc dataset)
- **Conditions:** control (ctrl) + ycomp
- **Source:** `vinc_combined_label_Annabel_20260816.csv`
- **Total patches:** 1,224 (after filtering "Uncertain")
  - No adhesion: 770 (62.9 %)
  - adhesion: 454 (37.1 %)  ← NA + FC + FA + Fib collapsed to one class
  - control: 539 · ycomp: 685
- **Folds:** 5-fold, patch-level random split (seed=42), ~245 patches per fold

### Fold splits

All 4 configurations share the **same fold splits** (generated once, reused):

```
/net/projects/CLS/lding/data/fa_data_analysis/labelling/ms_b2_ds1_ps64_lat64p32/fold_splits.csv
```

Per-config train CSVs (pointing into that shared fold_splits) live under:

```
/net/projects/CLS/lding/data/fa_data_analysis/labelling/{run_tag}/
```

### Source frames

Patches are cropped **online** at training time from full-resolution TIFFs via `CoordCropDataset`:

```
/net/projects/CLS/lding/data/fa_data_analysis/ae_results/source_frames/cio_mode_prt/vinc/control/
/net/projects/CLS/lding/data/fa_data_analysis/ae_results/source_frames/cio_mode_prt/vinc/ycomp/
```

Frame filename pattern: `{cond}_f{NNNN}_{channel}.tif`

---

## 2×2 Configuration Grid

| run_tag | patch_size | latent_dim | proj_dim | status |
|---|---|---|---|---|
| `ms_b2_ds1_ps32_lat12p8`  | 32 | 12 | 8  | PENDING (SLURM 1609171) |
| `ms_b2_ds1_ps32_lat64p32` | 32 | 64 | 32 | PENDING (SLURM 1609170) |
| `ms_b2_ds1_ps64_lat12p8`  | 64 | 12 | 8  | PENDING (SLURM 1609300) |
| `ms_b2_ds1_ps64_lat64p32` | 64 | 64 | 32 | **DONE** — bal_acc 0.961 ± 0.011 |

Common training settings: epochs=500, lr=0.001, batch=128, val_split=0.2,
weight_decay=1e-4, group_split=True, recon_loss=nl1, λ_recon=1.0, λ_contrast=0.5,
λ_supcon=5.0, temperature=0.5, intensity_scale=[0.8, 1.2].

---

## Completed results

### ps=64, lat=64, proj=32

| fold | train | test | bal_acc |
|---|---|---|---|
| fv0 | ~979 | 245 | 0.9483 |
| fv1 | ~979 | 245 | 0.9786 |
| fv2 | ~979 | 245 | 0.9675 |
| fv3 | ~980 | 244 | 0.9601 |
| fv4 | ~979 | 245 | 0.9524 |
| **mean ± std** | | | **0.9614 ± 0.0109** |

Classifier: LightGBM (n_estimators=300, class_weight=balanced).
Train latents from `latents.csv`; test latents via model inference on held-out fold.

---

## File inventory

### Setup scripts

| script | purpose |
|---|---|
| `scripts/setup_ms_b2_ds1_ps64.py` | Generate fold splits + YAML configs for all 4 runs from the B2 label CSV |

Generates output under `labelling/{run_tag}/` (data dir, not tracked) and `config/{run_tag}/` (tracked).

### Training

| file | purpose |
|---|---|
| `scripts/run_ae_from_config.py` | Main AE training loop (existing) |
| `scripts/sbatch_ms_b2_ds1_ps64.sh` | SLURM array script (a40 GPU, 8 h, 32 GB) |
| `config/ms_b2_ds1_ps{32,64}_lat{12p8,64p32}/` | 5 YAML configs + job_list.txt per run |

### Evaluation

| script | purpose |
|---|---|
| `scripts/eval_ms_b2_ds1.py` | 5-fold LightGBM eval; loads latents.csv for train, runs model for test |

### Visualisation

| script | output |
|---|---|
| `scripts/make_pptx_ms_b2_ds1_ps64.py` | `results/ms_b2_ds1_ps64.pptx` — 7 slides |

### Dataset pipeline fix

`subcellae/modelling/dataset.py` — `CoordCropDataset.__init__` was missing attributes
expected by `ae_pipeline.py`:
- `self.condition_name`
- `self.label_order_2`, `self.num_classes_2`, `self.label_to_int_2`

---

## Running on a local GPU workstation

The YAML configs use `device: "auto"` so they will pick up any available CUDA GPU.
Clone/pull the repo on the workstation, then for each pending fold:

```bash
# set up environment
export PYTHONPATH=/path/to/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python3.11  # or local venv

# run one fold (replace RUN_TAG and k as needed)
RUN_TAG=ms_b2_ds1_ps32_lat12p8
$PYTHON scripts/run_ae_from_config.py config/${RUN_TAG}/${RUN_TAG}_fv{k}.yaml
```

Or loop over all 5 folds for a given run tag:

```bash
RUN_TAG=ms_b2_ds1_ps32_lat12p8
for k in 0 1 2 3 4; do
    $PYTHON scripts/run_ae_from_config.py config/${RUN_TAG}/${RUN_TAG}_fv${k}.yaml
done
```

**Data paths are hard-coded** to `/net/projects/CLS/lding/data/fa_data_analysis/...`.
If running on a machine where that NFS path is not mounted, either mount it or update
`root_folder` in the YAML.

After all 5 folds of a run complete:

```bash
# evaluate (adjust --patch-size / --latent-dim / --proj-dim for each run_tag)
$PYTHON scripts/eval_ms_b2_ds1.py \
    --run-tag ms_b2_ds1_ps32_lat12p8 \
    --patch-size 32 --latent-dim 12 --proj-dim 8 --device cuda
```

Eval flags for each run:

| run_tag | --patch-size | --latent-dim | --proj-dim |
|---|---|---|---|
| ms_b2_ds1_ps32_lat12p8  | 32 | 12 | 8  |
| ms_b2_ds1_ps32_lat64p32 | 32 | 64 | 32 |
| ms_b2_ds1_ps64_lat12p8  | 64 | 12 | 8  |
| ms_b2_ds1_ps64_lat64p32 | 64 | 64 | 32 |

---

## PPT slides (`results/ms_b2_ds1_ps64.pptx`)

1. **Dataset overview** — fold breakdown, label & condition distribution
2. **Example patches** — same FA/NoAD patches shown at ps=32 and ps=64 side-by-side
3. **Experiment setup** — 2×2 config table with training hyper-parameters
4. **Classification results** — 5-fold balanced accuracy per config (placeholders for pending runs)
5. **Reconstruction** — input vs reconstructed patches from ps=64 lat=64 model (fv0)
6. **UMAP (2-class + condition)** — 3 000 training latents coloured by label / condition
7. **UMAP (5-class FA subtypes)** — same embedding coloured by original 5-class labels

Rebuild after pending runs finish:

```bash
$PYTHON scripts/make_pptx_ms_b2_ds1_ps64.py
```

---

## Result directories

```
/net/projects/CLS/lding/data/fa_data_analysis/ae_results/multiscale/
├── ms_b2_ds1_ps64_lat64p32/          ← DONE
│   ├── ms_b2_ds1_ps64_lat64p32_fv{0-4}/
│   │   ├── model_best.pt
│   │   ├── latents.csv
│   │   └── ...
├── ms_b2_ds1_ps32_lat12p8/           ← pending
├── ms_b2_ds1_ps32_lat64p32/          ← pending
└── ms_b2_ds1_ps64_lat12p8/           ← pending
```
