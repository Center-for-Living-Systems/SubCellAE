# Label-Efficiency Benchmark Overview

**Project:** SubCellAE — Focal Adhesion Detection  
**Last updated:** 2026-09-08

---

## Goal

Quantify how many labeled patches are needed to train a reliable focal adhesion (FA) classifier (adhesion vs. no adhesion), and measure how much SupCon-AE's learned latent representation reduces that labeling requirement relative to hand-engineered feature baselines.

Two orthogonal questions drive the experiments:

1. **Label efficiency:** For a fixed image dataset, how does classification performance scale with the number of labeled patches?
2. **Image efficiency:** For a fixed annotation budget, does training the representation on *more* images improve downstream classification?

---

## Methods compared

| Method | Features | Classifier | Uses unlabeled data? |
|---|---|---|---|
| CellProfiler | 56-dim (intensity + Haralick GLCM) | LightGBM (LGBM) or logistic regression | No |
| ilastik | Pixel feature stack (Gaussian, LoG, Hessian, gradient, structure tensor) | LGBM or logreg | No |
| **SupCon-AE** | Latent z (12- or 64-dim) from supervised contrastive autoencoder | LGBM or logreg | **Yes** (all patches) |
| EfficientNet-B0 | Raw pixels (32×32, upsampled to 64×64) | End-to-end fine-tuning | No |

SupCon-AE is the key method: it trains on **all** available patches (labeled + unlabeled, ~14k–27k) using a combined reconstruction loss and supervised contrastive loss, then the frozen latent vectors are classified by LGBM. This lets it leverage unlabeled data, which is the hypothesized advantage at low label budgets.

---

## Datasets

| Tag | Drug | Channel | Conditions | Patches |
|---|---|---|---|---|
| DS1 | vincristine (vinc) | ch0 = vinculin (PAX) | ctrl, ycomp | ~14k ctrl + ~13k ycomp |
| DS2 | paxillin antibody (pfak) | ch0 = pFAK | ctrl, ycomp | ~7k ctrl + ~1.4k ycomp |

Patches are 32×32 px, extracted with 10 px minimum inter-patch spacing (mr10).

---

## Annotation sets

### B2 — Annabel (primary, high quality)

- DS1: 1,224 patches (770 no-adhesion, 454 adhesion) across 18 images (ctrl + ycomp)
- DS2: 244 patches across 8 images
- Collected August 2026 using the systematic random-sampling protocol (`FA_Random_Sampling_Labeling_Protocol.md`)

### B1 — Margaret (earlier, lower consistency)

- DS1: 327 patches (same images as B2, matched subset)
- Collected ~June 2026 with an earlier less-structured protocol
- Labels are noisier: at matched budget and identical features, B1 gives **~0.17–0.20 lower balanced accuracy** than B2

### B12 combined

- DS1 B1 + B2 merged (~2,428 patches after dropping 53 conflicts), B2 takes priority on disagreements
- Used to test whether more annotation diversity helps SupCon-AE

### B1/B2 matched set

- Exact same 327 patches relabelled by both annotators
- Used to isolate the effect of **label quality alone** (same images, same features, different annotators)

---

## Experiment series

### Series 1 — DS1 B2 baseline (`le_b2_supcon`, `le_b2_lat12p8`)

Single annotator (Annabel), DS1 only.  
Budgets: 10, 20, 25, 50, 75, 100, 150, 200, 300, 500, 750.  
Result: SupCon-AE trails CellProfiler at low budgets but approaches it above ~300 labels.

### Series 2 — DS1 B12 combined (`le_b12_ds1_lat12p8`, `le_b12_ds1_lat64p32`)

B1+B2 merged annotation pool, DS1.  
Budgets: 10–1500 (15 levels).  
Latent sizes: 12/proj=8 and 64/proj=32.  
Result: Adding B1 labels to the pool does not meaningfully improve SupCon-AE's LE curve. CP and ilastik remain the strongest methods at low budgets.

### Series 3 — DS2 B12 (`le_b12_ds2_lat64p32`)

B12 labels on pFAK channel (DS2).  
Budgets: 10–150 (fewer patches available).  
Result: Similar pattern to DS1; CP competitive with SupCon-AE.

### Series 4 — B1 vs B2 label quality (`le_b1b2_{b1|b2}_{lat12p8|lat64p32}`)

**Key question:** Is the performance gap between B1 and B2 due to label quality, or image/feature differences?

Experimental design:
- Same 327 matched patches, different labels (B1 vs B2)
- Same features (CP, ilastik), same LGBM classifier, same CV protocol
- Budgets: 10, 20, 25, 50, 75, 100, 150, 200

**Results (completed, CP + ilastik):**

| Budget | CP B1 | CP B2 | Gap | IL B1 | IL B2 | Gap |
|---|---|---|---|---|---|---|
| 10  | 0.669 | 0.843 | +0.174 | 0.635 | 0.749 | +0.114 |
| 50  | 0.712 | 0.931 | +0.219 | 0.681 | 0.831 | +0.150 |
| 100 | 0.734 | 0.934 | +0.200 | 0.712 | 0.870 | +0.158 |
| 200 | 0.760 | 0.947 | +0.187 | 0.725 | 0.897 | +0.172 |

**Conclusion: label quality dominates at every budget.** This is entirely due to annotation consistency — B2 labels are dramatically better.

SupCon-AE jobs for B1/B2 matched set are in progress on the cluster (SLURM jobs 1606630, 1609145).

### Series 5 — CNN direct classifier (`eval_cnn_le_b1b2.py`)

**Question:** How does a direct end-to-end CNN (EfficientNet-B0, no unsupervised pre-training) compare to feature-based methods at different budgets?

- Only uses labeled patches (unlike SupCon-AE which uses all ~27k)
- 100 epochs, ImageNet pretrained weights, 1-channel adapted
- SLURM jobs 1609536 (B1) and 1609537 (B2) submitted; also runnable locally (see `local_gpu_handoff.md`)
- Results will show whether handcrafted features (CP, ilastik) can be beaten by raw pixels + CNN at high budgets

---

## Evaluation protocol

All series use the same protocol for fair comparison:

- **Cross-validation:** 5-fold stratified split on labeled patches
- **Repeats:** 5 random subsamples per budget (except `budget=all`, repeat=0 only)
- **Test set:** held-out fold, with training patches excluded from test
- **Metric:** balanced accuracy (handles adhesion/no-adhesion class imbalance)
- **LGBM params:** n_estimators=200, num_leaves=31, lr=0.05, class_weight=balanced
- **Logreg params:** max_iter=2000, class_weight=balanced, StandardScaler

---

## File map

### Scripts

| Script | Purpose |
|---|---|
| `scripts/setup_le_b12_ds1_lat12p8.py` | Generate YAML configs for DS1 B12 lat=12 SupCon jobs |
| `scripts/setup_le_b12_ds1_lat64p32.py` | Same for lat=64 |
| `scripts/setup_le_b1b2_le.py` | Generate annotation CSVs + configs for B1/B2 matched benchmark |
| `scripts/sbatch_le_b1b2_le.sh` | SLURM array script for B1/B2 SupCon jobs |
| `scripts/sbatch_cnn_le_b1b2.sh` | SLURM job to run full CNN benchmark for one batch |
| `scripts/eval_supcon_latents.py` | Extract latents and evaluate all SupCon-AE runs |
| `scripts/eval_le_b1b2_handcrafted.py` | Evaluate CP and ilastik on B1/B2 matched splits |
| `scripts/eval_cnn_le_b1b2.py` | EfficientNet-B0 direct classifier on B1/B2 matched splits |
| `scripts/make_pptx_le_benchmark.py` | Build `results/le_benchmark.pptx` from all eval CSVs |

### Config directories

| Dir | Description |
|---|---|
| `config/le_b12_ds1_lat12p8/` | 380 YAML jobs, `job_list_ds1.txt` |
| `config/le_b12_ds1_lat64p32/` | 380 YAML jobs, `job_list_ds1.txt` |
| `config/le_b1b2_b1_lat12p8/` | 205 YAML jobs for B1 lat=12 SupCon |
| `config/le_b1b2_b2_lat12p8/` | 205 YAML jobs for B2 lat=12 SupCon |
| `config/le_b1b2_b1_lat64p32/` | 205 YAML jobs for B1 lat=64 SupCon |
| `config/le_b1b2_b2_lat64p32/` | 205 YAML jobs for B2 lat=64 SupCon |

### Eval result CSVs (on cluster, `ae_results/features/eval_results/`)

| CSV | Status |
|---|---|
| `cp_b1b2_b1_ds1.csv`, `cp_b1b2_b2_ds1.csv` | ✓ Complete |
| `il_b1b2_b1_ds1.csv`, `il_b1b2_b2_ds1.csv` | ✓ Complete |
| `supcon_le_b12_ds1_lat12p8_ds1.csv` | ✓ Complete |
| `supcon_le_b12_ds1_lat64p32_ds1.csv` | ✓ Complete |
| `cp_b12_ds1.csv`, `ilastik_b12_ds1.csv` | ✓ Complete (budgets 10–1500) |
| `cnn_b1b2_b1_ds1.csv`, `cnn_b1b2_b2_ds1.csv` | ⏳ GPU jobs running |
| `supcon_le_b1b2_b1_*.csv`, `supcon_le_b1b2_b2_*.csv` | ⏳ SLURM jobs running |

---

## PPT output

`results/le_benchmark.pptx` — built by `make_pptx_le_benchmark.py`

Key slides for the B1/B2 label quality story:

- **Slide 22c** — B1 vs B2 label quality: 2-panel (CP left, ilastik right), B1 dashed vs B2 solid, gap annotated
- **Slide 22d** — CNN vs feature-based: EfficientNet-B0 direct vs CP+LGBM vs ilastik+LGBM (auto-updates when CNN CSVs arrive)

To rebuild after new results arrive:

```bash
python scripts/make_pptx_le_benchmark.py
```
