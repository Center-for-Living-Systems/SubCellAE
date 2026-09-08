# Local GPU Handoff — CNN Label-Efficiency Benchmark

**Date:** 2026-09-08  
**Purpose:** Run `eval_cnn_le_b1b2.py` (EfficientNet-B0 direct patch classifier) on local GPU to generate CNN label-efficiency curves for the B1/B2 matched benchmark.

---

## Data already copied

All required data is at:

```
/home/lding/lding/dsicluster_CLS_rsync_folder/localgpuattempt/
├── ae_results/
│   └── patches/
│       └── cio/
│           └── vinc/
│               ├── control/
│               │   └── tiff_patches32_mr10/    ← 116 MB, 14,879 files
│               └── ycomp/
│                   └── tiff_patches32_mr10/    ←  96 MB, ~12k files
└── labelling/
    ├── le_b1b2_le/          ← per-budget annotation CSVs (B1 and B2, all folds/repeats)
    └── le_b1b2_matched/     ← fold_splits_b1.csv, fold_splits_b2.csv
```

Patches are **32×32 px float32 TIFF** (already normalized), no CZI preprocessing needed.

---

## Environment setup

```bash
conda create -n subcellae_local python=3.11 -y
conda activate subcellae_local
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # adjust for your CUDA
pip install timm tifffile pandas scikit-learn
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Running the CNN benchmark

Clone or copy the script from the cluster:

```
scripts/eval_cnn_le_b1b2.py
```

### Full benchmark — B1 labels (205 jobs, ~30–60 min on GPU)

```bash
python scripts/eval_cnn_le_b1b2.py \
    --batch b1 \
    --data-dir /home/lding/lding/dsicluster_CLS_rsync_folder/localgpuattempt
```

### Full benchmark — B2 labels

```bash
python scripts/eval_cnn_le_b1b2.py \
    --batch b2 \
    --data-dir /home/lding/lding/dsicluster_CLS_rsync_folder/localgpuattempt
```

### Single fold/budget quick test

```bash
python scripts/eval_cnn_le_b1b2.py \
    --batch b1 --fold 0 --budget 50 --repeat 0 \
    --data-dir /home/lding/lding/dsicluster_CLS_rsync_folder/localgpuattempt
```

---

## Output

Results are saved to:

```
{data-dir}/ae_results/features/eval_results/
├── cnn_b1b2_b1_ds1.csv
└── cnn_b1b2_b2_ds1.csv
```

Each CSV has columns: `name, batch, fold, budget, repeat, n_train, n_test, bal_acc`

If you re-run a partial result (e.g. re-running a single fold), the script **appends and deduplicates** rather than overwriting, so partial runs are safe.

---

## Copy results back to cluster

```bash
rsync -avz \
    /home/lding/lding/dsicluster_CLS_rsync_folder/localgpuattempt/ae_results/features/eval_results/cnn_b1b2_*.csv \
    liyading@cluster:/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/eval_results/
```

Once the CSVs are back on the cluster, regenerate the PPT:

```bash
python scripts/make_pptx_le_benchmark.py
```

Slide 22d will automatically switch from the "pending" watermark to real CNN curves.

---

## Model details

| Setting | Value |
|---|---|
| Architecture | EfficientNet-B0 (timm, ImageNet pretrained) |
| Input channels | 1 (first conv weight averaged 3→1) |
| Input size | 32×32 upsampled to 64×64 (nearest neighbour) |
| Epochs | 100 |
| Optimizer | Adam, lr=5×10⁻⁴, weight_decay=1×10⁻⁴ |
| LR schedule | Cosine annealing |
| Augmentation | Random H/V flip, random rot90 |
| Loss | CrossEntropyLoss with inverse-frequency class weights |
| Batch size | min(64, n_train) |
| Workers | 2 (set `NUM_WORKERS=0` if DataLoader hangs) |

---

## Benchmark structure

- **Batches:** B1 (Margaret labels), B2 (Annabel labels)
- **Same images:** 327 matched patches per batch (vinc ctrl + ycomp, DS1)
- **Budgets:** 10, 20, 25, 50, 75, 100, 150, 200 (+ "all" budget, repeat=0 only)
- **CV:** 5 folds × 5 repeats per numeric budget → 205 jobs per batch
- **Test set:** held-out fold (excl. training patches), same across all methods
- **Metric:** balanced accuracy (handles class imbalance)

This is a direct apple-to-apple comparison with the CellProfiler + LGBM and ilastik + LGBM curves already computed on the cluster.

---

## Troubleshooting

**DataLoader deadlock:** Set `NUM_WORKERS = 0` at the top of `eval_cnn_le_b1b2.py`.

**CUDA OOM:** Reduce `BATCH_SIZE` (currently 64) or `IMG_SIZE` (currently 64).

**HuggingFace warning about unauthenticated requests:** Harmless — weights are cached after first download in `~/.cache/huggingface/hub/`.

**Missing annotation CSV for a budget/fold/repeat:** The job is silently skipped (`n_skip` counter). This is expected if you only copied a subset of `le_b1b2_le/`.
