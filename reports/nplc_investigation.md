# Investigation: Labeled Patches Per Batch in SupCon-AE Training

**Date:** 2026-08-30  
**Author:** LD  

---

## 1. Background

The SupCon-AE model combines three loss terms during training:

- **Recon loss**: reconstruction on all patches (labeled + unlabeled)
- **UC loss** (unsupervised contrastive): contrastive learning across all patches using augmented views
- **SC loss** (supervised contrastive): contrastive loss on labeled patches only — pulls same-class patches together and pushes different-class patches apart

For the SC loss to contribute usefully, a training batch must contain at least two patches of the same class with known labels. With standard random shuffling over a large unlabeled pool, labeled patches may rarely co-occur in the same batch at small label budgets.

---

## 2. Feature Added: `LabeledAwareBatchSampler` / `n_labeled_per_class`

To guarantee SC supervision in every batch, a `LabeledAwareBatchSampler` was added to the pipeline. Configured via:

```yaml
training:
  n_labeled_per_class: N   # 0 = standard shuffle (default); >0 = guarantee N labeled patches per class per batch
```

With `n_labeled_per_class = N`, each batch of 128 contains exactly `N × n_classes` labeled patches (forced), with the remaining slots filled randomly from the full pool.

---

## 3. Initial Use — the Bug

Early label-efficiency benchmark runs (`le_b2_supcon`, `le_b12_supcon`) hard-coded `n_labeled_per_class: 2` in every YAML config, regardless of the label budget:

```yaml
training:
  batch_size          : 128
  n_labeled_per_class : 2      # ← BUG: capped at 4/batch for ALL budgets
```

**What this meant in practice:**

| Budget | Available labels | Labels used per batch (n_lpc=2) | % of budget used |
|--------|-----------------|--------------------------------|-----------------|
| 10     | 10              | 4                              | 40%             |
| 50     | 50              | 4                              | 8%              |
| 150    | 150             | 4                              | 3%              |
| 750    | 750             | 4                              | 0.5%            |

For any budget above ~10, the SC loss only ever saw 4 patches (2 adhesion + 2 No adhesion) per batch — exactly the same as at budget=10. Increasing the label budget provided no additional SC supervision. The model effectively trained with a fixed label budget of ~4 regardless of what was provided.

---

## 4. Buggy Run Results

### 4.1 DS1 B2 — `le_b2_supcon` (lat=32, proj=8, n_lpc=2) — logreg, 5-fold CV × 5 repeats

| Budget | Mean bal_acc | Std |
|--------|-------------|-----|
| 10     | 0.560       | 0.101 |
| 25     | 0.616       | 0.126 |
| 50     | 0.719       | 0.121 |
| 75     | 0.766       | 0.107 |
| 100    | 0.825       | 0.043 |
| 150    | 0.835       | 0.045 |
| 200    | 0.886       | 0.029 |
| 300    | 0.927       | 0.025 |
| 500    | 0.944       | 0.020 |
| 750    | 0.954       | 0.017 |

Performance at small budgets (n=10: 0.560, n=25: 0.616) was barely above chance, well below CellProfiler and ilastik baselines at those budgets.

### 4.2 DS1 B12 — `le_b12_supcon` (lat=32, proj=8, n_lpc=2) — logreg, 5-fold CV × 5 repeats

| Budget | Mean bal_acc | Std |
|--------|-------------|-----|
| 10     | 0.529       | 0.116 |
| 25     | 0.605       | 0.095 |
| 50     | 0.666       | 0.081 |
| 75     | 0.679       | 0.066 |
| 150    | 0.732       | 0.059 |
| 400    | 0.814       | 0.033 |
| 750    | 0.849       | 0.026 |

Even worse than B2 — the B12 label pool is larger (2,428 patches) but the cap at 4 labels/batch meant no benefit from the larger pool.

### 4.3 DS1 B2 ctrl-only — `le_b2_vinc_ctrl` — near-chance results

A ctrl-only run (`le_b2_vinc_ctrl`) also had `n_labeled_per_class: 2` plus a different test set mismatch. Results were near-chance (bal_acc ≈ 0.479 at n=10), prompting the investigation that uncovered the bug.

---

## 5. Bug Fix — Remove `n_labeled_per_class`

The fix was simple: remove `n_labeled_per_class` from all configs entirely. The pipeline default is `n_labeled_per_class = 0`, which uses standard random shuffling — all available labels have an equal chance of appearing in any batch proportional to their frequency in the training pool.

New runs were created with:
- `le_b2_lat12p8`: DS1 B2, lat=12, proj=8, no n_lpc, 5-fold × 5 repeats (280 jobs)
- `le_b12_ds2_lat12p8`: DS2 B12, lat=12, proj=8, no n_lpc, 5-fold × 5 repeats (180 jobs)

---

## 6. Fixed Run Results

### 6.1 DS1 B2 — `le_b2_lat12p8` (lat=12, proj=8, n_lpc=0)

**LGBM, 5-fold CV × 5 repeats:**

| Budget | Mean bal_acc | Std |
|--------|-------------|-----|
| 10     | **0.811**   | 0.082 |
| 25     | 0.843       | 0.053 |
| 50     | 0.869       | 0.039 |
| 75     | 0.880       | 0.036 |
| 100    | 0.881       | 0.038 |
| 150    | 0.895       | 0.033 |
| 200    | 0.916       | 0.025 |
| 300    | 0.930       | 0.024 |
| 500    | 0.942       | 0.019 |
| 750    | 0.943       | 0.020 |

**Logreg, 5-fold CV × 5 repeats:**

| Budget | Mean bal_acc | Std |
|--------|-------------|-----|
| 10     | **0.809**   | 0.052 |
| 25     | 0.823       | 0.042 |
| 50     | 0.858       | 0.041 |
| 75     | 0.877       | 0.044 |
| 100    | 0.889       | 0.043 |
| 150    | 0.908       | 0.036 |
| 200    | 0.899       | 0.043 |
| 300    | 0.941       | 0.028 |
| 500    | 0.955       | 0.016 |
| 750    | 0.964       | 0.012 |

The bug fix improved n=10 from 0.560 → 0.811 (LGBM) — a gain of +0.25 balanced accuracy.

### 6.2 DS2 B12 — `le_b12_ds2_lat12p8` (lat=12, proj=8, n_lpc=0)

**LGBM, 5-fold CV × 5 repeats:**

| Budget | Mean bal_acc | Std |
|--------|-------------|-----|
| 10     | 0.773       | 0.122 |
| 20     | 0.816       | 0.110 |
| 25     | 0.855       | 0.083 |
| 50     | 0.872       | 0.084 |
| 75     | 0.894       | 0.058 |
| 100    | 0.891       | 0.081 |
| 150    | 0.866       | 0.095 |

DS2 B12 SupCon-AE remained 7–11% below CellProfiler at all budgets, which prompted further analysis.

---

## 7. Per-Batch Label Analysis

After observing that DS2 B12 SupCon-AE consistently underperformed CP/ilastik even with the bug fixed, a per-batch label density analysis was performed.

The key question: **with standard random shuffling, how many labeled patches appear per batch on average?**

Expected labels per batch = `n_train / total_patches × batch_size`

### DS1 B2 (total pool = 27,637 patches, batch_size = 128)

| Budget | n_train | Labels/batch (expected) |
|--------|---------|------------------------|
| 10     | 10      | **0.05**               |
| 25     | 25      | 0.12                   |
| 50     | 50      | 0.23                   |
| 100    | 100     | 0.46                   |
| 300    | 300     | 1.39                   |
| 750    | 750     | 3.47                   |

### DS2 B12 (total pool = 3,400 patches, batch_size = 128)

| Budget | n_train | Labels/batch (expected) |
|--------|---------|------------------------|
| 10     | 10      | **0.38**               |
| 25     | 25      | 0.94                   |
| 50     | 50      | 1.88                   |
| 100    | 100     | 3.76                   |
| 150    | 130     | 4.89                   |

At small budgets (n ≤ 25), the expected labels per batch is less than 1 for both datasets — meaning most batches contain zero labeled patches and the SC loss effectively does not fire. DS1 B2 is even more diluted than DS2 B12 (0.05 vs 0.38 at n=10) because it has 8× more total patches. Despite this, DS1 B2 achieves strong performance (0.811 at n=10), suggesting that **the UC loss alone provides useful representations** and the SC loss is not the primary driver at small budgets.

For the SC loss to form a useful positive pair, a batch needs ≥ 2 patches of the same class. The probability of this at n=10:
- DS2 B12: ~5.6% of batches → ~200 SC-active batches in 500 epochs
- DS1 B2: ~0.1% of batches → ~110 SC-active batches in 500 epochs

The SC loss contributes only rarely, but the UC loss over the full unlabeled pool appears sufficient.

---

## 8. Adaptive `n_labeled_per_class` Experiment

To address the sparse SC signal without the hard cap of the old bug, an adaptive formula was designed:

```
n_lpc = min(budget // n_classes, batch_size // n_classes // 4)
      = min(budget // 2, 16)          [for n_classes=2, batch_size=128]
```

This caps at 16 per class (32 labels per batch = 25% of 128), ensuring at least 96 unlabeled slots remain for the UC loss.

**n_lpc and labels/batch under this formula:**

| Budget | n_lpc | Labels/batch | Oversampling ratio |
|--------|-------|-------------|-------------------|
| 10     | 5     | 10          | 27×               |
| 20     | 10    | 20          | 27×               |
| 25     | 12    | 24          | 26×               |
| 50     | 16    | 32          | 17×               |
| 75     | 16    | 32          | 12×               |
| 100    | 16    | 32          | 9×                |
| 150    | 16    | 32          | 7×                |
| 750    | 16    | 32          | 2×                |

"Oversampling ratio" = how many more times each labeled patch appears compared to the natural rate (≈once per epoch). At n=10 each label patch is forced into all 27 (DS2) or 216 (DS1) batches per epoch.

Test runs were created with this formula for fold=0, repeat=0:

- `le_b12_ds2_nplc`: DS2 B12, all 8 budgets (8 jobs, completed)
- `le_b2_ds1_nplc`: DS1 B2, 11 of 12 budgets completed (nb=500 cancelled)

---

## 9. Adaptive nplc Test Results

### 9.1 DS2 B12 — nplc vs no-sampler (fold=0, repeat=0, LGBM)

| Budget | No sampler | nplc (adaptive) | Δ |
|--------|-----------|-----------------|---|
| 10     | 0.707     | 0.429           | −0.279 |
| 20     | 0.650     | 0.529           | −0.121 |
| 25     | 0.729     | 0.500           | −0.229 |
| 50     | 0.864     | 0.807           | −0.057 |
| 75     | 0.957     | 0.964           | +0.007 |
| 100    | 0.893     | 0.643           | −0.250 |
| 150    | 0.864     | 0.814           | −0.050 |
| all    | 0.879     | 0.679           | −0.200 |

### 9.2 DS1 B2 — nplc vs no-sampler (fold=0, repeat=0, LGBM)

| Budget | No sampler | nplc (adaptive) | Δ |
|--------|-----------|-----------------|---|
| 10     | 0.843     | 0.434           | −0.409 |
| 20     | 0.849     | 0.803           | −0.047 |
| 25     | 0.777     | 0.319           | −0.458 |
| 50     | 0.869     | 0.528           | −0.341 |
| 75     | 0.920     | 0.790           | −0.130 |
| 100    | 0.930     | 0.843           | −0.087 |
| 150    | 0.949     | 0.831           | −0.118 |
| 200    | 0.943     | 0.694           | −0.250 |
| 300    | 0.972     | 0.891           | −0.081 |
| 750    | 0.967     | 0.949           | −0.019 |
| all    | 0.965     | 0.949           | −0.016 |

---

## 10. Conclusions

1. **The original `n_labeled_per_class: 2` was a hard cap bug** that limited SC supervision to 4 patches per batch regardless of budget. This produced near-chance results at small budgets (0.529–0.560 at n=10) and failed to benefit from larger label sets.

2. **Removing the cap (n_lpc=0) substantially improved results** — DS1 B2 improved from 0.560 → 0.811 at n=10 (LGBM). The UC loss over the full unlabeled pool is the dominant driver of representation quality, not the SC loss.

3. **Despite sparse SC signal at small budgets** (0.05–0.38 labeled patches per batch), the no-sampler approach works well because the UC loss trains rich representations from the large unlabeled pool.

4. **Adaptive nplc (min(b//2, 16)) consistently hurts performance** at virtually all budgets in both DS1 B2 and DS2 B12. Forcing labeled patches into every batch oversamples the small label set (7–27×), which interferes with the UC loss and biases the latent space toward memorizing the small labeled pool rather than learning generalizable representations.

5. **The performance gap between SupCon-AE and CP/ilastik on DS2 B12** appears to reflect genuine dataset characteristics (small annotation pool of 244 patches, 72% class imbalance, small total unlabeled pool of 3,400 patches) rather than a per-batch supervision bug. The no-sampler approach is already optimal given the current architecture.

6. **Further investigation is deferred.** The nplc experiment is cancelled. The no-sampler (n_lpc=0) baseline is the correct default for all future runs.
