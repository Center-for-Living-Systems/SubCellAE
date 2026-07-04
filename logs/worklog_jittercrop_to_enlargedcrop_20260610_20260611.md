# Work Log — JitterCropDataset → EnlargedCropDataset
# Jun 10–11, 2026

## Overview

End-to-end development of on-the-fly GPU-native augmentation for contrastive AE training.
Starting point: raw source frames and 32×32 TIFF patches.
End point: `EnlargedCropDataset` (58px context) + `_jitter_rot_crop()` (GPU batched affine)
delivering independent random crops for view1 and view2, with a single bilinear interpolation pass.

---

## 1. Background and Motivation (Jun 10)

Standard contrastive learning requires two independently augmented views of the same patch.
Our earlier pipeline re-used the same (rotated, jittered) patch for both views — losing the
view-independence that makes NT-Xent / SupCon effective.

The desired augmentation:
- Translation ±4 px (to add position invariance across jitter in the original FA detection)
- Rotation ±15° (to add orientation invariance)
- Each view gets its own independent random draw

The challenge: `scipy.ndimage.rotate` is a CPU call, single-threaded, called inside `__getitem__`.
At batch size 128 with 6 workers, augmentation overhead dominates DataLoader throughput.

On **Jun 10 22:31**, frame extraction (SLURM job 933033) completed — source frames for all
vinc/control and vinc/ycomp images were written to
`ae_results/source_frames/cio_rb/vinc/{control,ycomp}/`.
This unlocked context-aware loading, which is necessary for both JitterCrop and EnlargedCrop.

---

## 2. Patch Preparation (Jun 11 morning)

**Job 934012** — `patchprep_mr10`: extracted 32×32 patches with min-radius 10 filter
(removes patches too close to image borders).

| Condition | Patches   |
|-----------|-----------|
| control   | 14 879    |
| ycomp     | 12 758    |
| **Total** | **27 637** |

Group-aware 80/20 split: 21 784 train (73 images) / 5 853 val (18 images).

---

## 3. JitterCropDataset — Design

`JitterCropDataset` in `subcellae/modelling/dataset.py` loads a context patch larger than
32×32 at __getitem__ time, rotates it via scipy, and returns a center-cropped 32×32 view.

### Context size formula

The context patch must be large enough that after an arbitrary rotation θ the inscribed
square of size `patch_size` remains fully inside the padded frame.

```
θ = math.radians(max_angle_deg)          # 15°
ctx = math.ceil(patch_size * (math.cos(θ) + math.sin(θ))) + 4
self._ctx = ctx + (ctx % 2)              # round to even → 44 for ps=32, θ=15°
```

For ps=32, θ=15°: cos(15°)+sin(15°) = 0.9659+0.2588 = 1.225 → ceil(32·1.225)+4 = 44.

### Pre-padding at init

Every source frame is padded once at `__init__` (reflect mode) so `__getitem__` is a
zero-allocation numpy slice:

```python
self._pad_px = self._ctx // 2 + max_shift_px + 2   # = 22 + 4 + 2 = 28
self._frames_padded = {
    fkey: np.pad(arr, self._pad_px, mode='reflect')
    for fkey, arr in self._frames.items()
}
```

### __getitem__ pipeline

1. Slice `(ctx × ctx)` region from padded frame centered on patch centroid
2. `scipy.ndimage.rotate(region, angle_deg, reshape=False, order=1)` — single image
3. Center-crop to `(patch_size × patch_size)`
4. Return `(1, 32, 32)` float32 tensor

---

## 4. First Jitter Submissions — Failures

### 4a. Jobs 934119 / 934126 — matplotlib error

**Time**: ~13:44

```
ModuleNotFoundError: No module named 'matplotlib'
```

**Root cause**: sbatch scripts did not specify the Python path; SLURM used the system Python
instead of the conda env at `/net/projects/CLS/lding/conda_env/core_env/bin/python3`.

**Fix**: Added explicit `PYTHON=` variable to all sbatch scripts:
```bash
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
$PYTHON scripts/run_ae_from_config.py ...
```

### 4b. Jobs 934139 / 934150 — DataLoader hung for 44+ minutes

**Time**: resubmitted after matplotlib fix

**Symptom**: Both jobs started, printed the dataset loading INFO lines, then went silent.
No epoch output appeared after 44 minutes. Jobs were cancelled manually.

**Root cause**: `np.pad(full_frame, pad_px, mode='reflect')` was called inside `__getitem__`
— i.e., per patch, per epoch, across all 6 DataLoader worker processes simultaneously.
For 27 637 patches × 500 epochs, this is millions of full-frame padding calls. The workers
were each spending virtually all their time padding frames rather than returning batches.

**Fix**: Moved pre-padding to `__init__`:
```python
self._frames_padded = {
    fkey: np.pad(arr, self._pad_px, mode='reflect')
    for fkey, arr in self._frames.items()
}
```
`__getitem__` became a pure numpy slice — zero allocation, sub-microsecond per call.

Same fix was later applied to `EnlargedCropDataset` (jobs 934185/934186 had the same issue).

---

## 5. Successful Jitter Jobs (Jun 11 15:02)

**Jobs 934152 (ConAE) and 934153 (SupCon)** — both on node g003 (NVIDIA A40), started 15:02.

### ConAE jitter — job 934152

Config: `ae_contrastive_cio_rb_vinc_lat12proj8_jitter.yaml`
- `ctx=44px`, `max_shift=4px`, `max_angle=15°`, `warmup=0`, `lr_scheduler=cosine`, 500 epochs

| Metric | Value |
|--------|-------|
| Final recon (train/val) | 0.0070 / 0.0075 |
| Final contrast (train/val) | 3.8059 / 3.8597 |
| ep500 recon max | 0.354 (input max 1.190) |
| ep500 recon mean | 0.142 (input mean 0.143) |
| Best checkpoint | ep200, val=0.0078 |
| End time | 17:33:36 |

NT-Xent contrastive overfitting: train contrast improves slowly, val contrast worsens.
Reconstruction output mean equals input mean — the model learned to output a
spatially-smooth near-constant. Max output is only ~30% of input maximum.

### SupCon jitter — job 934153

Config: `ae_supcon_cio_rb_vinc_lat12proj8_jitter.yaml`
- Same as ConAE except `model_type=supcon`, `noise_prob=0.0`, `lr_scheduler=none`
- No warmup support yet at this point (warmup implementation came later)

| Metric | Value |
|--------|-------|
| Final recon (train/val) | 0.0041 / 0.0045 |
| Final contrast (train/val) | 3.8508 / 3.9097 |
| ep500 recon max | 0.731 (input max 1.155) |
| ep500 recon mean | 0.137 (input mean 0.145) |
| End time | 17:33:40 |

**SupCon reconstructs significantly better than ConAE** (0.73 vs 0.35 max).
Reason: supervised class labels (5 FA morphology categories) force the encoder to learn
morphologically-discriminative features, which the decoder can exploit for reconstruction.
NT-Xent only groups augmented views of the same patch — coarser representation.

Both jobs completed at 17:33, a 2h31m wall time for 500 epochs on 27 637 patches.

---

## 6. Design Problem: Double Interpolation in JitterCropDataset

Inspecting the training loop revealed a problem with JitterCropDataset:

1. `__getitem__` rotates the context patch via `scipy.ndimage.rotate` — **interpolation #1**
2. `augment_contrastive_view()` in the training loop applies a second random rotation — **interpolation #2**

Two sequential bilinear interpolations introduce compounding blur. Also, both views were drawn
from the same scipy-rotated patch (same random angle), partially defeating view independence.

The correct design:
- Load a larger context patch (no rotation at load time)
- Apply **independent** random rotation + jitter to each view inside the training loop
- On GPU using `F.affine_grid` + `F.grid_sample` — a single bilinear pass

---

## 7. EnlargedCropDataset — Design

### Branch decision

`JitterCropDataset` was kept for the **nonad-vs-ad** binary classification branch (which uses
ordered DataLoaders and does not have the view-independence issue). A new class
`EnlargedCropDataset` was created for the contrastive training branch.

### Context size formula

The context must fully enclose a 32×32 patch after rotation ±15° AND translation ±4px.

Worst-case sampling radius after rotation and shift:
```
r = √2 · (ps/2 + max_shift) = √2 · (16 + 4) = √2 · 20 ≈ 28.28 px
context_size = 2 · ceil(r) = 2 · 29 = 58 px
```

For ps=32, shift=4: 58px context, worst-case sample at radius 28.28 < 29 = 58/2. ✓

### Pre-padding at init

```python
self._pad_px = self.context_size // 2 + 4   # = 29 + 4 = 33
self._frames_padded = {
    fkey: np.pad(arr, self._pad_px, mode='reflect')
    ...
}
```

### __getitem__

Returns a `(1, 58, 58)` float32 tensor — just a raw context slice, no rotation or jitter.
Zero allocation, sub-microsecond.

Optional: `input_divisor` scales the patch before returning (e.g., ÷2 for sc2 experiment).

---

## 8. _jitter_rot_crop — GPU Batched Affine

`_jitter_rot_crop(x, max_shift_px, max_angle_deg, out_size)` in `autoencoders.py`:

Takes a `(B, 1, H, W)` batch of 58×58 context patches and returns `(B, 1, out_size, out_size)`
cropped patches with independent random affine transforms for each image in the batch.

**Theta matrix** (2×3 affine, normalized coordinates):

```
cos·s    sin·s    tx
-sin·s   cos·s    ty
```

where `s = out_size / H` (scale factor to crop from 58px to 32px), and `tx`, `ty` are uniform
random shifts in normalized-coordinate units. All B images get independent draws.

**Implementation**:
```python
theta = torch.zeros(B, 2, 3, device=x.device)
theta[:, 0, 0] =  cos_a * s
theta[:, 0, 1] =  sin_a * s
theta[:, 0, 2] =  tx
theta[:, 1, 0] = -sin_a * s
theta[:, 1, 1] =  cos_a * s
theta[:, 1, 2] =  ty
grid = F.affine_grid(theta, (B, 1, out_size, out_size), align_corners=False)
return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
```

Single GPU call — one bilinear interpolation, no double-degradation.

### View independence in training loop

For each batch:
```python
view1 = _jitter_rot_crop(ctx_batch, max_shift_px=4, max_angle_deg=15.0, out_size=32)
view2 = _jitter_rot_crop(ctx_batch, max_shift_px=4, max_angle_deg=15.0, out_size=32)
```
Each call draws fresh random angles and shifts → view1 and view2 are independently augmented
versions of the same patch.

---

## 9. First EnlargedCrop Submissions — np.pad Bug Again (Jun 11 16:04)

**Jobs 934185 (ConAE) and 934186 (SupCon)** — both on node g004 (NVIDIA A40), started 16:04.

Both printed dataset loading lines (27 637 patches, group-aware split) and then went silent.
**Cancelled at 17:49** — 105 minutes with no epoch output.

**Root cause**: The pre-padding fix applied to JitterCropDataset was not yet ported to
`EnlargedCropDataset`. The `__getitem__` in the new class was still calling
`np.pad(full_frame, ...)` per-patch.

**Fix**: Applied the same init-time padding to `EnlargedCropDataset`:
```python
self._frames_padded = {
    fkey: np.pad(arr, self._pad_px, mode='reflect')
    for fkey, arr in self._frames.items()
}
```

---

## 10. EnlargedCrop Results (Jun 11 17:49)

### 10a. stdout buffering fix

During the enlcrop debugging, epoch loss lines were also not appearing in SLURM log files.

**Root cause**: Python `print()` is fully buffered (~8 KB buffer) in non-TTY mode. The buffer
never flushed because each epoch print was well under 8 KB and no newline-flush trigger existed.

**Fix**: Added `flush=True` to all epoch `print()` calls in `train_contrastive_ae` and
`train_supervised_contrastive_ae`.

### 10b. Job 934303 — ConAE enlcrop (warmup=200)

Start: 17:49. End: 18:15:37. Duration: ~26 min.

| Epoch | recon (train) | recon (val) | contrast (train) | contrast (val) |
|-------|---------------|-------------|------------------|----------------|
| 10 (warmup) | 0.0074 | 0.0079 | — | — |
| 100 (warmup) | 0.0073 | 0.0078 | — | — |
| 200 (warmup) | 0.0073 | 0.0079 | — | — |
| 500 (final) | 0.0073 | 0.0078 | 3.9065 | 4.2474 |

ep500: input max 1.212, recon max 0.333, recon mean 0.145 ≈ input mean 0.145

**Finding**: ConAE warmup provided zero benefit. With sparse FA patches, MSE converges to a
near-constant "mean output" within the first few epochs (recon barely moved from 0.0074 to 0.0073
over 200 warmup epochs). The contrastive objective post-warmup could not rescue the collapsed encoder.

**Best-checkpoint bug found here**: `model_best.pt` was saved at ep200 with val_loss=0.0079.
This is the last warmup epoch, whose val loss (pure recon) permanently beats post-warmup val loss
(recon + contrast combined ≈ 2.1). See section 11 for the fix.

### 10c. Job 934304 — SupCon enlcrop (no warmup; warmup not yet implemented)

Start: 17:49. End: 18:15:40. Duration: ~26 min.

| Epoch | recon (train) | recon (val) | contrast (train) | contrast (val) |
|-------|---------------|-------------|------------------|----------------|
| 50 | 0.0060 | 0.0068 | — | — |
| 200 | 0.0048 | 0.0055 | — | — |
| 450 | 0.0046 | 0.0053 | 3.8948 | 4.3471 |
| 500 | 0.0046 | 0.0053 | 3.8943 | 4.3475 |

ep500: input max 1.212, recon max 0.749, recon mean 0.147 ≈ input mean 0.145

**Best visual quality** of all enlcrop runs. SupCon class labels (5 FA morphology categories)
force the encoder to learn FA-discriminative features → decoder can reconstruct FA structure.
Best checkpoint not tracked (warmup support not yet added to `train_supervised_contrastive_ae`).

---

## 11. Best-Checkpoint Bug and SupCon Warmup Implementation

### Bug

In `train_contrastive_ae`, the condition:
```python
if (epoch + 1) >= max(min_epochs_for_best, 1) and vl < best_val_loss:
```
With `min_epochs_for_best=200`, tracking starts at epoch 200 — the last warmup epoch.
Warmup val_loss ≈ 0.008 (recon only). Post-warmup val_loss ≈ 2.1 (recon + contrast).
The warmup epoch permanently wins; `model_best.pt` is always the pre-contrastive model.

### Fix (both training functions)

```python
past_warmup = (epoch + 1) > warmup_epochs
if past_warmup and (epoch + 1) >= max(min_epochs_for_best, 1) and vl < best_val_loss:
    best_val_loss = vl
    best_state    = copy.deepcopy(model.state_dict())
    best_epoch    = epoch + 1
```

### SupCon warmup implementation

`train_supervised_contrastive_ae` previously had no warmup support. Added:
- `warmup_epochs`, `weight_decay`, `min_epochs_for_best`, `lr_scheduler`, `lr_min` parameters
- `in_warmup = warmup_epochs > 0 and epoch < warmup_epochs`; `eff_lambda_contrast = 0 if in_warmup`
- LR reset to initial value and scheduler restart at warmup→contrast transition
- `best_val_loss` / `best_state` / `best_epoch` tracking with `model_best.pt` save
- `[warmup]` tag in epoch prints

---

## 12. Job 934341 — SupCon enlcrop with Warmup (Jun 11 ~18:20)

Config: warmup=100. Started immediately after 934304 completed.

**Warmup phase (ep1–100)**:
```
ep10:  train recon=0.0074  [warmup]
ep100: train recon=0.0073  [warmup]  val recon=0.0078
[epoch 100] recon  min=0.028  max=0.302  mean=0.132
SupCon AE  warmup complete — LR reset to 1.00e-03, no scheduler
```

The warmup stalled at recon ≈ 0.0073-0.0074 throughout — same mean-collapse behavior as ConAE.
At ep100, recon max was only 0.302 — barely better than constant output (input max 1.212).

**Post-warmup (ep101–500)**:
- SupCon introduced at ep101; contrast loss drives encoder to learn FA morphology
- Best checkpoint: **ep118, val_loss=2.0958** (correctly tracked post-warmup with the fix)
- Recon quality similar to 934304 (no-warmup) — warmup did not help for this sparse FA data

End: 19:29.

---

## 13. sc2 Experiment — Input ÷2 + No Sigmoid

### Motivation

FA pixel values reach ~1.2. The `nn.Sigmoid()` output activation clips at 1.0.
Hypothesis: dividing input by 2 (→ [0, 0.6]) and removing Sigmoid would let the decoder
match input amplitude without clipping.

### Code changes

- `EnlargedCropDataset`: `input_divisor: float = 1.0`; applied as `region / input_divisor` in `__getitem__`
- `ContrastiveAE`: `output_sigmoid: bool = True`; decoder conditionally appends `nn.Sigmoid()`
- `AEConfig`: added `enlarged_crop_input_divisor` and `output_sigmoid` fields
- `run_ae_from_config.py`: parses both from YAML

### Jobs 934342 (ConAE sc2) and 934343 (SupCon sc2) — original configs, NO Sigmoid

**ConAE sc2 (934342)**:
- ep500: input max 0.606, recon max 0.158, recon mean 0.072 ≈ input mean 0.072
- Recon loss 0.0018 = 0.0073/4 as expected from ÷2 scaling (MSE scales as 1/divisor²)
- Best checkpoint: ep11, val=2.0787 — model quickly found the mean-output minimum
- Same 26% ratio (0.158/0.606) as before; dividing input did not improve relative amplitude

**SupCon sc2 (934343)**:
- **Complete collapse**: recon constant (min=max=mean ≈ 0.051), contrast stuck at 5.53
- ln(256) ≈ 5.55 = random ceiling for batch size 128 in NT-Xent/SupCon → contrast learned nothing

### Root cause — mean collapse without Sigmoid

Without `nn.Sigmoid()`, the linear decoder converges to outputting `dataset_mean` as a constant
within the first few epochs. For sparse FA patches (mean ≈ 0.07), MSE is minimized at exactly
one constant value with zero spatial gradients. The encoder receives no useful gradient.

**SupCon sc2 was worst**: 100-epoch warmup (pure recon, no contrastive) completely collapsed the
encoder before SupCon was introduced. By ep101, the encoder was a near-identity mapping to the
mean, and SupCon could not recover it.

**Sigmoid's beneficial role**: The logistic nonlinearity has nonzero slope near the
constant-output equilibrium, creating gradient friction that slows collapse and gives the
contrastive objective time to drive representation learning.

### Fix

- Restored `output_sigmoid: true` in both sc2 configs
- Set SupCon sc2 `warmup_epochs: 0` (removing 100-epoch pre-collapse warmup)
- sc2 (÷2) design remains valid: input [0, 0.6] fits inside Sigmoid (0,1) with no clipping
  (logit(0.6) ≈ 0.43 — easily achievable pre-sigmoid activation)

---

## 14. Jobs Running at End of Day

| Job | Model | Config | Status |
|-----|-------|--------|--------|
| 934376 | ConAE | enlcrop_sc2 (input÷2, Sigmoid, no warmup) | running |
| 934377 | SupCon | enlcrop_sc2 (input÷2, Sigmoid, no warmup) | running |

---

## 15. Summary of All Jobs (Chronological)

| Job | Model | Key detail | Outcome |
|-----|-------|-----------|---------|
| 933033 | frame extraction | Jun 10 22:31 | completed |
| 934012 | patch prep mr10 | 27637 patches | completed |
| 934119 | ConAE jitter | matplotlib error | failed |
| 934126 | SupCon jitter | matplotlib error | failed |
| 934139 | ConAE jitter | np.pad bottleneck | cancelled after 44+ min |
| 934150 | SupCon jitter | np.pad bottleneck | cancelled after 44+ min |
| 934152 | ConAE jitter | np.pad fix, ctx=44 | completed; max recon 0.35 |
| 934153 | SupCon jitter | np.pad fix, ctx=44 | completed; max recon 0.73 |
| 934185 | ConAE enlcrop | np.pad not yet fixed | cancelled after 105 min |
| 934186 | SupCon enlcrop | np.pad not yet fixed | cancelled after 105 min |
| 934303 | ConAE enlcrop | np.pad fix + flush | completed; mean collapse, best=ep200 (bug) |
| 934304 | SupCon enlcrop | no warmup (not impl.) | completed; max recon 0.75 (best) |
| 934341 | SupCon enlcrop | warmup=100 + fix | completed; best ep118 val=2.0958 |
| 934342 | ConAE sc2 | no Sigmoid | completed; same 26% ratio, best=ep11 |
| 934343 | SupCon sc2 | no Sigmoid + warmup | collapsed; contrast stuck at random ceiling |
| 934376 | ConAE sc2 | Sigmoid restored | running |
| 934377 | SupCon sc2 | Sigmoid + no warmup | running |

---

## 16. Key Findings

1. **np.pad must be at init, not __getitem__**: Any full-frame padding inside the DataLoader
   worker loop scales as O(N_patches × N_epochs) and causes complete DataLoader stall.

2. **SupCon consistently beats NT-Xent on reconstruction quality**: FA class labels force
   morphology-discriminative latents; the decoder exploits these for better reconstruction.
   ConAE (NT-Xent) recon max ≈ 0.35; SupCon recon max ≈ 0.73-0.75 (input max ≈ 1.2).

3. **Warmup is ineffective for sparse FA data**: MSE on FA patches converges to a mean-output
   minimum within the first few epochs (recon barely changes: 0.0074 → 0.0073 over 200 epochs).
   The contrastive objective is what drives meaningful representation learning.

4. **Best-checkpoint tracking must gate on past_warmup**: Warmup val_loss (recon-only, ~0.008)
   will always beat post-warmup val_loss (recon+contrast, ~2.1). Without the gate, `model_best.pt`
   is always the pre-contrastive model.

5. **Sigmoid is a necessary inductive bias for sparse FA reconstruction**: Without it, the linear
   decoder collapses to dataset mean in <10 epochs. Sigmoid gradient friction prevents this.

6. **EnlargedCrop context formula**: `2·ceil(√2·(ps/2 + max_shift))` = 58px for ps=32, shift=4.
   Worst-case sample at 28.28px < 29px = context_radius. No out-of-bounds samples.

7. **GPU affine via F.affine_grid + F.grid_sample**: One bilinear pass, B images in parallel,
   independent per-image random angles and shifts. Eliminates double-interpolation from
   JitterCropDataset's scipy-at-load → augment_contrastive_view pipeline.

---

## 17. Files Changed

| File | Change |
|------|--------|
| `subcellae/modelling/dataset.py` | `EnlargedCropDataset` (new class); pre-padding fix for `JitterCropDataset`; `input_divisor` param |
| `subcellae/modelling/autoencoders.py` | `_jitter_rot_crop()` (new); enlcrop path in `train_contrastive_ae`; warmup + scheduler + best-checkpoint in `train_supervised_contrastive_ae`; `output_sigmoid` in `ContrastiveAE`; post-warmup best-checkpoint fix in both functions |
| `subcellae/pipeline/ae_pipeline.py` | `AEConfig` new fields; dataset/model construction wired |
| `scripts/run_ae_from_config.py` | Parse `enlarged_crop.input_divisor`, `model.output_sigmoid` |
| `config/contrastive_config/ae_contrastive_..._jitter.yaml` | New (JitterCropDataset baseline) |
| `config/contrastive_config/ae_supcon_..._jitter.yaml` | New (JitterCropDataset baseline) |
| `config/contrastive_config/ae_contrastive_..._enlcrop.yaml` | New (EnlargedCrop baseline) |
| `config/contrastive_config/ae_supcon_..._enlcrop.yaml` | New (EnlargedCrop baseline) |
| `config/contrastive_config/ae_contrastive_..._enlcrop_sc2.yaml` | New (sc2, Sigmoid restored) |
| `config/contrastive_config/ae_supcon_..._enlcrop_sc2.yaml` | New (sc2, Sigmoid, warmup=0) |
| `scripts/sbatch_conae_vinc_jitter.sh` | New |
| `scripts/sbatch_supcon_vinc_jitter.sh` | New |
| `scripts/sbatch_conae_vinc_enlcrop.sh` | New |
| `scripts/sbatch_supcon_vinc_enlcrop.sh` | New |
| `scripts/sbatch_conae_vinc_enlcrop_sc2.sh` | New |
| `scripts/sbatch_supcon_vinc_enlcrop_sc2.sh` | New |

---

## 18. Open Items

- **Check 934376/934377 tomorrow**: sc2 + Sigmoid should keep reconstruction intact while
  fitting input [0, 0.6] fully within Sigmoid range. SupCon no-warmup should reproduce 934304.
- **Fundamental amplitude gap**: Even best run (934304) max recon 0.749 vs input max 1.212.
  Sigmoid clips any output above 1.0; input values 0.7–1.2 are unreachable. Consider removing
  Sigmoid only after verifying sc2 avoids collapse (pending 934376/934377).
- **Contrastive overfitting**: All runs show train contrast improving while val contrast worsens.
  Inherent to small-dataset contrastive learning. No fix planned.
- **nonad-vs-ad branch**: JitterCropDataset pre-padding fix not yet ported;
  `num_workers=0` in ordered DataLoaders still hardcoded.
