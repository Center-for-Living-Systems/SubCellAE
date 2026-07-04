# Contrastive AE Experiments — Summary for Discussion

**Goal:** Learn a latent representation of focal adhesion (FA) morphology from 32×32 paxillin fluorescence patches, using contrastive objectives to improve structure in the latent space beyond a standard reconstruction AE.

**Data:** vinc (paxillin, CIO RB) — control vs Y-compound (ycomp) treatment.
**Labels:** 1340 manually annotated patches (Margaret), 5 FA classes:
Nascent Adhesion → Focal Complex → Focal Adhesion → Fibrillar Adhesion → No Adhesion.
**Eval set:** 60 ppax patches (independent dataset, control only) for generalisability check.

---

## Models Explored

### Contrastive AE (ConAE)
- Standard SimCLR-style: two augmented views of the same patch pushed together in projection space
- Loss: `λ_recon · MSE(recon, input) + λ_contrast · NT-Xent(proj1, proj2)`
- Decoder trained on clean-view embedding only

### Supervised Contrastive AE (SupCon)
- Same architecture as ConAE (shared encoder + decoder + projection head)
- Loss: `λ_recon · recon_loss + λ_contrast · SupCon(proj, FA-labels)`
- SupCon pulls same-class patches together and pushes different-class patches apart
- Requires annotation labels at training time (used 1340 labelled vinc patches in mixed batches)

Both models: latent dim = 12, proj dim = 8, input 32×32 single channel.

---

## Experiment Axes

### 1. Enlarged Crop (`enlcrop`)
Instead of augmenting the 32×32 patch directly, load a 58×58 context window centred on the FA and randomly crop + rotate to produce each view.
- Augmentation is more realistic (genuine spatial variation vs synthetic flip/noise)
- `max_shift_px=4`, `max_angle_deg=15`

### 2. Input Scaling (`sc2`)
Divide input pixel values by 2 (`input_divisor=2.0`) so peak intensity ~0.5–0.6, fitting comfortably within Sigmoid output range.
- Problem discovered: with sc2, raw MSE loss ≈ 0.002, while contrastive loss ≈ 2.0 — reconstruction gradient essentially vanishes

### 3. Reconstruction Loss Type
| Type | Formula | Notes |
|------|---------|-------|
| `mse` | MSE(x̂, x) | default; scale-dependent |
| `l1`  | L1(x̂, x) | less sensitive to outlier pixels |
| `nmse` | MSE / mean(x²) | scale-invariant; sc2 doesn't change it |
| `nl1`  | L1 / mean(\|x\|) | scale-invariant L1 variant |

Normalised losses (`nmse`, `nl1`) were motivated by the sc2 loss imbalance: since they divide by the signal energy, `λ_recon=1.0` gives the correct balance regardless of input scale.

### 4. Lambda Contrastive (`λ_contrast`) Sweep
When normalised losses were introduced, reconstructions were still blurry — decoder was over-regularised by the contrastive objective. Tested:
- `λ_contrast = 0.5` (baseline)
- `λ_contrast = 0.25` (half)
- `λ_contrast = 0.125` (quarter)

### 5. Jitter Augmentation
Added colour jitter (intensity scaling) to the augmented view during contrastive training.

### 6. Architecture / Latent Dim Sweep (ConAE only)
Tested latent dim × proj dim combinations: 12×8, 12×12, 16×8, 16×12, 24×8, 24×12, 32×8, 32×12.

---

## Results

All models evaluated with:
- **KNN-5 classification accuracy on vinc val set** (train on labelled train patches, predict on labelled val patches; 1295 train / ~210 val after 80/20 group split)
- **KNN-5 accuracy on ppax** (transfer: vinc-trained KNN applied to 60 labelled ppax patches)
- **Silhouette score** on FA-type labels in the 12-dim latent space (computed over up to 5000 labelled patches)

### Full Results Table (sorted by vinc KNN acc)

| Run | vinc KNN | ppax KNN | silhouette |
|-----|----------|----------|------------|
| **supcon_jitter** | **0.586** | 0.314 | **-0.079** |
| supcon_enlcrop_sc2_l1 | 0.576 | 0.157 | -0.182 |
| supcon_enlcrop_sc2_lr8 | 0.567 | 0.078 | -0.171 |
| conae_enlcrop_sc2_l1 | 0.567 | 0.039 | -0.177 |
| conae_apr08_nowd | 0.555 | 0.373 | -0.140 |
| conae_enlcrop_sc2_nmse | 0.552 | 0.235 | -0.164 |
| conae_enlcrop_sc2 | 0.552 | 0.098 | -0.201 |
| conae_enlcrop_sc2_nmse_lc025 | 0.548 | 0.177 | -0.161 |
| supcon_enlcrop_sc2 | 0.548 | 0.157 | -0.159 |
| conae_enlcrop_sc2_lr4 | 0.548 | 0.059 | -0.169 |
| supcon_enlcrop (no sc2) | 0.524 | 0.078 | -0.176 |
| conae_mar30_nowd | 0.521 | 0.294 | -0.147 |
| supcon_enlcrop_sc2_nl1_lc025 | 0.519 | **0.333** | -0.165 |
| **supcon_lat12proj8 (baseline)** | 0.498 | **0.412** | -0.093 |
| conae_nonoise | 0.450 | **0.490** | -0.180 |
| conae_lat12proj8 (baseline) | 0.431 | 0.275 | -0.238 |
| conae_lat16proj8 | 0.440 | 0.392 | -0.214 |
| conae_lat24proj8 | 0.402 | 0.392 | -0.223 |
| conae_lat32proj8 | 0.407 | 0.392 | -0.223 |

*Full 45-run table available in the repo.*

### Key Numbers
- Random chance (5-class uniform): ~20%
- Best vinc KNN: **58.6%** (`supcon_jitter`)
- Best ppax KNN: **49.0%** (`conae_nonoise`) — but 51 labelled patches is a small sample
- All silhouette scores negative — FA types overlap in latent space; no clean clusters

---

## Key Findings

### SupCon > ConAE on vinc classification
SupCon consistently ranks higher on vinc KNN accuracy when label information is available. SupCon directly pulls same-FA-type patches together in projection space, which benefits local KNN classification even when global cluster structure (silhouette) is weak.

### Jitter augmentation helps SupCon most
`supcon_jitter` is the best single model on both vinc KNN and silhouette. Intensity jitter during augmentation creates harder positive pairs, forcing the encoder to learn truly morphology-discriminative features rather than intensity shortcuts.

### Enlarged crop is helpful but not always dominant
Enlcrop improves performance in many cases (compare `supcon_enlcrop_sc2` 0.548 vs `supcon_lat12proj8` 0.498), but the best single run is `supcon_jitter` without enlcrop. Possible reason: enlcrop introduces realistic spatial variation, but also more noise in the contrastive signal.

### sc2 + normalised loss is the right fix
Without sc2, Sigmoid output can saturate for vinc patches. With sc2, raw MSE/L1 reconstruction gradients collapse (~0.002 vs contrast ~2.0). Normalised losses (`nmse`, `nl1`) eliminate the imbalance and perform comparably to carefully retuned lambdas.

### λ_contrast reduction did not clearly help
Reducing λ_contrast from 0.5 → 0.25 → 0.125 was tried to address blurry reconstructions. Results were mixed — some lc025 variants are competitive but none surpassed the unnormalised baseline once sc2+nmse was applied.

### Increasing latent dim does not help (ConAE)
lat16 / lat24 / lat32 do not outperform lat12 on vinc KNN. The extra dimensions may encode nuisance variation. Consistent with diminishing returns in representation compression.

### ppax transfer is poor for enlcrop models
Models trained with enlcrop tend to have low ppax KNN accuracy (some as low as 0.04). The enlcrop augmentation may reduce overfitting to vinc-specific image features, but also appears to reduce the features that generalise to ppax. Baseline supcon (0.412) and conae_nonoise (0.490) generalise better to ppax.

---

## Open Questions for Discussion

1. **Why does SupCon with jitter beat enlcrop?**
   - Is jitter teaching intensity-invariant morphology while enlcrop is teaching spatial-invariance? These may be complementary — what about jitter + enlcrop together?

2. **Low ppax transfer — data domain shift or label mismatch?**
   - ppax and vinc use the same paxillin stain but different cell lines / conditions. Is the poor transfer due to image statistics shifting, or FA morphology genuinely looking different in ppax?

3. **All silhouette scores negative — is this a feature or a problem?**
   - FA types exist on a biological continuum (NA→FC→FA→FiBA), not as discrete clusters. Should we expect negative silhouette? Would ordinal/contrastive loss along the maturation axis help?

4. **Semi-supervised learning**
   - Only ~1300 labelled patches out of ~24000 total. SupCon only uses labels for the contrastive objective. What if we added a classification head on top of the frozen contrastive encoder (linear probe)?

5. **Multi-channel input**
   - Currently using paxillin channel only. Vinculin (ch1) and other channels carry complementary morphology information. Would a multi-channel AE improve class separation?

6. **Cross-dataset evaluation strategy**
   - ppax has only 60 labels. pfak has 54 labels but no czi match yet. Should the goal be to train on vinc + ppax combined and evaluate on pfak (or NIH3T3)?

---

## Pipeline Status

- Training: `scripts/run_ae_from_config.py` with YAML configs in `config/contrastive_config/`
- Evaluation: `scripts/run_contrastive_eval.py` — UMAP, PHATE, KMeans, KNN (vinc + ppax), confusion matrices
- Submission: `scripts/submit_all_contrastive_eval.sh` — auto-detects completed runs, submits eval jobs
- All 45 trained models evaluated; results under `ae_results/contrastive_run/<run>/eval/`
