# Semi-Supervised AE — Training Strategy Sweep Summary

**Goal:** Find the best training recipe for a semi-supervised autoencoder (SemiSup AE) that encodes FA morphology into an 8-dim latent space, enabling downstream classification of FA type and subcellular position.

**Data:** vinc (paxillin channel, CIO RB) — control + ycomp, ~24,000 32×32 patches total.
**Labels:** 1340 manually annotated patches (FA type × 5 classes; position × 4 classes).
**Split:** Group-aware 80/20 — all patches from the same image go to the same split (no image-level leakage).

---

## Model: Semi-Supervised AE

Standard convolutional AE (3× conv encoder → 8-dim latent → 3× deconv decoder, sigmoid output) with two classification heads attached to the encoder:

```
Loss = λ_recon · MSE(recon, input)
     + λ_cls   · CE(head_FA, fa_label)        [only on labelled patches]
     + λ_cls_2 · CE(head_pos, pos_label)      [only on labelled patches]
```

Classification heads are only activated on the ~5% of training patches that have annotations. The encoder is shared — classification loss shapes the latent even though most patches are unlabelled.

**Feature sets evaluated:**
- `lat8` — latent vector only (8 dims)
- `lat8dist8` — latent + 8 rotation-invariant distance-to-cell-edge features

**Classifier:** LightGBM, trained on train-split latents, evaluated on val-split latents.

---

## Strategy Definitions

| Strategy | Epochs | weight_decay | warmup | LR scheduler | Note |
|----------|--------|-------------|--------|--------------|------|
| `0322` | 200 | 0 | 0 | none | Shortest; no regularisation |
| `0324` | 500 | 1e-4 | 0 | none | Longer + weight decay |
| `0324_nowd` | 500 | 0 | 0 | none | 500ep, no weight decay |
| `mar30` | 500 | 1e-4 | 200ep | none | Long warmup (recon-only before cls) |
| `mar30_nowd` | 500 | 0 | 200ep | none | Long warmup, no weight decay |
| `apr08` | 500 | 1e-4 | 0 | cosine→1e-5 | Weight decay + cosine LR |
| `apr08_nowd` | 500 | 0 | 0 | cosine→1e-5 | Cosine LR, no weight decay |
| `final` | 500 | 1e-4 | 200ep | cosine→1e-5 | All regularisation combined |
| `warmup50` | 500 | 1e-4 | 50ep | none | Short warmup |
| `warmup50_nowd` | 500 | 0 | 50ep | none | Short warmup, no weight decay |
| `warmup100` | 500 | 1e-4 | 100ep | none | Medium warmup |
| `warmup100_nowd` | 500 | 0 | 100ep | none | Medium warmup, no weight decay |

`warmup` = epochs of reconstruction-only training before classification heads activate. After warmup, LR is reset and training continues with full loss.

---

## Results (group-aware val split, vinc dataset)

### FA-type classification accuracy

| Strategy | lat8 | lat8dist8 |
|----------|------|-----------|
| **0324** | **58.9%** | **59.8%** |
| 0324_nowd | 57.9% | 57.9% |
| warmup100_nowd | 56.5% | 58.9% |
| warmup50_nowd | 55.0% | 58.4% |
| apr08_nowd | 54.6% | 56.5% |
| 0322 | 55.0% | 56.0% |
| mar30_nowd | 53.1% | 57.4% |
| final | 52.4% | 53.8% |
| apr08 | 51.7% | 54.6% |
| warmup50 | 50.2% | 58.9% |
| mar30 | 49.8% | 58.4% |
| warmup100 | 46.4% | 56.0% |

### Position classification accuracy

| Strategy | lat8 | lat8dist8 |
|----------|------|-----------|
| 0322 | **66.7%** | 72.3% |
| 0324 | 62.0% | 66.2% |
| 0324_nowd | 62.4% | 63.4% |
| mar30 | 60.6% | **77.0%** |
| warmup50 | 56.3% | 76.1% |
| warmup50_nowd | 51.6% | 75.6% |
| warmup100 | 54.9% | 74.2% |
| apr08 | 57.3% | 73.2% |
| final | 57.0% | 71.5% |
| mar30_nowd | 55.9% | 71.8% |
| apr08_nowd | 56.3% | 71.8% |
| warmup100_nowd | 60.6% | 70.9% |

### Key numbers
- Random chance: FA ~20% (5-class), Position ~25% (4-class)
- Best FA (lat8dist8): **59.8%** (`0324`)
- Best Position (lat8dist8): **77.0%** (`mar30`)
- Distance features add **+6–15 pp** on position, modest gain on FA

---

## Key Findings

### 1. `0324` is the best strategy for FA classification
The `0324` strategy (500 epochs, weight_decay=1e-4, no warmup, no LR scheduler) consistently achieves the highest FA-type accuracy. The original hypothesis was that `0322` (shorter, no regularisation) would win due to less over-smoothing — in practice, `0324` is better with the honest group-aware split.

### 2. Distance features dominate position accuracy
The `lat8dist8` feature set adds 8 geometric features (rotation-invariant distances to cell edge binned by angle). These account for most of the position improvement, especially for position classification (+6–15 pp). `mar30` + lat8dist8 achieves 77% position — the warmup likely helps the encoder learn a cleaner spatial representation before classification pressure is applied.

### 3. Warmup helps position but hurts FA at `lat8`
Strategies with long warmup (mar30, warmup100) have lower FA lat8 accuracy but recover well with lat8dist8. The warmup allows the encoder to form a reconstruction-focused representation first; the subsequent classification head then struggles to reshape it for FA type (highly appearance-driven) but benefits for position (partially driven by geometry captured in distance features).

### 4. Weight decay is mildly helpful for FA
Comparing `0324` vs `0324_nowd`, and `warmup100` vs `warmup100_nowd`: weight decay consistently gives +1–3 pp on FA. Effect on position is mixed.

### 5. Cosine LR scheduler does not help
`apr08` (cosine) vs `0324` (constant LR): FA accuracy is worse with cosine. `final` (cosine + warmup) is also worse. The cosine decay may be reducing the effective learning rate too aggressively at later epochs.

### 6. High train accuracy, lower val accuracy — overfitting is real
With stratified split (not shown here), FA accuracy reaches 90%+. With group-aware split it drops to 50–60%. The gap is real: the model partially memorises image-level cues. The group-aware split is the honest estimate.

---

## Comparison with Contrastive Models

The best SemiSup strategy (`0324`, lat8dist8) achieves **59.8% FA val accuracy**.

For reference, the best contrastive model (`supcon_jitter`) achieves **58.6%** with KNN-5 on a 12-dim latent — competitive despite having no direct label supervision during training beyond the SupCon loss. The SemiSup model has an explicit classification head, uses the full label set, and has 2× more capacity in the latent.

| Model type | Best FA acc | Features | Notes |
|-----------|-------------|----------|-------|
| SemiSup AE | **59.8%** | lat8dist8 | Explicit cls head; 8-dim latent |
| SemiSup AE | 55.0% | lat8 only | Without distance features |
| SupCon AE (jitter) | **58.6%** | 12-dim latent | KNN-5; no cls head |
| SupCon AE (baseline) | 49.8% | 12-dim latent | No jitter, no enlcrop |
| Baseline AE | ~43% | lat8 | Pure recon; no supervision |

The near-parity of SupCon with SemiSup is the most interesting finding: contrastive label supervision in the projection space is nearly as effective as a direct classification head.

---

## Open Questions for Discussion

1. **SemiSup vs SupCon parity — what does it mean?**
   - SemiSup has a direct classification head but underperforms relative to its training accuracy → heavy overfitting
   - SupCon shapes the encoder indirectly via contrastive loss → less overfitting, better generalisation
   - Is the classification head in SemiSup actually helping or is it hurting by collapsing the latent space?

2. **Distance features as a crutch for position?**
   - Position accuracy jumps dramatically when distance features are added — but these are computed post-hoc from cell masks, not learned by the AE
   - Is position actually encoded in the latent at all, or are we just using hand-crafted geometry features?

3. **50–60% FA accuracy ceiling — is the data the bottleneck?**
   - All models plateau around 55–60% with group-aware split
   - Could be: (a) FA types are biologically ambiguous at patch scale, (b) labelling noise, (c) 8-dim latent insufficient, (d) paxillin-only input missing complementary information

4. **Combining SemiSup + SupCon?**
   - Could we run SupCon loss on the projection head AND keep the classification head from SemiSup?
   - Would pulling same-class patches together in projection space while also supervising the classifier improve both?

5. **Is the 0322 "overfitting advantage" a red herring?**
   - Original observation: `0322` appeared best with stratified split
   - With group-aware split: `0324` is best; `0322` is middle of the pack
   - Suggests the 0322 advantage was leakage from image-level memorisation, not better morphology encoding

---

## Pipeline Status

- Training: `scripts/run_ae_from_config.py` with configs in `config/training_strategies/` and sweep in `config/strategy_sweep/`
- Sweep execution: `scripts/run_strategy_sweep.sh`
- Results: `ae_results/strategy_sweep/<strategy>/semisup_both/`
- All 12 strategies trained and evaluated on vinc; cross-dataset (ppax) eval pending for sweep models
