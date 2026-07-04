# Session Log — 2026-05-21

## 1. Label CSV Standardization

### Problem
`labels_vinc_20260521.csv` (generated previous session) had 376/1340 rows with missing
`czi_filename`. Root cause: `LABEL_COLS` in `scripts/labels/helpers.py` deliberately
excluded `czi_filename` when combining batch1 labels into
`paxdata_paxpatch_batch1and2_combined_labels.csv`. Margaret CSVs only covered batch2
rows (964 rows); batch1 source CSVs (project-13, project-15) had the column all along
but it was dropped at combine time.

### Fix
Updated `scripts/labels/label_org.py` to include all 3 batch1 source CSVs in `czi_map`:
- `project-13-at-2025-12-18-15-32-df671bd2.csv` (53 ctrl rows)
- `project-13-at-2025-12-19-15-41-effff775.csv` (176 ctrl rows)
- `project-15-at-2025-12-22-18-44-b7e23381.csv` (194 ycomp rows)

### Output
Three canonical label CSVs produced (timestamp 20260521):
| File | Rows | czi_filename |
|------|------|-------------|
| `labels_vinc_20260521.csv` | 1340 | 1340/1340 ✓ |
| `labels_ppax_20260521.csv` | 60 (control only) | 60/60 ✓ |
| `labels_pfak_20260521.csv` | 54 | 0/54 (not available) |

All use consistent columns: `dataset`, `unique_ID` (hyphen format), `condition`,
`crop_img_filename`, `czi_filename`, `classification`, `Position`, `annotator`.

Updated 4 vinc contrastive cls configs to use `labels_vinc_20260521.csv`.

---

## 2. Contrastive AE Re-run (stages 3–4, job 869750)

Re-ran classification + visualization only (skipped AE retraining) using new label CSV
with 1340 rows vs 959 in the old `vinc_combined_labels.csv`.

Script: `scripts/sbatch_contrastive_cio_rb_vinc_lat12proj8_cls_vis.sh`
Results: `ae_results/contrastive_run/contrastive_cio_rb_vinc_lat12proj8/`

---

## 3. Contrastive AE with 0322 Training Strategy (job 869760)

### Motivation
Default contrastive AE (500 epochs) gave poorly separated FA-type clusters.
Tested whether the 0322 training strategy (which gives best FA classification in semisup)
would help the contrastive AE.

### 0322 settings applied to contrastive AE
- `epochs: 200` (from 500)
- `weight_decay: 0.0` (from default 1e-4)
- `min_epochs_for_best: 0` (use final model, not best checkpoint)
- `lr_scheduler: "none"` (already default)

New configs: `config/contrastive_config/*_0322*.yaml`
Script: `scripts/sbatch_contrastive_cio_rb_vinc_lat12proj8_0322.sh`
Results: `ae_results/contrastive_run/contrastive_cio_rb_vinc_lat12proj8_0322/`

### Observation
0322 contrastive AE: features more clustered than default, but FA classes still mixed.
Same qualitative result as default strategy.

---

## 4. Key Insight: Warmup Phase ≡ ConAE

The semisup warmup phase (recon-only, no classification loss) is structurally equivalent
to what the contrastive AE does — both produce a latent space shaped by visual similarity
(intensity, texture) with no FA-type discrimination. The conAE experiment confirmed what
"warmup at convergence" looks like.

**Why 0322 semisup outperforms strategies with warmup:**
- With warmup (mar30, final): encoder first learns a visually-structured space, then
  classification heads must fight that inertia to reshape z toward FA types.
- Without warmup (0322): classification loss active from epoch 1 → encoder never
  settles into a visual-only structure → FA-type discrimination baked in from the start.
- The apparent overfitting in 0322 is a consequence of strong cls supervision, not noise.

**Testable prediction:** Shorter warmup (50 or 100 epochs instead of 200) should
interpolate: shorter warmup → less inertia → better FA classification. This would
cleanly isolate warmup duration as the key variable.

---

## 5. Branch Merge & Cleanup

Merged both feature branches into main:

1. Committed pending work on `exp/contrastive_projector` (0322 configs, label_org.py,
   updated vinc label paths)
2. Merged `exp/training_strategy_sweep` → `exp/contrastive_projector` (zero conflicts;
   shared code changes were cherry-picked so diffs were identical)
3. Fast-forwarded `main` to `66c9813`
4. Deleted both feature branches
5. Removed git worktree for `SubCellAE_contrastive_projector`
   (directory still has NFS lock from closed jobs; safe to `rm -rf` once lock clears)

**Main repo going forward:** `/net/projects/CLS/lding/gitcode/SubCellAE` on `main`

---

## 6. Planned Next Steps

- **Strategy × conAE dim sweep**: sweep training strategies (0322/0324/mar30/final +
  shorter-warmup variants) combined with conAE latent/proj dim variants
- **Warmup duration experiment**: add intermediate strategies (warmup=0/50/100/200) to
  isolate warmup as the key variable driving 0322 advantage
- **Generalizability evaluation**: apply vinc-trained models to ppax/pfak/nih3t3 using
  `config/other_paxillin_cio_config/` pipeline (needs MODEL_RUN paths wired once
  strategy sweep models are trained)
