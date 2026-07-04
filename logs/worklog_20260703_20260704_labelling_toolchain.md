# Worklog — H5 Packing & Labelling Interface
**Dates:** 2026-07-03 – 2026-07-04  
**Branch:** `main`

---

## Summary

Built and iterated on a complete labelling toolchain for manual FA patch annotation:
- `pack_patches_label_h5.py` — lightweight cluster-side packer (patches + source frames → H5)
- `label_patches.py` — browser-based Panel/Bokeh labelling interface
- Supporting frameextract pipeline, patchprep configs, sbatch scripts, and user documentation

Work was split across two machines (Ubuntu workstation and DSI cluster) and merged iteratively.

---

## 2026-07-03

### 19:39 — `d522a7a` Initial labelling toolchain
Added the foundational pieces in a large commit:
- **`subcellae/pipeline/frameextract_pipeline.py`** — new pipeline that reads CZI files, applies rolling-ball background subtraction and CIO (cell-inside/outside) normalization, and saves full-frame per-channel TIFFs to `ae_results/source_frames/cio_rb/{ds}/{cond}/`. These serve as the labeller canvas images.
- **`config/frameextract_config/`** — 8 YAML configs (4 datasets × 2 conditions) specifying channel indices, names, scale factors, and segmentation parameters.
- **`config/patchprep_config/*_cio_rb_label.yaml`** — 8 patchprep configs for the label patch extraction run (broader cell mask: `seg_threshold=0.05`, `mask_ratio=0.1`; output to `tiff_patches32_label/`).
- **`scripts/pack_patches_label_h5.py`** — new packer that reads pre-extracted patches from `patches/cio_rb/{ds}/{cond}/tiff_patches32_label/` and source frames from `source_frames/cio_rb/{ds}/{cond}/`, then writes one H5 per (dataset, condition).
  - H5 layout: `patches/raw`, `images/raw` (paxillin main canvas), `images/{ch}` (marker, zyx, act), `meta/csv`, `images/meta`
  - Patch metadata columns: `filename`, `patch_group`, `condition_name`, `canvas_cx`, `canvas_cy`, `ps`
  - CSVs stored as `np.bytes_()` scalar for `.decode()` compatibility in the labeller
  - `PAD_SIZE=64` applied to convert padded patch coordinates back to unpadded canvas coordinates
- **`scripts/label_patches.py`** — initial browser-based labelling tool (Panel + Bokeh). Main canvas with patch rectangles, label buttons, click-to-label, finish & save.
  - Save filename includes the H5 stem for traceability (e.g. `vinc_control_label_lding_20260703.csv`)
  - Resume from CSV: load button lets annotators restore labels from a previous session
- **`scripts/sbatch_frameextract_all.sh`**, **`sbatch_frameextract_vinc.sh`** — cluster job scripts.
- **`.gitignore`** updated: added `slides/`, `logs/slurm/`, `*.pptx`.

### 22:24 — `04a63a1` Enhanced labeller: 4-channel canvas and patchprep packer
- Added 3 side canvases (other channels, linked pan/zoom with main canvas).
- Added 4 patch thumbnail figures (one per channel), initially driven by `patches/allch`.
- Hardcoded channel name fallback detection from H5 path (vinc/pfak/ppax keyword search).
- Wired `_MAIN_CH` detection from `channel_names` rather than assuming index 0.

### 22:39 — `ddad571` Support `pack_patches_label_h5` format in labeller
- `load_h5()` gained a second format branch (`elif img_meta is not None and images_raw is not None`) to handle the new packer's layout (separate `images/{ch}` datasets rather than `images/allch/{group}`).
- Channel order read from `attrs['channels']` (preserving insertion order) instead of alphabetical sort, preventing name↔image index desync.
- `images_allch` built on-the-fly from individual channel arrays for compatibility with the rest of the labeller logic.

### Cluster iteration (07-03, evening)
Initial packs had 3 channels (pax, act, marker); updated packer to include zyxin as the 4th channel with correct order `[pax, marker, zyx, act]`. Repacked all 8 H5 files (4 datasets × 2 conditions) after each fix. The channel order bug (labeller using `sorted()` on extra channel names, causing zyxin slot to show actin data) was identified and fixed in this iteration.

---

## 2026-07-04

### 08:39 — `1f1e975` Add zyxin channel to packer and labeller
- **Packer:** changed channel list from `["pax", "act", ds_ch]` to `["pax", ds_ch, "zyx", "act"]`, adding zyxin and fixing channel order so side canvases display **marker | zyxin | actin**.
- **Labeller:** added `'zyx': 'zyxin'` to the `_ABBR` abbreviation map so zyxin is correctly named in channel titles and thumbnails.
- All existing NAS H5 files identified as needing re-packing (missing zyxin, old channel order).

### 08:44 — `e0b4b61` On-the-fly patch thumbnail extraction
- `patches/allch` does not exist in the new packer format (only `patches/raw`, single-channel paxillin).
- Labeller tap handler updated: when `patches_allch is None`, thumbnails are extracted on-the-fly by cropping `images_allch_norm` at `(canvas_cx, canvas_cy)` with the patch's `ps` radius.
- Correctly handles image boundary clamping.

### 09:41 — `61ae77c` Fix channel order, side canvas overlays, resume CSV default
- **Channel order fix confirmed:** `load_h5()` now uses `attrs['channels']` order (not sorted). Verified correct name↔image mapping for both old and new H5 formats.
- **Side canvas overlays:** patch rectangles now drawn on side canvases (faint outlines) as well as main canvas, with selected patch highlighted.
- **Resume CSV default:** auto-detects the most recent `{h5_stem}_*.csv` in the H5 folder and pre-fills the resume path field. Annotators no longer need to locate the file manually.

### 10:21 — `678828d` Session context file
- Added `logs/context_labeling_toolchain_20260704.md` documenting current H5 file states, channel layout, known issues, pending tasks, and key script descriptions for session continuity.

### 12:31 — `9ab23ac` Frameextract scale updates and display normalization (cluster)

**Dim image root cause:** all channels appeared very dim in the labeller. `arr / arr.max()` was being crushed by outlier bright pixels — e.g. zyxin had `max=8.88` while `p99=1.21`, so typical cell pixels displayed at ~14% of full brightness. Secondary finding: the paxillin scale was `8.0` in the frameextract configs but `5.0` in patchprep — a 1.6× mismatch introduced when the frameextract configs were first written.

**Fixes:**
- **`label_patches.py`:** replaced all four `arr / arr.max()` normalization sites with `_display_norm()` — clips at the 99.9th percentile then scales to [0, 1]. Display-only; stored H5 values are untouched.
- **Frameextract configs:** adopted calibrated scale values across all 8 configs: pax `8.0→5.0`, zyx `5.0→4.0`, act `5.0→4.0`, marker unchanged at `5.0`.
- Re-ran frameextract for all 4 datasets in parallel (SLURM jobs 948774–948777) and repacked H5s (job 948778).

**Parallel sbatch infrastructure:**
- `sbatch_frameextract_all.sh` had run everything sequentially (~8 h wall time). Split into per-dataset jobs (`sbatch_frameextract_{vinc,nih3t3,ppax,pfak}.sh`) run in parallel, with `sbatch_pack_label_h5.sh` as an `afterok` dependency. Reduced effective wall time to ~4 h (bottleneck: vinc with 91 CZIs).
- Added `sbatch_pack_label_h5.sh` as a standalone packer job script.

**Python environment fix (cluster):**
- `subcellae-cuda` conda env was missing `yaml`, `czifile`, and `matplotlib` in some contexts.
- Solution: use `miniconda3/bin/python3` (has `czifile`, `tifffile`, `scipy`, `skimage`) with `core_env` site-packages on `PYTHONPATH` (contributes `yaml`, `matplotlib`). Applied to all frameextract and patchprep sbatch scripts.

### 12:37 — `54b72c6` Fix side canvas showing paxillin; add rsync script (Ubuntu)
- **Bug fix:** side canvases were iterating over all `_n_canvas_ch` channels (0–3), causing paxillin (channel 0) to appear as the first side canvas — duplicating the main canvas. Users were seeing the main stain instead of the marker channel.
  - Added `_side_ch_indices = [ci for ci in range(_n_canvas_ch) if ci != _MAIN_CH]`.
  - Side canvas creation loop and `_update_ch_canvas()` now iterate over `_side_ch_indices`, correctly showing **marker | zyxin | actin**.
  - Removed a stale post-hoc filter in the layout section that was attempting the same fix incorrectly (it dropped the marker figure instead of paxillin).
- **`scripts/packedh5_rsync.sh`:** new script to rsync packed H5 files from cluster to local mirror and NAS mount for all 4 datasets × 2 conditions.

### 12:46 — `5e53f40` Patchprep label config tuning (cluster)
- Raised `seg_threshold` in all 8 patchprep label configs: `0.05 → 0.08` (tighter cell mask, reduces spurious background patches at cell edges).
- Cleared old `tiff_patches32_label/` content (~44k files total) before re-running.
- Created parallel per-dataset patchprep sbatch scripts (`sbatch_patchprep_label_{vinc,nih3t3,ppax,pfak}.sh`).
- Re-ran patchprep (SLURM jobs 948801–948804) and repacked H5s (job 948805).
- Added 4 per-dataset patchprep sbatch scripts to repo.

### 14:57 — `62a313d` Docs folder
- Created `docs/` with four files:
  - `labeling_guide.md` — technical guide for running the server and labelling interface.
  - `labeling_guide_for_annotators.md` — simplified version for non-technical annotators; includes the full launch command for all 8 H5 files.
  - `FA_Prototype_Labeling_Protocol.md` — FA type labelling protocol.
  - `FA_Random_Sampling_Labeling_Protocol.md` — random sampling annotation protocol.

### 15:00–15:06 — `9723462`, `f840eb8` Annotator guide revisions
- Updated `labeling_guide_for_annotators.md` with formatting improvements (indentation, line breaks) and content refinements based on review.

---

## Current state

- All code changes pushed to `origin/main` (HEAD: `94efcfb`).
- **Cluster jobs completed:** frameextract (jobs 948774–948777), H5 repack (948778), patchprep (948801–948804), H5 repack (948805).
- **Pending:** rsync new H5 files to NAS (`scripts/packedh5_rsync.sh`), then restart labeller and verify side canvases show **marker | zyxin | actin**.
