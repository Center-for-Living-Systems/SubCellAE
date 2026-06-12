# CLAUDE.md — SubCellAE nonad-vs-ad branch

## Branch purpose
`exp/nonad-vs-ad-cls`: classification of non-adhesion vs adhesion FA patches using
the contrastive/supcon AE framework. Adds `pair_weights` / `pair_weights_classes`
to `AEConfig` for a weighted SupCon loss with a custom K×K class-pair weight matrix.

---

## Recent changes ported from main branch (not in git history here)

### 1. On-the-fly jitter crop dataset (`JitterCropDataset`)

**Motivation:** Pre-cropped 32×32 patches cannot provide translation or small
free-angle rotation diversity. Instead of augmenting the static patches, we now
load full-frame source TIFFs and crop on-the-fly every epoch.

**Files changed:**

#### `subcellae/modelling/dataset.py`
- Added imports: `math`, `scipy.ndimage.rotate as _ndrotate`
- Added regex `_COORD_RE` — parses full coordinate block from patch filenames:
  `control_f0002x0784y0624ps32.tif` → condition_str, frame_idx, cx_pad, cy_pad, ps
- Added class `JitterCropDataset(Dataset)`:
  - Constructor args: `patch_dir`, `frame_dir`, `channel`, `condition`,
    `max_shift_px=4`, `max_angle_deg=15.0`, `patch_size=32`, `pad_size=64`,
    plus same annotation kwargs as `PatchDataset`
  - At `__init__`: scans `patch_dir` filenames for FA center coordinates
    (subtracts `pad_size=64` to recover unpadded frame coords), loads source
    frame TIFFs from `{frame_dir}/{condition_str}_f{fidx:04d}_{channel}.tif`
    into a per-frame cache (`self._frames` dict — one array per unique frame)
  - At `__getitem__`: samples random `dx,dy ∈ [-max_shift_px, +max_shift_px]`
    and `angle ∈ [-max_angle_deg, +max_angle_deg]`, pads the frame with
    `mode='reflect'`, extracts a context region of size
    `ceil(ps*(cos θ + sin θ)) + 4` (rounded to even; 44px for ps=32, θ=15°),
    rotates with `scipy.ndimage.rotate(order=1, mode='reflect')`, center-crops
    to `patch_size × patch_size`
  - Returns same 5-tuple as `PatchDataset`: `(image, condition, ann1, ann2, path)`
  - Patch TIF pixel data is **never read** — only filenames are used for coordinates

#### `subcellae/pipeline/ae_pipeline.py`
- Import: added `JitterCropDataset`
- Added fields to `AEConfig`:
  ```python
  jitter_crop: bool            = False
  jitter_crop_channel: str     = "pax"
  jitter_crop_max_shift: int   = 4
  jitter_crop_max_angle: float = 15.0
  jitter_crop_pad_size: int    = 64
  ```
- Dataset construction (in `run_ae_pipeline`): added `elif cfg.jitter_crop and "frame_dir" in entry`
  branch between `channel_dirs` and the default `PatchDataset` path

#### `scripts/run_ae_from_config.py`
- `patch_dirs` entry parsing: passes through `frame_dir` key when present
- Added parsing of `jitter_crop:` YAML section:
  ```yaml
  jitter_crop:
    enabled       : true
    channel       : "pax"
    max_shift_px  : 4
    max_angle_deg : 15.0
    pad_size      : 64
  ```
- Passes all five fields to `AEConfig(...)`

**How to use in a config:**
```yaml
data:
  patch_dirs:
    - path           : root_folder + "/ae_results/patches/cio_rb/vinc/control/tiff_patches32_mr10"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_rb/vinc/control"
      condition      : 0
      condition_name : "control"

jitter_crop:
  enabled       : true
  channel       : "pax"
  max_shift_px  : 4
  max_angle_deg : 15.0
  pad_size      : 64
```

---

### 2. Patchprep mask_ratio lowered from 0.4 → 0.1

**Motivation:** The 40% cell-mask coverage threshold excluded boundary/nascent FAs.
Coworker needs boundary patches for nascent FA analysis.

**Files changed:** `config/patchprep_config/*_cio_rb.yaml` (all 8: vinc/ppax/pfak/nih3t3 × control/ycomp)
- `mask_ratio: 0.4` → `mask_ratio: 0.1`
- Output dirs renamed: `tiff_patches32` → `tiff_patches32_mr10`, `plot_patches32` → `plot_patches32_mr10`
- `root_folder` corrected to `/net/projects/CLS/lding/data/fa_data_analysis`

**Note:** Patchprep must be re-run to populate `tiff_patches32_mr10/`. In the main
repo this was submitted as SLURM job 934012 via `scripts/sbatch_patchprep_mr10.sh`.
The `patchprep_config/` in this branch may not yet reflect these changes — check
before running.

---

### 3. Source frame extraction pipeline (main repo only)

Full-frame per-channel CIO-RB normalized TIFFs are stored at:
`ae_results/source_frames/cio_rb/{dataset}/{condition}/{condition}_f{idx:04d}_{channel}.tif`

These were extracted by `subcellae/pipeline/frameextract_pipeline.py` +
`scripts/run_frameextract_from_config.py` (scripts exist in main repo, may not
be present here). All 4 datasets × 2 conditions × 4 channels have been extracted
on the cluster (SLURM job 933033, completed).

---

## Key conventions

| Item | Value |
|---|---|
| Patch size | 32×32 px |
| Pad size (patchprep coord offset) | 64 px |
| Jitter context size (ps=32, θ=15°) | 44×44 px → crop to 32×32 |
| Channel layout (all datasets) | ch0=marker, ch1=pax(scale=8), ch2=zyx(scale=5), ch3=act(scale=5) |
| vinc ch0 | vinculin |
| ppax ch0 | pPaxillin |
| pfak ch0 | pFAK |
| nih3t3 ch0 | vinculin (same layout as vinc, different cell line) |
| Augmentation stack (contrastive) | jitter crop → rot90/flip → noise → intensity scale |
| Salt-and-pepper noise | soft: `mean ± std/3` per image (not hard 0/1) |
| Data root (cluster) | `/net/projects/CLS/lding/data/fa_data_analysis` |
