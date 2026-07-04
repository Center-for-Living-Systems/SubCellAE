# Labeling Toolchain — Context File
**Date:** 2026-07-04  
**Repo:** `github.com:Center-for-Living-Systems/SubCellAE`  
**Branch:** `main` (HEAD: `61ae77c`)  
**Working dir:** `/home/lding/lding/gitcode/SubCellAE`

---

## Overview

The goal is to manually label focal adhesion (FA) patches from 4-channel fluorescence microscopy CZIs. Patches were pre-extracted by `patchprep` and packed into HDF5 files for use with a browser-based Panel/Bokeh labeling interface.

### Channel layout (all datasets)
| CZI index | Name | Description |
|-----------|------|-------------|
| ch0 | marker | vinculin (vinc), pFAK (pfak), or pPax (ppax) depending on dataset |
| ch1 | pax | paxillin — **main canvas** |
| ch2 | zyx | zyxin |
| ch3 | act | actin |

---

## Key Scripts

### `scripts/label_patches.py` (698 lines)
Main Panel/Bokeh labeling interface. Launch with:
```bash
python scripts/label_patches.py \
  /mnt/p/Liya/FA_patch_group_label/vinc/vinc_control_label.h5 \
  /mnt/p/Liya/FA_patch_group_label/vinc/vinc_ycomp_label.h5 \
  --port 5007 --serve
```
Then open `http://localhost:5007/vinc_control_label` etc.

**Features:**
- 5 label options: Nascent Adhesion, focal complex, focal adhesion, fibrillar adhesion, No adhesion
- Main canvas (paxillin, 720×720) with patch rectangles overlaid
- 3 side canvases (400×400 each, linked pan/zoom) for other channels
- 4 patch thumbnails (225×250) extracted on click from full-canvas images
- Click = label nearest patch; click outside patch boundary = no-op
- Double-click = remove label
- Resume from previous CSV (auto-populates with most recent matching CSV in H5 folder)
- Save filename: `{h5_stem}_{annotator}_{timestamp}.csv`

**H5 format support:**
1. `pack_patches_label_h5.py` format — `images/raw` + `images/{ch}` per channel + `images/meta` CSV
2. `pack_interactive_h5.py` / `pack_labeler_from_prep.py` format — `images/allch/{group}` per group

**Critical function `load_h5()`** (line ~60):
- Detects format automatically
- For format 1: builds `images_allch` from separate channel arrays using `attrs['channels']` ORDER (not sorted alphabetically — this matters for correct name↔image mapping)
- `_MAIN_CH` is derived dynamically by finding 'paxillin' in `channel_names`

### `scripts/pack_patches_label_h5.py` (230 lines)
Lightweight packer for cluster use. Reads from `patchprep` outputs:
- Patches: `ae_results/patches/cio_rb/{ds}/{cond}/tiff_patches32_label/*.tif`
- Frames: `ae_results/source_frames/cio_rb/{ds}/{cond}/{cond}_f{N}_{ch}.tif`

**Channel order packed:** `["pax", ds_ch, "zyx", "act"]`  
This ensures side panels display **[marker, zyxin, actin]** after excluding paxillin.

**Run on cluster:**
```bash
python scripts/pack_patches_label_h5.py                   # all datasets
python scripts/pack_patches_label_h5.py --datasets vinc   # single dataset
python scripts/pack_patches_label_h5.py --datasets vinc pfak --conditions control
```
Output: `ae_results/patches/cio_rb/{ds}/{ds}_{cond}_label.h5`  
Then rsync to NAS: `/mnt/p/Liya/FA_patch_group_label/{ds}/{ds}_{cond}_label.h5`

### `scripts/pack_interactive_h5.py` (550 lines)
Full-featured packer used when AE latent features / UMAP are available.  
Produces `images/allch/{group}` format (per-group, all channels stacked).

### `scripts/pack_labeler_from_prep.py` (242 lines)
Lightweight packer that reads directly from `patchprep` plot output directories (no cluster needed). Useful for datasets processed locally.
```bash
python scripts/pack_labeler_from_prep.py /path/to/plot_dir \
    --image-folder condition:/path/to/czis \
    --pad-size 64 --image-scale 0.5
```

---

## NAS H5 Files (current state as of 2026-07-04)

Mount: `/mnt/p/Liya/FA_patch_group_label/`  
NAS: `smb://psd-gardelnas.uchicago.edu/expansion/Liya/FA_patch_group_label/`

| Dataset | Condition | H5 file | Channels packed | Notes |
|---------|-----------|---------|-----------------|-------|
| vinc | control | `vinc/vinc_control_label.h5` | pax, act, vinc | **missing zyxin** |
| vinc | ycomp | `vinc/vinc_ycomp_label.h5` | pax, act, vinc | **missing zyxin** |
| pfak | control | `pfak/pfak_control_label.h5` | pax, act, pfak | **missing zyxin** |
| pfak | ycomp | `pfak/pfak_ycomp_label.h5` | pax, act, pfak | **missing zyxin** |
| ppax | control | `ppax/ppax_control_label.h5` | pax, act, ppax | **missing zyxin** |
| ppax | ycomp | `ppax/ppax_ycomp_label.h5` | pax, act, ppax | **missing zyxin** |
| nih3t3 | control | `nih3t3/nih3t3_control_label.h5` | pax, act, vinc | **missing zyxin** |
| nih3t3 | ycomp | `nih3t3/nih3t3_ycomp_label.h5` | pax, act, vinc | **missing zyxin** |

**All H5 files need to be re-packed on the cluster** to include zyxin and fix channel order.

### Existing label CSVs on NAS
- `pfak/`: multiple CSVs from Annabel, Liya (2026-04)
- `ppax/`: CSV from 2026-06-16
- `nih3t3/`: CSVs from 2026-04, 2026-06
- `vinc/`: `vinc_control_label_liya_20260704_0933.csv` (new format filename)

---

## Pending Tasks

### Immediate — on cluster
1. `git pull origin main` on cluster
2. Re-run packer:
   ```bash
   python scripts/pack_patches_label_h5.py
   ```
   This will now include zyxin and use correct channel order `[pax, marker, zyx, act]`.
3. Copy new H5 files to NAS (rsync script: `scripts/packedh5_rsync.sh` if it exists)

### Testing — local
After re-pack and NAS copy:
- Restart labeler on new H5 files
- Verify side canvases show **marker | zyxin | actin** (in that order, left to right)
- Verify patch thumbnails appear on click (4 channels)
- Verify resume CSV auto-populates

### Labeling coordination
- Labeling protocol being finalized (Liya, 2026-07-04)
- Updated patch set with more complete boundary regions coming
- Send protocol + new patches to annotators tomorrow

---

## Known Issues / History

### Channel mismatch (fixed in code, needs re-pack)
Old packer stored channels in insertion order as `['pax', 'act', ds_ch]`.  
`load_h5()` was sorting extra channel keys alphabetically → name↔image index desync.  
Fix: `load_h5()` now reads order from `attrs['channels']`; packer now outputs `[pax, ds_ch, zyx, act]`.

### Zero-padding normalization
`img_meta` groups use `control_f0000` (4-digit padded); `images_allch` keys may be unpadded.  
Fixed with `_norm_fkey()` regex: `re.sub(r'_f0*(\d)', r'_f\1', k)` (label_patches.py line ~148).

### Route collision (fixed)
Multiple H5 files from same folder got same URL route. Fixed to use H5 stem as route (`/vinc_control_label`).

### Patch thumbnails with no `patches/allch` (fixed)
New packer only has `patches/raw` (single channel). Thumbnails now extracted on the fly from `images_allch` at click time.

### Side canvas brightness (fixed)
`_update_ch_canvas()` now normalizes each channel to [0,1] before display.

---

## Frameextract Configs
Located at `config/frameextract_config/`.  
Each config maps CZI channel index → name and sets normalization scale.  
Example (`vinc_control_cio_rb.yaml`):
```yaml
channels:
  - index: 0  name: "vinc"  scale: 5.0
  - index: 1  name: "pax"   scale: 8.0
  - index: 2  name: "zyx"   scale: 5.0
  - index: 3  name: "act"   scale: 5.0
```
Frameextract outputs: `ae_results/source_frames/cio_rb/{ds}/{cond}/{cond}_f{N:04d}_{ch}.tif`

---

## Git State
- **main** is up to date with origin/main (`61ae77c`)
- **exp/training_strategy_sweep** is 12 commits behind main (its work was merged at `66c9813`)
- Stash `On main: exp-branch-leftovers` contains modified config YAMLs from the exp branch (not needed on main)
