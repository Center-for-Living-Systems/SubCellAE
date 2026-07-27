# Context: Interactive Model Viewer — `view_interactive.py`

This document gives Claude Code enough background to continue development of the
interactive model viewer system without re-explaining prior decisions.

---

## What this system is

A Panel + Bokeh web app for exploring AE model outputs (latents, UMAP, predictions,
reconstructions) alongside the raw multi-channel microscopy data.

Two directions of exploration:
- **UMAP → Canvas**: tap a UMAP dot → canvas auto-pans to the patch, detail panel
  updates, side-channel patch thumbnails appear.
- **Canvas → UMAP**: click a patch rectangle on the canvas → UMAP dot highlighted
  with the patch's FA-label colour, detail panel updates.

---

## Files in this system

| Script | Role |
|--------|------|
| `scripts/view_interactive.py` | Panel + Bokeh viewer (this doc) |
| `scripts/pack_model_h5.py` | Pack one model result dir → `model.h5` |
| `scripts/pack_data_h5.py` | Pack patches + images → `data.h5` (shared across models) |

---

## H5 two-file design

### `data.h5` — static, shared across models (built by `pack_data_h5.py`)

```
patches/raw       float32 (N, 32, 32)   all patches, all conditions
images/raw        float32 (M, H, W)     paxillin frames
images/{ch}       float32 (M, H, W)     extra channels: vinc / zyx / act / ppax / pfak
images/meta       bytes (CSV)           group, frame (row index in images/raw),
                                        condition_name, frame_idx
meta/csv          bytes (CSV)           filename, condition_name, group, frame_idx,
                                        canvas_cx, canvas_cy, ps,
                                        mean_intensity, annotation_label, annotation_label_name
attrs: pad_size, image_scale, dataset, n_patches, n_frames, channels (JSON list)
```

`channels` attr is a JSON list like `["pax","vinc","zyx","act"]`.
`images/raw` = first channel (pax). `images/{ch}` = remaining channels (same frame order).

### `model.h5` — per-model outputs (built by `pack_model_h5.py`)

```
meta/csv          bytes (CSV)           filename, z_*, UMAP_1, UMAP_2,
                                        fa_pred, fa_prob_*, pos_pred, pos_prob_*,
                                        annotation_label, split, …
patches/recon     float32 (N, 32, 32)   AE reconstructions
plots/{name}      bytes (PNG)           analysis plots (MSE etc.)
attrs: pad_size, image_scale, result_dir, n_patches, model_name
```

---

## `pack_model_h5.py` — what it handles

Primary CSV candidates (tried in order, first found wins):
1. `{result_dir}/analysis/analysis_results.csv`
2. `{result_dir}/latents_newdata.csv`
3. `{result_dir}/latents.csv`

UMAP (tried in order):
1. `UMAP_1`/`UMAP_2` already in primary CSV → use as-is
2. Merge from `analysis/analysis_results.csv` if different from primary
3. **Load a saved UMAP model** (`umap_all_model.pkl`) from any of:
   `fa_cls_lat8/`, `fa_cls_zproj/`, `fa_cls_zrecon/`, `vis_lat8/`,
   `vis_lat8dist8/`, `analysis/` — transform `z_*` latents with `joblib.load` +
   `umap_model.transform(z_arr)`. Needed when only UMAP PNG plots are saved,
   not coordinates.

FA predictions (first found): `fa_cls_lat8/` → `fa_cls_zproj/` → `fa_cls_zrecon/`
 — files: `predictions_all.csv` or `classification_results.csv`.
 Handles both `proba_*` and `prob_*` column prefixes.

Position predictions: same pattern with `pos_cls_*` subdirs.

Recon patches — two formats handled:
- **New (stacked)**: `recon/patches_recon.tif` + `recon/patches_index.csv`
- **Old (individual TIFFs)**: `recon/patches/recon_{split}_{stem}.tif`
  (or `recon_{stem}.tif` without split prefix). Split comes from `split`
  column in primary CSV.

---

## `load_sources` — data merge logic

Two-file mode (`data.h5` + `model.h5`):
- Images + raw patches come from `data.h5` (higher quality, all conditions)
- Model df is left-joined with `data.h5` `meta/csv` on `filename` to get
  `canvas_cx`, `canvas_cy`, `ps`, `mean_intensity`, `annotation_label`
- `patches_raw` is re-indexed from `data.h5` row order to model df row order;
  unmatched model patches get zero arrays (warning printed, no crash)
- Extra channel images loaded from `data.h5`; `ch_keys` from `channels` attr

Mismatched patches (model has patches not in data.h5, or vice versa):
- Model-only patches: NaN `canvas_cx`/`canvas_cy` → skipped in rect drawing
  and side-patch crops; no crash
- Data-only patches: don't appear in UMAP at all

---

## Viewer layout

```
pn.Column(
  header,
  pn.Row(
    left_canvas_col,          # fixed structure — extra channels nested here
    Spacer(12, sizing_mode='fixed'),
    detail_col,               # dynamic content — grows on click
    Spacer(8, sizing_mode='fixed'),
    side_patch_col,           # 3 extra-channel patch crops, appears on click
  )
)

left_canvas_col = pn.Column(
  pn.Row(left_col, Spacer(12), canvas_col),   # UMAP | pax canvas
  pn.Row(*extra_ch_panes),                     # 3 linked channel canvases below
)
```

**Why nested like this**: `detail_col` and `side_patch_col` grow when a patch is
clicked. If `extra_ch_panes` were a sibling of `detail_col` in the outer Column,
they would shift down when `detail_col` grew. Nesting them inside `left_canvas_col`
(which has fixed content) isolates them from that height change.

**Spacers must use `sizing_mode='fixed'`** — the global
`pn.extension(sizing_mode='stretch_width')` would otherwise make them expand and
create a large gap between left_canvas_col and detail_col.

### left_col (non data-only mode)
- `color_select` Bokeh Select (FA type / Position) with a JS callback that
  swaps the `color` column in `umap_src.data` client-side
- `umap_filter` Panel RadioButtonGroup ("All patches" / "Labeled only")
  — "Labeled only" filters to `annotation_label >= 0` (human labels from
  `data.h5`, NOT predictions)
- UMAP Bokeh scatter (500×500px) with tap + hover tools

### canvas_col (pax canvas)
- Image selector dropdown (condition | group_key)
- Dim toggle: vmax=1.0 ↔ 2.0, updates `gray_mapper` + all `extra_ch_mappers`
- Bokeh canvas 520×520px with `_flip_for_bokeh` (flipud so row-0 at top)
- Patch rect overlay: coloured by max intensity (>4 red, >2 magenta, else hidden)
- White border rect on selected patch (`sel_src`) — also drawn on extra ch canvases

### extra_ch_panes (3 × 340px Bokeh figures)
- Share `x_range`/`y_range` with pax figure → synchronized pan/zoom
- `sel_src` rect renderer added to each → selection box synced automatically
- Updated in `_load_group` by reading `extra_ch_images[k][frame_row]`

### detail_col (right panel)
- Prediction text (FA type + position, colour-coded)
- L1 stat line: `L1: 0.1823 → p64 | p10=… p50=… p90=…` — computed at startup
  from `patch_l1s = |patches_raw|.mean(axis=(1,2))`
- Raw + Recon patch figure (matplotlib, 2.6×2.6 in each panel)
- Legend (all FA + position labels with colour swatches)
- MSE plot button (opens PNG in new window)

### side_patch_col (165px column)
- 3 matplotlib thumbnails (1.56×1.56 in each) — crops of extra channel frames
  at the selected patch's `canvas_cx`, `canvas_cy`, `ps` coordinates
- Updated in `_show_detail`; empty before first click

---

## Key coordinate conventions

```
canvas_cx, canvas_cy  = patch centre in unpadded canvas space (px)
                        patch covers [cy-ps//2 : cy+ps//2, cx-ps//2 : cx+ps//2]

Bokeh display (flipud):
  bx = canvas_cx * image_scale
  by = (H - canvas_cy) * image_scale
```

Auto-pan on UMAP click keeps equal x/y span (preserves aspect ratio):
```python
xe = min(W_sc, bx + margin)
xs = max(0,    xe - 2*margin)   # slide start back, don't clip the span
ye = min(H_sc, by + margin)
ys = max(0,    ye - 2*margin)
```

---

## Channel label format

Module-level dicts used in both data_viewer.py and view_interactive.py:
```python
_CH_IDX   = {'pax': 1, 'zyx': 2, 'act': 3, 'vinc': 0, 'ppax': 0, 'pfak': 0}
_CH_SHORT  = {'pax': 'pax', 'zyx': 'zyxin', 'act': 'actin', 'vinc': 'vinc', …}

def _ch_label(key): return f'{_CH_SHORT.get(key,key)}-ch{_CH_IDX.get(key,"?")}'
# → 'pax-ch1', 'vinc-ch0', 'zyxin-ch2', 'actin-ch3'
```

---

## Robustness — mismatched patches

Patches in model but not in data.h5:
- `canvas_cx`/`canvas_cy` are NaN after left join → skipped in rect drawing
- `_on_umap_tap`: `if not pd.isna(cx) and not pd.isna(cy)` guards canvas update
- `patches_raw[idx]` = zeros → blank patch image shown (no crash)

Groups in model but not in `img_meta`:
- `_unique_groups_set` checked before calling `_load_group`; warning printed if missing
- `_get_canvas` returns `None` if `_get_frame` finds no match; `_load_group` returns early

Frame index out of bounds for extra channels:
- `if frame_row >= len(frames): continue` (warning printed)

Empty patch crop (near-edge coordinates):
- `if patch.size == 0: continue` (warning printed)

---

## UMAP filter internals

Two filters compose via AND:

```python
_umap_base = {k: np.array(v) for k, v in umap_data.items()}   # full data

def _compute_umap_mask():
    # Dataset filter
    val = ds_select.value   # 'Training' | 'All' | 'vinc' | 'ppax' | ...
    if val == 'Training':
        ds_mask = np.isin(_umap_base['split'], ['train', 'val'])
    elif val == 'All':
        ds_mask = np.ones(n, dtype=bool)
    else:
        ds_mask = (_umap_base['dataset'] == val)
    # Labeled-only toggle
    lab_mask = labeled_mask if umap_filter.value == 'Labeled only' else np.ones(n, dtype=bool)
    return ds_mask & lab_mask

def _update_umap_src(event=None):
    idxs = np.where(_compute_umap_mask())[0]
    umap_src.data = {k: v[idxs] for k, v in _umap_base.items()}
```

`umap_data` dict now includes `dataset`, `split`, `color_ds` columns in addition to
the old `color_fa`, `color_pos`, `idx`, `filename`, `condition`, `fa_pred`, `pos_pred`.

`_on_umap_tap` uses `int(umap_src.data['idx'][new[0]])` (not `int(new[0])`)
so it always gets the original df row index regardless of which filter is active.

---

## Running the viewer

### CLI syntax

```bash
# Single model — 4 datasets
python scripts/view_interactive.py \
    /path/patches/cio/vinc/data.h5 \
    /path/patches/cio/ppax/data.h5 \
    /path/patches/cio/pfak/data.h5 \
    /path/patches/cio/nih3t3/data.h5 \
    --model /path/contrastive_run/<model>/model.h5

# Multiple models — shows a button bar at the top for switching
python scripts/view_interactive.py \
    /path/patches/cio/vinc/data.h5 \
    /path/patches/cio/ppax/data.h5 \
    /path/patches/cio/pfak/data.h5 \
    /path/patches/cio/nih3t3/data.h5 \
    --model /path/contrastive_run/modelA/model.h5 \
    --model /path/contrastive_run/modelB/model.h5 \
    --model /path/contrastive_run/modelC/model.h5

# Legacy test_run_overfit_20260322 (single pax channel, 32x32 patches)
python scripts/view_interactive.py \
    /path/test_run_overfit_20260322/data.h5 \
    --model /path/test_run_overfit_20260322/baseline/model.h5 \
    --model /path/test_run_overfit_20260322/semisup_fa/model.h5 \
    --model /path/test_run_overfit_20260322/semisup_pos/model.h5 \
    --model /path/test_run_overfit_20260322/semisup_both/model.h5

# Network-accessible (bind to 0.0.0.0)
python scripts/view_interactive.py ... --port 5006 --serve

# No args → loader UI (paste paths into text boxes)
python scripts/view_interactive.py
```

### New viewer features (as of 2026-07-26)

- **Dataset filter dropdown** (`ds_select`): default = Training (split ∈ {train, val}).
  Options: Training | vinc | ppax | pfak | nih3t3 | All. Composes with labeled-only toggle.
- **Colour by Dataset**: new option in the Bokeh colour-by selector (blue/orange/green/red per dataset).
- **Multi-model button bar**: `build_multi_app` wraps `build_app`; data.h5 files are cached
  so switching between models is instant after first load.
- **Multi-line model input** in loader UI: one model.h5 path per line.

### Cluster paths (DSI cluster)

| Path | Contents |
|------|----------|
| `/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio/{vinc,ppax,pfak,nih3t3}/data.h5` | CIO-normalised patches + images |
| `/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio_rb/{vinc,ppax,pfak,nih3t3}/data.h5` | CIO-RB-normalised patches + images |
| `/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run/*/model.h5` | Flat ConAE / SupConAE models |
| `/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run/ds_combo_*/*/model.h5` | ds_combo models |
| `/net/projects/CLS/lding/data/fa_data_analysis/ae_results/test_run_overfit_20260322/data.h5` | Legacy pax-only dataset (per-image norm, 32×32) |
| `/net/projects/CLS/lding/data/fa_data_analysis/ae_results/test_run_overfit_20260322/{baseline,semisup_fa,semisup_pos,semisup_both}/model.h5` | Legacy semisup models |

---

## Rsync commands (run from local Ubuntu machine)

Cluster login: `liyading@login.ds.uchicago.edu`
Local base: `/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results`

### Group 1 — cio data.h5 (4 datasets, ~1.5 GB) — ready now

```bash
REMOTE=liyading@login.ds.uchicago.edu
LOCAL=/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results

for ds in vinc ppax pfak nih3t3; do
    mkdir -p $LOCAL/patches/cio/$ds
    rsync -avh --progress \
        $REMOTE:/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio/$ds/data.h5 \
        $LOCAL/patches/cio/$ds/
done
```

### Group 2 — legacy test_run_overfit_20260322 (~237 MB) — ready now

```bash
REMOTE=liyading@login.ds.uchicago.edu
LOCAL=/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results
SRC=/net/projects/CLS/lding/data/fa_data_analysis/ae_results/test_run_overfit_20260322

mkdir -p $LOCAL/test_run_overfit_20260322/{baseline,semisup_fa,semisup_pos,semisup_both}

rsync -avh --progress \
    $REMOTE:$SRC/data.h5 \
    $LOCAL/test_run_overfit_20260322/

for m in baseline semisup_fa semisup_pos semisup_both; do
    rsync -avh --progress \
        $REMOTE:$SRC/$m/model.h5 \
        $LOCAL/test_run_overfit_20260322/$m/
done
```

### Group 3 — contrastive_run model.h5 files — wait for SLURM job 1226459 to finish

```bash
REMOTE=liyading@login.ds.uchicago.edu
LOCAL=/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results
SRC=/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run

# Flat models (contrastive_cio_*, supcon_cio_*)
rsync -avh --progress \
    --include="*/" \
    --include="model.h5" \
    --exclude="*" \
    $REMOTE:$SRC/ \
    $LOCAL/contrastive_run/

# ds_combo nested models (ds_combo_*/combo_name/model.h5)
for parent in ds_combo_enlcrop_clip01_l1 ds_combo_enlcrop_sc2_clip02_l1 \
              ds_combo_enlcrop_sc2 ds_combo_enlcrop_sc2_lc010_bal \
              ds_combo_enlcrop_sc2_lc010_bal_l1 ds_combo_enlcrop_sc2_lc010_bal_mse; do
    rsync -avh --progress \
        --include="*/" \
        --include="model.h5" \
        --exclude="*" \
        $REMOTE:$SRC/$parent/ \
        $LOCAL/contrastive_run/$parent/
done
```

---

## Things to know / traps

- **`pn.extension(sizing_mode='stretch_width')` is global** — any `pn.Spacer`
  without `sizing_mode='fixed'` will stretch and create unwanted gaps.
- **Extra channel canvases are in `left_canvas_col`**, not in the outer Column.
  Don't move them to a sibling row of `detail_col` — that causes the
  "shift on click" layout bug.
- **`_load_group` clears `highlight_src` with 3 keys**: `dict(x=[], y=[], color=[])`
  (not just `x`, `y`) — the UMAP highlight scatter uses a `color` column.
- **UMAP filter swaps entire `umap_src.data`** — the `idx` column in the
  filtered subset still holds original df row indices, so tap handlers work.
- **`dim_toggle` affects all mappers**: `gray_mapper` (pax) + every entry in
  `extra_ch_mappers` list.
- **`_refresh_detail` must be called after `dim_toggle` changes** so the
  currently-shown patch thumbnails re-render at the new vmax.
- **Recon patches are read from `model.h5 patches/recon`** at startup — no
  on-the-fly computation.
- **`pack_model_h5.py` old-format recon**: looks for
  `recon/patches/recon_{split}_{stem}.tif` (with optional split prefix).
  `split` comes from the `split` column in the primary latents CSV.
- **UMAP pkl transform**: `joblib.load` (not `pickle.load`) — the pkl format
  is protocol 4 and joblib handles it correctly.
