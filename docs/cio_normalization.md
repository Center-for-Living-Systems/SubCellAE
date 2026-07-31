# CIO Patch Normalisation — Design Log

All variants normalise a raw channel image `I` using masks derived from
paxillin segmentation:

- **`seg`** — main cell mask (centre-region cell only, `min_size_final = 30 000`)
- **`true_bg`** — true background pixels: `~seg_all`, where `seg_all` is the
  *broad exclusion mask* (all cell-like regions including partial border cells,
  `min_size_final = 1 000`, no centre-region filter).  Earlier runs used
  `~seg` as background, which contaminated background statistics with partial
  cells at image borders.

The scale factor **×5** (where present) is a historical convention that places
the trimmed-mean cell intensity at 0.2, leaving headroom to 1.0 for brighter
FA structures.

---

## Variants

### `cio_inlier`

```
normalised = (I - trimmed_mean(I[true_bg])) / (trimmed_mean(I[seg]) × 5)
```

- **Background**: trimmed mean of `true_bg` pixels (1–99 %)
- **Cell reference**: trimmed mean of `seg` pixels (1–99 %) × 5
- Background can drift above 0 if fluorescence haze fills the image; not
  robust to datasets with high out-of-cell fluorescence.

| dataset | min | p1 | p95 | p99 | p99.5 | p99.8 |
|---------|-----|----|-----|-----|-------|-------|
| vinc    | −0.007 | −0.005 | 0.51 | 0.76 | 0.88 | 1.05 |
| nih3t3  | −0.011 | −0.007 | 0.82 | 1.74 | 2.22 | 2.97 |
| ppax    | −0.011 | −0.007 | 0.76 | 1.60 | 2.00 | 2.53 |
| pfak    | −0.012 | −0.010 | 0.85 | 1.83 | 2.22 | 2.73 |

---

### `cio_med`

```
normalised = (I - median(I[true_bg])) / (median(I[seg]) × 5)
```

- **Background**: median of `true_bg` pixels
- **Cell reference**: median of `seg` pixels × 5
- Median background is much higher than the mode (median pulls toward
  fluorescence haze); cell median is lower than trimmed mean, so the
  effective scale is stretched → large upper-tail values.  Not recommended.

| dataset | min | p95 | p99 | p99.8 |
|---------|-----|-----|-----|-------|
| vinc    | −0.004 | 0.65 | 1.04 | 1.57 |
| nih3t3  | −0.008 | 1.79 | 3.97 | 7.11 |
| ppax    | −0.007 | 1.42 | 3.26 | 5.55 |
| pfak    | −0.008 | 1.86 | 4.10 | 6.25 |

---

### `cio_mode`

```
normalised = (I - mode(I[true_bg])) / (trimmed_mean(I[seg]) × 5)
```

- **Background**: most-frequent 16-bit count in `true_bg` pixels, i.e. the
  camera dark-current floor (≈ 24 counts for vinc, ≈ 15 for nih3t3 in 2^16
  units).
- **Cell reference**: trimmed mean of `seg` pixels (1–99 %) × 5
- Mode background gives near-zero negatives and puts boundary (empty) patches
  at ≈ 0 rather than slightly negative.
- Upper tail similar to `cio_inlier`; p99.8 reaches 1–3 × across datasets.

| dataset | min | p1 | p95 | p99 | p99.5 | p99.8 |
|---------|-----|----|-----|-----|-------|-------|
| vinc    | −0.002 | −0.000 | 0.51 | 0.75 | 0.86 | 1.03 |
| nih3t3  | −0.004 | −0.001 | 0.81 | 1.70 | 2.18 | 2.91 |
| ppax    | −0.003 | −0.001 | 0.74 | 1.57 | 1.96 | 2.48 |
| pfak    | −0.002 | −0.001 | 0.83 | 1.79 | 2.17 | 2.66 |

---

### `cio_mode_prt` ✓ preferred

```
normalised = (I - mode(I[true_bg])) / (mean(I[seg] | P95 < I < P99.5) - mode(I[true_bg]))
```

- **Background**: mode of `true_bg` pixels (camera dark-current floor, per channel)
- **Cell reference**: mean of cell pixels strictly between their own P95 and
  P99.5 — anchors the scale to the brightest non-saturated FA-rich pixels.
- **No ×5 factor**: the P95–P99.5 mean is already 4–5 × the trimmed mean, so
  the resulting range is comparable to `cio_inlier × 5` without needing a
  hand-tuned constant.
- Naturally self-calibrating across datasets: p99 clusters at ≈ 1.25–1.30
  across nih3t3 / ppax / pfak (vs 1.57–1.79 spread for `cio_mode`).
- Applied to **all channels** (pax, marker, zyxin, actin) independently,
  each channel using its own background and cell statistics with the shared
  segmentation mask.

| dataset | min | p1 | p95 | p99 | p99.5 | p99.8 |
|---------|-----|----|-----|-----|-------|-------|
| vinc    | −0.002 | −0.001 | 0.81 | 1.15 | 1.27 | 1.43 |
| nih3t3  | −0.003 | −0.001 | 0.63 | 1.30 | 1.65 | 2.18 |
| ppax    | −0.002 | −0.001 | 0.64 | 1.29 | 1.58 | 1.98 |
| pfak    | −0.002 | −0.000 | 0.62 | 1.29 | 1.56 | 1.88 |

---

## Background mask evolution

| Version | Background definition | Problem |
|---------|-----------------------|---------|
| Early | `~seg` (invert main cell mask) | Border/partial cells included → background median elevated |
| Current | `~seg_all` (broad exclusion mask, `min_size=1000`, no centre filter) | All cell-like regions removed; true detector floor visible |

The broad exclusion mask (`_segment_exclude_mask`) mirrors `segment_cell_mask`
steps 0–8 but skips step 9 (centre-region filter) so partial cells at image
borders are masked out of background statistics.

---

## HDF5 layout (current)

```
patches/raw               (N, 32, 32)   paxillin TIF patches (patchprep output)
patches/{norm}/{ch}       (N, 32, 32)   norm ∈ {cio_inlier, cio_med, cio_mode, cio_mode_prt}
                                         ch  ∈ {pax, vinc/pfak/ppax, zyx, act}
images/{norm}/{ch}        (M, H, W)     full-frame canvas per norm × channel
images/meta               bytes (CSV)   frame metadata
meta/csv                  bytes (CSV)   per-patch metadata
```
