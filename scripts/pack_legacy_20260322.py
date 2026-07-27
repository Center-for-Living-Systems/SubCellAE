#!/usr/bin/env python3
"""
pack_legacy_20260322.py
=======================
Pack test_run_overfit_20260322 results into the current viewer H5 format.

Creates:
  <ROOT>/data.h5                      -- shared: patches + images (per-image norm [0,1])
  <ROOT>/baseline/model.h5            -- latents + UMAP + plots
  <ROOT>/semisup_fa/model.h5
  <ROOT>/semisup_pos/model.h5
  <ROOT>/semisup_both/model.h5

Normalization note:
  Patches were created on 2026-03-22 with per-image normalization (each full-field
  image divided by its own max before patch extraction).  Values are float32 in [0,1].
  No CIO or /65535 normalization was applied.

Usage:
  python scripts/pack_legacy_20260322.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import tifffile

ROOT   = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/test_run_overfit_20260322")
MODELS = ["baseline", "semisup_fa", "semisup_pos", "semisup_both"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _pack_csv(hf: h5py.File, key: str, src: Path | pd.DataFrame | str) -> bool:
    if isinstance(src, pd.DataFrame):
        text = src.to_csv(index=False)
    elif isinstance(src, Path):
        if not src.exists():
            return False
        text = src.read_text()
    else:
        text = str(src)
    hf.create_dataset(key, data=np.bytes_(text))
    return True


def _pack_png(hf: h5py.File, key: str, path: Path) -> None:
    hf.create_dataset(key, data=np.frombuffer(path.read_bytes(), dtype=np.uint8))


def _load_umap_pkl(pkl_path: Path) -> object | None:
    """Try loading an existing UMAP pkl; return None on version-mismatch errors."""
    if not pkl_path.exists():
        return None
    try:
        m = joblib.load(str(pkl_path))
        print(f"  Loaded existing UMAP: {pkl_path.name}")
        return m
    except Exception as e:
        print(f"  Cannot load {pkl_path.name} ({type(e).__name__}): {e!s:.120}")
        return None


def _fit_umap(z: np.ndarray, out_pkl: Path,
              existing_pkls: list[Path] | None = None) -> np.ndarray:
    """Load existing UMAP model and transform, or re-fit if loading fails.

    Tries each path in existing_pkls in order before fitting from scratch.
    Saves a fresh pkl at out_pkl (compatible with current env).
    """
    from umap import UMAP

    z = z.astype(np.float32)

    # Try existing pkls first (may fail due to numba version mismatch)
    for candidate in (existing_pkls or []):
        reducer = _load_umap_pkl(candidate)
        if reducer is not None:
            try:
                emb = reducer.transform(z)
                joblib.dump(reducer, str(out_pkl))
                return emb
            except Exception as e:
                print(f"  transform() failed ({e!s:.80}), will re-fit")

    # Re-fit (deterministic with random_state=42)
    print(f"  Fitting UMAP on {z.shape}  [random_state=42, reproducible] …")
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(z)
    joblib.dump(reducer, str(out_pkl))
    print(f"  UMAP done  → {out_pkl.name}")
    return emb


# ── 1. data.h5  (shared, model-agnostic) ──────────────────────────────────────

def pack_data_h5() -> None:
    out = ROOT / "data.h5"
    # All 4 models share the same raw patches/images — use semisup_fa as source.
    src = ROOT / "semisup_fa"

    df = pd.read_csv(
        src / "latents.csv",
        usecols=["filename", "condition", "condition_name", "group", "split"],
    )
    n = len(df)
    print(f"[data.h5] {n} patches, building array …")

    patch_dir = src / "recon" / "patches"
    patches = np.zeros((n, 32, 32), dtype=np.float32)
    missing = 0
    for i, row in df.iterrows():
        fname = f"raw_{row['split']}_{row['filename']}"
        p = patch_dir / fname
        if p.exists():
            patches[i] = tifffile.imread(str(p)).astype(np.float32)
        else:
            missing += 1
    if missing:
        print(f"  WARNING: {missing} patch files not found (left as zeros)")

    # Full-field images
    image_dir  = src / "recon" / "images"
    img_files  = sorted(image_dir.glob("raw_*.tif"))
    images_raw = np.stack(
        [tifffile.imread(str(f)).astype(np.float32) for f in img_files], axis=0
    )
    img_meta = pd.DataFrame([
        {"frame": i, "group": f.stem[4:]}   # strip leading "raw_"
        for i, f in enumerate(img_files)
    ])
    print(f"[data.h5] images {images_raw.shape}")

    with h5py.File(str(out), "w") as hf:
        _pack_csv(hf, "meta/csv", df)
        hf.create_dataset("patches/raw", data=patches,
                          compression="gzip", compression_opts=4)
        hf.create_dataset("images/raw",  data=images_raw,
                          compression="gzip", compression_opts=4)
        _pack_csv(hf, "images/meta", img_meta)
        hf.attrs["pad_size"]    = 32.0
        hf.attrs["image_scale"] = 1.0
        hf.attrs["channels"]    = json.dumps(["pax"])

    print(f"[data.h5] → {out}  ({out.stat().st_size/1e6:.1f} MB)")


# ── 2. model.h5 per model ─────────────────────────────────────────────────────

def pack_model_h5(model_name: str) -> None:
    model_dir = ROOT / model_name
    out = model_dir / "model.h5"

    df = pd.read_csv(model_dir / "latents.csv")
    print(f"[{model_name}] {len(df)} rows, cols: {list(df.columns)}")

    # Compute UMAP on z_* latent dims
    # Prefer existing pkl from fa_cls_lat8/ (same latent space); re-fit if version mismatch.
    z_cols = [c for c in df.columns if c.startswith("z_")]
    if z_cols:
        existing_pkls = [
            model_dir / "fa_cls_lat8"      / "umap_all_model.pkl",
            model_dir / "fa_cls_lat8dist8" / "umap_all_model.pkl",
        ]
        emb = _fit_umap(df[z_cols].values,
                        model_dir / "umap_model.pkl",
                        existing_pkls=existing_pkls)
        df["UMAP_1"] = emb[:, 0]
        df["UMAP_2"] = emb[:, 1]
    else:
        print(f"  WARNING: no z_* cols found in latents.csv")

    with h5py.File(str(out), "w") as hf:
        _pack_csv(hf, "meta/latents_csv", df)

        # Cluster predictions (fa_cls_lat8/predictions_all.csv)
        # pred_label → fa_pred equivalent for the viewer colour-by
        pred_csv = model_dir / "fa_cls_lat8" / "predictions_all.csv"
        if pred_csv.exists():
            pred_df = pd.read_csv(pred_csv)
            # Rename to match viewer expectations: pred_label → fa_pred
            pred_df = pred_df.rename(columns={"pred_label": "fa_pred"})
            _pack_csv(hf, "meta/cluster_labels_csv", pred_df)
            print(f"  cluster_labels (fa_cls_lat8 predictions): {len(pred_df)} rows")

        # PNGs: model root
        n_plots = 0
        for png in sorted(model_dir.glob("*.png")):
            _pack_png(hf, f"plots/{png.stem}", png)
            n_plots += 1

        # PNGs: classification sub-folders (umap + confusion matrices etc.)
        for sub in ("fa_cls_lat8", "pos_cls_lat8",
                    "fa_cls_lat8dist8", "pos_cls_lat8dist8"):
            subdir = model_dir / sub
            if subdir.is_dir():
                for png in sorted(subdir.glob("*.png")):
                    _pack_png(hf, f"plots/{sub}__{png.stem}", png)
                    n_plots += 1

        print(f"  plots: {n_plots} PNGs packed")

        hf.attrs["model_name"] = model_name
        hf.attrs["result_dir"] = str(model_dir)
        hf.attrs["n_patches"]  = int(len(df))

    print(f"[{model_name}] → {out}  ({out.stat().st_size/1e6:.1f} MB)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Packing data.h5")
    print("=" * 60)
    pack_data_h5()

    for m in MODELS:
        print()
        print("=" * 60)
        print(f"Packing {m}/model.h5")
        print("=" * 60)
        pack_model_h5(m)

    print("\nAll done.")
    print()
    print("Viewer usage:")
    print(f"  python scripts/view_interactive.py \\")
    print(f"    {ROOT}/data.h5 \\")
    for m in MODELS:
        print(f"    --model {ROOT}/{m}/model.h5 \\")


if __name__ == "__main__":
    main()
