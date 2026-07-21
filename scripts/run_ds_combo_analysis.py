#!/usr/bin/env python3
"""
run_ds_combo_analysis.py

UMAP + KMeans cluster panels for a single ds-combo model dir.
Reads latents.csv (generated during training), fits UMAP on z_proj
(or z_recon if no projector), encodes test-set patches with the trained
model, then generates:

  eval/umap_combo_condition.png          — training conditions only
  eval/umap_combo_condition_with_test.png — training + test conditions overlaid
  eval/umap_combo_split.png
  eval/cluster_panels_combo/
    cluster_labels.csv
    cluster_XX_nNNN.tif
    all_clusters.tif
    umap_combo_kmeans_k10.png

Usage:
  python scripts/run_ds_combo_analysis.py <model_dir>
  python scripts/run_ds_combo_analysis.py <model_dir> --k 10 --n-panel 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from sklearn.cluster import KMeans
from umap import UMAP

sys.path.insert(0, str(Path(__file__).parents[1]))

SPLIT_COLOR = {"train": "#333333", "val": "#AAAAAA"}

# Max patches per condition for UMAP fitting — keeps all conditions equally weighted
UMAP_MAX_PER_COND = 3000
# Max patches per test condition to encode (transform only, so can be more)
TEST_MAX_PER_COND = 5000

ROOT_FOLDER = Path("/net/projects/CLS/lding/data/fa_data_analysis")

# (dataset_name, condition_name, patch_dir relative to ROOT_FOLDER)
EXTERNAL_DATASETS = [
    ("vinc",   "control", "ae_results/pax_ch_patch/cio_rb/vinc/control/tiff_patches32"),
    ("vinc",   "ycomp",   "ae_results/pax_ch_patch/cio_rb/vinc/ycomp/tiff_patches32"),
    ("pfak",   "control", "ae_results/pax_ch_patch/cio_rb/pfak/control/tiff_patches32"),
    ("pfak",   "ycomp",   "ae_results/pax_ch_patch/cio_rb/pfak/ycomp/tiff_patches32"),
    ("ppax",   "control", "ae_results/pax_ch_patch/cio_rb/ppax/control/tiff_patches32"),
    ("ppax",   "ycomp",   "ae_results/pax_ch_patch/cio_rb/ppax/ycomp/tiff_patches32"),
    ("nih3t3", "control", "ae_results/pax_ch_patch/cio_rb/nih3t3/control/tiff_patches32"),
    ("nih3t3", "ycomp",   "ae_results/pax_ch_patch/cio_rb/nih3t3/ycomp/tiff_patches32"),
]

KNOWN_DATASETS = {"vinc", "nih3t3", "ppax", "pfak"}

_DS_SHORT = {"vinc": "ds1", "pfak": "ds2", "ppax": "ds3", "nih3t3": "ds4"}
_COND_SHORT = {"control": "ctrl", "ycomp": "yc"}

# Fixed tab20 color assignment: same condition → same color across all plots
COND_ORDER = [
    "ds1_ctrl", "ds1_yc",
    "ds2_ctrl", "ds2_yc",
    "ds3_ctrl", "ds3_yc",
    "ds4_ctrl", "ds4_yc",
]


def _cond_label(cond: str) -> str:
    """'vinc_control' → 'ds1_ctrl', 'nih3t3_ycomp' → 'ds4_yc'"""
    for ds, short in _DS_SHORT.items():
        if cond.startswith(ds):
            suffix = cond[len(ds):].lstrip("_")
            suffix = _COND_SHORT.get(suffix, suffix)
            return f"{short}_{suffix}" if suffix else short
    return cond


def _cond_color(lbl: str):
    """Return fixed tab20 color for a condition label."""
    tab20 = plt.get_cmap("tab20")
    idx = COND_ORDER.index(lbl) if lbl in COND_ORDER else 8
    return tab20(idx)


# ── config helpers ────────────────────────────────────────────────────────────

def _read_model_config(model_dir: Path) -> dict:
    try:
        import yaml
        for yf in list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml")):
            with open(yf) as fh:
                return yaml.safe_load(fh) or {}
    except Exception:
        pass
    return {}


def _dataset_from_condition(condition_name: str) -> str:
    for ds in KNOWN_DATASETS:
        if condition_name == ds or condition_name.startswith(ds + "_"):
            return ds
    return condition_name.split("_")[0]


def _get_training_datasets(model_dir: Path) -> set[str]:
    cfg = _read_model_config(model_dir)
    datasets: set[str] = set()
    for entry in cfg.get("data", {}).get("patch_dirs", []):
        cond = str(entry.get("condition_name", ""))
        ds = _dataset_from_condition(cond)
        if ds in KNOWN_DATASETS:
            datasets.add(ds)
    if not datasets:
        # fallback: infer from combo dir name
        combo = model_dir.name
        for part in combo.split("_"):
            if part in KNOWN_DATASETS:
                datasets.add(part)
    return datasets or {"vinc"}


def _read_input_divisor(model_dir: Path) -> float:
    cfg = _read_model_config(model_dir)
    ec = cfg.get("enlarged_crop", {})
    if ec.get("enabled", False):
        return float(ec.get("input_divisor", 1.0))
    return 1.0


# ── test-set latent extraction ────────────────────────────────────────────────

def _encode_test_latents(model_dir: Path,
                          training_datasets: set[str],
                          feat_dim: int,
                          use_projector: bool,
                          device: str = "cpu",
                          rng: np.random.Generator | None = None,
                          ) -> dict[str, np.ndarray]:
    """Encode test-dataset patches with the trained model.

    Returns dict: condition_label → latent array (N, feat_dim).
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from subcellae.modelling.dataset import PatchDataset
    except ImportError as e:
        print(f"  WARNING: torch unavailable ({e}) — skipping test inference")
        return {}

    model_pt = model_dir / "model_final.pt"
    if not model_pt.exists():
        model_pt = model_dir / "model_best.pt"
    if not model_pt.exists():
        print("  WARNING: no model checkpoint — skipping test inference")
        return {}

    input_divisor = _read_input_divisor(model_dir)
    print(f"  Loading model for test inference (÷{input_divisor}) …", flush=True)
    model = torch.load(str(model_pt), map_location=device, weights_only=False)
    model.eval()
    has_projector = hasattr(model, "project") and use_projector

    result: dict[str, np.ndarray] = {}
    rng = rng or np.random.default_rng(42)

    for ds_name, cond_name, rel_path in EXTERNAL_DATASETS:
        if ds_name in training_datasets:
            continue
        patch_dir = ROOT_FOLDER / rel_path
        if not patch_dir.exists():
            print(f"    [skip] {ds_name}/{cond_name} — dir not found")
            continue

        lbl = _cond_label(f"{ds_name}_{cond_name}")
        ds = PatchDataset(str(patch_dir), condition=0, condition_name="")
        if len(ds) == 0:
            continue

        loader = DataLoader(ds, batch_size=256, shuffle=False,
                            drop_last=False, num_workers=0)
        latents = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0]
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                if input_divisor != 1.0:
                    x = x / input_divisor
                x = x.to(device)
                z = model.encode(x)
                if has_projector:
                    z = model.project(z)
                latents.append(z.cpu().numpy())

        arr = np.concatenate(latents, axis=0).astype(np.float32)
        if len(arr) > TEST_MAX_PER_COND:
            idx = rng.choice(len(arr), TEST_MAX_PER_COND, replace=False)
            arr = arr[idx]
        result[lbl] = arr
        print(f"    {lbl}: {len(arr)} patches encoded")

    del model
    return result


# ── scatter helpers ───────────────────────────────────────────────────────────

def _scatter_conditions(ax, emb: np.ndarray, labels: list[str],
                         marker: str = "o", s: float = 3,
                         alpha: float = 0.4, hollow: bool = False) -> None:
    """Plot each unique condition label onto ax with fixed tab20 colors."""
    unique_labels = list(dict.fromkeys(labels))  # preserves order, deduplicates
    label_arr = np.array(labels)
    for lbl in unique_labels:
        mask = label_arr == lbl
        color = _cond_color(lbl)
        if hollow:
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       facecolors="none", edgecolors=[color],
                       marker=marker, s=s, alpha=alpha, linewidths=0.8,
                       label=lbl + " [test]")
        else:
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[color], marker=marker, s=s,
                       alpha=alpha, linewidths=0,
                       label=lbl)


# ── cluster panels ────────────────────────────────────────────────────────────

def _make_cluster_panels(labels: np.ndarray, patch_paths: list[Path],
                          out_dir: Path, k: int, n_panel: int,
                          umap_emb: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_tifs = []
    all_rows = []

    for c in range(k):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        sel = idx[:n_panel]
        imgs = []
        for i in sel:
            p = patch_paths[i]
            if p.exists():
                img = tifffile.imread(str(p)).astype(np.float32)
                if img.ndim == 3:
                    img = img[0]
                mx = float(np.percentile(img, 99.9))
                if mx > 0:
                    img = np.clip(img, 0, mx) / mx
                imgs.append((img * 255).astype(np.uint8))
        if not imgs:
            continue

        ps = imgs[0].shape[0]
        cols = min(n_panel, len(imgs))
        panel = np.zeros((ps, ps * cols), dtype=np.uint8)
        for j, img in enumerate(imgs):
            panel[:, j*ps:(j+1)*ps] = img
        tif_name = out_dir / f"cluster_{c:02d}_n{len(idx)}.tif"
        tifffile.imwrite(str(tif_name), panel)
        panel_tifs.append(panel)
        all_rows.append({"cluster": c, "n_patches": len(idx), "file": tif_name.name})

    if panel_tifs:
        max_w = max(p.shape[1] for p in panel_tifs)
        stacked = []
        for p in panel_tifs:
            if p.shape[1] < max_w:
                pad = np.zeros((p.shape[0], max_w - p.shape[1]), dtype=np.uint8)
                p = np.concatenate([p, pad], axis=1)
            stacked.append(p)
        tifffile.imwrite(str(out_dir / "all_clusters.tif"), np.stack(stacked))

    pd.DataFrame(all_rows).to_csv(out_dir / "cluster_labels.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=labels,
                    cmap="tab20", s=2, alpha=0.4, linewidths=0)
    plt.colorbar(sc, ax=ax, label="cluster")
    ax.set_title(f"KMeans k={k}")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(out_dir / f"umap_combo_kmeans_k{k}.png"), dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def run(model_dir: Path, k: int = 10, n_panel: int = 16) -> None:
    eval_dir    = model_dir / "eval"
    latents_csv = model_dir / "latents.csv"
    if not latents_csv.exists():
        print(f"  SKIP: no latents.csv in {model_dir}")
        return

    eval_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(latents_csv)
    print(f"  {len(df)} patches, columns: {list(df.columns[:8])}…")

    # Choose feature space: prefer z_proj (p_ cols), fallback to z_ cols
    proj_cols  = [c for c in df.columns if c.startswith("p_")]
    recon_cols = [c for c in df.columns if c.startswith("z_")]
    feat_cols  = proj_cols if proj_cols else recon_cols
    feat_label = "z_proj" if proj_cols else "z_recon"
    use_projector = bool(proj_cols)
    print(f"  Using {feat_label} ({len(feat_cols)} dims)")

    Z = df[feat_cols].values.astype(np.float32)

    # Balanced subsample for UMAP fitting
    cond_col = "condition_name" if "condition_name" in df.columns else None
    rng = np.random.default_rng(42)
    if cond_col and df[cond_col].nunique() > 1:
        sub_idx = []
        for _, grp in df.groupby(cond_col):
            idx = grp.index.to_numpy()
            if len(idx) > UMAP_MAX_PER_COND:
                idx = rng.choice(idx, UMAP_MAX_PER_COND, replace=False)
            sub_idx.append(idx)
        sub_idx = np.concatenate(sub_idx)
        print(f"  UMAP subsample: {len(sub_idx)} patches "
              f"(max {UMAP_MAX_PER_COND}/condition)")
    else:
        sub_idx = np.arange(len(df))

    Z_sub  = Z[sub_idx]
    df_sub = df.iloc[sub_idx].reset_index(drop=True)

    reducer_path = eval_dir / "umap_reducer.pkl"
    if reducer_path.exists():
        import pickle
        print("  Loading saved UMAP reducer…")
        with open(reducer_path, "rb") as fh:
            reducer = pickle.load(fh)
        emb_sub = reducer.transform(Z_sub)
    else:
        print("  Fitting UMAP…")
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        emb_sub = reducer.fit_transform(Z_sub)
        import pickle
        with open(reducer_path, "wb") as fh:
            pickle.dump(reducer, fh)
    np.save(str(eval_dir / "umap_combo_emb.npy"), emb_sub)

    # Build condition label list for training patches
    train_labels = (
        [_cond_label(str(c)) for c in df_sub[cond_col]]
        if cond_col else ["unknown"] * len(df_sub)
    )

    # ── scatter: training conditions (fixed tab20) ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    if cond_col:
        _scatter_conditions(ax, emb_sub, train_labels, s=3, alpha=0.4)
        ax.legend(markerscale=3, fontsize=7, loc="best",
                  framealpha=0.7, borderpad=0.4)
    else:
        ax.scatter(emb_sub[:, 0], emb_sub[:, 1], s=3, alpha=0.3)
    ax.set_title(f"UMAP {feat_label} — training")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(eval_dir / "umap_combo_condition.png"), dpi=150)
    plt.close(fig)
    print("  Saved umap_combo_condition.png")

    # ── encode test-set patches and overlay ───────────────────────────────────
    training_datasets = _get_training_datasets(model_dir)
    print(f"  Training datasets: {sorted(training_datasets)}")
    test_latents = _encode_test_latents(
        model_dir, training_datasets,
        feat_dim=len(feat_cols), use_projector=use_projector,
        rng=rng,
    )

    if test_latents:
        print("  Transforming test latents with fitted UMAP…")
        test_embs: dict[str, np.ndarray] = {}
        for lbl, Z_test in test_latents.items():
            test_embs[lbl] = reducer.transform(Z_test)

        fig, ax = plt.subplots(figsize=(7, 6))
        # training (small filled dots)
        if cond_col:
            _scatter_conditions(ax, emb_sub, train_labels, s=3, alpha=0.35)
        # test (hollow circles, larger)
        for lbl, emb_test in test_embs.items():
            _scatter_conditions(ax, emb_test, [lbl] * len(emb_test),
                                s=18, alpha=0.7, hollow=True)
        ax.legend(markerscale=2, fontsize=7, loc="best",
                  framealpha=0.7, borderpad=0.4)
        ax.set_title(f"UMAP {feat_label} — train (filled) + test (hollow)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        fig.savefig(str(eval_dir / "umap_combo_condition_with_test.png"), dpi=150)
        plt.close(fig)
        print("  Saved umap_combo_condition_with_test.png")
    else:
        print("  No test latents — skipping with_test UMAP")

    # ── scatter: train/val split ──────────────────────────────────────────────
    if "split" in df_sub.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        for spl, grp in df_sub.groupby("split"):
            ax.scatter(emb_sub[grp.index, 0], emb_sub[grp.index, 1],
                       c=SPLIT_COLOR.get(str(spl), "#888888"), s=3,
                       alpha=0.5, linewidths=0, label=str(spl))
        ax.legend(markerscale=3, fontsize=8)
        ax.set_title(f"UMAP {feat_label} — split")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        fig.savefig(str(eval_dir / "umap_combo_split.png"), dpi=150)
        plt.close(fig)
        print("  Saved umap_combo_split.png")

    # ── KMeans + cluster panels ───────────────────────────────────────────────
    print(f"  KMeans k={k}…")
    km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z_sub)

    patch_paths: list[Path] = []
    if "patch_path" in df_sub.columns:
        patch_paths = [Path(p) for p in df_sub["patch_path"]]
    elif "filepath" in df_sub.columns:
        patch_paths = [Path(p) for p in df_sub["filepath"]]
    elif "filename" in df_sub.columns and "patch_dir" in df_sub.columns:
        patch_paths = [Path(str(d)) / str(f)
                       for d, f in zip(df_sub["patch_dir"], df_sub["filename"])]

    panels_dir = eval_dir / "cluster_panels_combo"
    if patch_paths:
        _make_cluster_panels(km_labels, patch_paths, panels_dir, k, n_panel, emb_sub)
        print(f"  Cluster panels → {panels_dir}/")
    else:
        print("  WARNING: no patch paths in latents.csv — skipping cluster panels")
        panels_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb_sub[:, 0], emb_sub[:, 1], c=km_labels,
                   cmap="tab20", s=3, alpha=0.4, linewidths=0)
        ax.set_title(f"KMeans k={k}")
        fig.tight_layout()
        fig.savefig(str(panels_dir / f"umap_combo_kmeans_k{k}.png"), dpi=150)
        plt.close(fig)
        print(f"  KMeans scatter → {panels_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--k",       type=int, default=10, help="KMeans clusters")
    parser.add_argument("--n-panel", type=int, default=16, help="Patches per cluster panel")
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        sys.exit(f"Not a directory: {args.model_dir}")

    print(f"Model dir: {args.model_dir}")
    run(args.model_dir, k=args.k, n_panel=args.n_panel)
    print("Done.")


if __name__ == "__main__":
    main()
