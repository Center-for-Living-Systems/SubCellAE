#!/usr/bin/env python3
"""
run_crossds_latents.py
======================
Project all 4-dataset patches into the embedding spaces already fitted (or
freshly fitted here) on the training dataset, and save a combined CSV + plots.

Two embedding spaces are computed for each model:
  z_recon  (z_* cols) — encoder latent; uses existing umap_model.pkl / phate_model.pkl
  z_proj   (p_* cols) — projector head;  fits umap_proj_model.pkl / phate_proj_model.pkl
                         fresh if not already saved (these never exist beforehand)

For the training dataset: reads z_*/p_* from latents.csv.
  - z_recon UMAP/PHATE: loaded from existing pkl (umap_model.pkl or umap_reducer.pkl);
    precomputed umap_emb.npy is used when it aligns with latents.csv, else transform().
  - z_proj  UMAP/PHATE: freshly fitted and saved to eval/.

For external datasets: runs model.encode() → z, model.project(z) → p, then
  umap_model.transform(z), umap_proj_model.transform(p), etc.

Outputs (written to <result_dir>/eval/):
  cross_dataset_latents.csv     — all patches: name, dataset, condition,
                                  z_*, p_*, UMAP_1, UMAP_2, PHATE_1, PHATE_2,
                                  UMAP_proj_1, UMAP_proj_2, PHATE_proj_1, PHATE_proj_2
  umap_4ds_{dataset,condition,annotation}.png       — z_recon UMAP
  phate_4ds_{dataset,condition,annotation}.png      — z_recon PHATE
  umap_proj_4ds_{dataset,condition,annotation}.png  — z_proj  UMAP
  phate_proj_4ds_{dataset,condition,annotation}.png — z_proj  PHATE

Saved pkl files (z_proj only; z_recon pkls are pre-existing):
  eval/umap_proj_model.pkl
  eval/umap_proj_emb.npy
  eval/phate_proj_model.pkl   (if phate available)
  eval/phate_proj_emb.npy

Usage:
  python scripts/run_crossds_latents.py <result_dir>
  python scripts/run_crossds_latents.py <result_dir> \\
      --root-folder /net/projects/CLS/lding/data/fa_data_analysis
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tifffile
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT_FOLDER     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
KNOWN_DATASETS  = {"vinc", "ppax", "pfak", "nih3t3"}
ALL_CONDITIONS  = ["control", "ycomp"]


# ── config helpers ────────────────────────────────────────────────────────────

def _read_cfg(result_dir: Path) -> dict:
    for yf in list(result_dir.glob("*.yaml")) + list(result_dir.glob("*.yml")):
        try:
            with open(yf) as fh:
                return yaml.safe_load(fh) or {}
        except Exception:
            pass
    return {}


def _input_divisor(result_dir: Path) -> float:
    cfg = _read_cfg(result_dir)
    ec = cfg.get("enlarged_crop", {})
    return float(ec.get("input_divisor", 1.0)) if ec.get("enabled") else 1.0


def _dataset_from_condition(cond_name: str) -> str:
    for ds in KNOWN_DATASETS:
        if str(cond_name).startswith(ds):
            return ds
    return "unknown"


def _training_datasets(result_dir: Path) -> set[str]:
    cfg = _read_cfg(result_dir)
    ds_set: set[str] = set()
    for entry in cfg.get("data", {}).get("patch_dirs", []):
        cond = str(entry.get("condition_name", ""))
        ds = _dataset_from_condition(cond)
        if ds in KNOWN_DATASETS:
            ds_set.add(ds)
    return ds_set or {"vinc"}


def _patch_template(result_dir: Path) -> str | None:
    """Infer patch dir template from config.
    Returns e.g. 'ae_results/patches/cio_rb/{}/{}/tiff_patches32_mr10'
    where the two {} placeholders are {dataset} and {condition}.
    """
    cfg = _read_cfg(result_dir)
    for entry in cfg.get("data", {}).get("patch_dirs", []):
        raw = str(entry.get("path", ""))
        raw = re.sub(r'root_folder\s*\+\s*"?', "", raw).strip(' "')
        m = re.search(r"ae_results/patches/(\w+)/\w+/\w+/(\w+)", raw)
        if m:
            return f"ae_results/patches/{m.group(1)}/{{}}/{{}}/{m.group(2)}"
    return None


# ── UMAP / PHATE helpers ──────────────────────────────────────────────────────

def _load_umap_model(eval_dir: Path):
    """Load z_recon UMAP: tries umap_model.pkl then umap_reducer.pkl."""
    for name in ("umap_model.pkl", "umap_reducer.pkl"):
        p = eval_dir / name
        if p.exists():
            m = joblib.load(str(p))
            print(f"[crossds]   loaded {name}")
            return m
    print("[crossds] WARNING: no z_recon UMAP pkl found")
    return None


def _fit_umap(vectors: np.ndarray, pkl_path: Path, npy_path: Path,
              label: str) -> tuple:
    """Fit a fresh UMAP on vectors, save pkl + npy, return (model, emb)."""
    from umap import UMAP
    print(f"[crossds] fitting UMAP on {len(vectors)} {label} vectors …", flush=True)
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(vectors)
    joblib.dump(reducer, str(pkl_path))
    np.save(str(npy_path), emb)
    print(f"[crossds]   saved {pkl_path.name}")
    return reducer, emb


def _fit_phate(vectors: np.ndarray, pkl_path: Path, npy_path: Path,
               label: str) -> tuple:
    """Fit a fresh PHATE on vectors, save pkl + npy, return (model, emb)."""
    try:
        import phate as phate_lib
    except ImportError:
        print("[crossds] WARNING: phate not installed — PHATE skipped")
        return None, None
    print(f"[crossds] fitting PHATE on {len(vectors)} {label} vectors …", flush=True)
    ph = phate_lib.PHATE(k=5, random_state=42, n_jobs=-1, verbose=0)
    emb = ph.fit_transform(vectors)
    joblib.dump(ph, str(pkl_path))
    np.save(str(npy_path), emb)
    print(f"[crossds]   saved {pkl_path.name}")
    return ph, emb


# ── patch encoding ────────────────────────────────────────────────────────────

class _TiffDir(Dataset):
    def __init__(self, patch_dir: Path):
        self.paths = sorted(patch_dir.glob("*.tif"))
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        arr = tifffile.imread(str(self.paths[i])).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        return torch.from_numpy(arr), self.paths[i].stem


@torch.no_grad()
def _encode(model, patch_dir: Path, device: str,
            batch_size: int, divisor: float
            ) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """Return (names, z (N,D), p (N,P) or None)."""
    ds = _TiffDir(patch_dir)
    if len(ds) == 0:
        return [], np.empty((0, 0)), None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    has_proj = hasattr(model, "projector")
    names, zs, ps = [], [], []
    for x_batch, name_batch in loader:
        x = x_batch.to(device)
        if divisor != 1.0:
            x = x / divisor
        z = model.encode(x)
        zs.append(z.cpu().numpy())
        if has_proj:
            ps.append(model.project(z).cpu().numpy())
        names.extend(name_batch)
    z_arr = np.concatenate(zs, axis=0)
    p_arr = np.concatenate(ps, axis=0) if ps else None
    return names, z_arr, p_arr


# ── scatter plots ─────────────────────────────────────────────────────────────

def _scatter(emb: np.ndarray, labels: list, label_order: list,
             title: str, out_path: Path, cmap: str = "tab10") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = plt.get_cmap(cmap)
    n = max(len(label_order) - 1, 1)
    for i, cat in enumerate(label_order):
        mask = np.array(labels) == cat
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   label=str(cat), s=3, alpha=0.4,
                   color=palette(i / n))
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _save_scatter_group(df_all: pd.DataFrame,
                        c1: str, c2: str,
                        prefix: str, emb_label: str,
                        eval_dir: Path,
                        model_name: str) -> None:
    """Save dataset / condition / annotation scatter plots for one embedding."""
    if c1 not in df_all.columns or df_all[c1].isna().all():
        return
    emb = df_all[[c1, c2]].values
    valid = ~np.isnan(emb[:, 0])

    ds_labels   = df_all["dataset"].tolist()
    cond_labels = df_all["condition_name"].tolist()
    ann_col     = "annotation_label_name"
    ann_labels  = (df_all[ann_col].fillna("unlabeled").tolist()
                   if ann_col in df_all.columns else ["unlabeled"] * len(df_all))

    ds_order   = sorted(KNOWN_DATASETS)
    cond_order = ALL_CONDITIONS
    ann_order  = sorted(set(ann_labels))

    for labels, order, suffix in [
        (ds_labels,   ds_order,   "dataset"),
        (cond_labels, cond_order, "condition"),
        (ann_labels,  ann_order,  "annotation"),
    ]:
        out = eval_dir / f"{prefix}_{suffix}.png"
        _scatter(emb[valid], [labels[i] for i, v in enumerate(valid) if v],
                 order,
                 f"{emb_label} (training fit) — {suffix}  ({model_name})",
                 out)
        print(f"[crossds]   {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(result_dir: Path, root_folder: Path,
        batch_size: int = 512, device: str = "cpu") -> None:

    eval_dir = result_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load training-dataset latents ─────────────────────────────────────
    lat_csv = result_dir / "latents.csv"
    if not lat_csv.exists():
        sys.exit(f"ERROR: {lat_csv} not found")

    train_df = pd.read_csv(lat_csv)
    z_cols = [c for c in train_df.columns if c.startswith("z_")]
    p_cols = [c for c in train_df.columns if c.startswith("p_")]
    print(f"[crossds] training latents: {len(train_df)} patches, "
          f"latent_dim={len(z_cols)}, proj_dim={len(p_cols)}")

    train_df["name"]    = train_df["filename"].apply(lambda f: Path(str(f)).stem)
    train_df["dataset"] = train_df["condition_name"].apply(_dataset_from_condition)

    z_train = train_df[z_cols].values.astype(np.float32)
    p_train = train_df[p_cols].values.astype(np.float32) if p_cols else None

    # ── 2. z_recon UMAP/PHATE — load existing models, attach embeddings ───────
    umap_model  = _load_umap_model(eval_dir)
    phate_model = None
    phate_pkl   = eval_dir / "phate_model.pkl"
    if phate_pkl.exists():
        phate_model = joblib.load(str(phate_pkl))
        print("[crossds]   loaded phate_model.pkl")
    else:
        print("[crossds] WARNING: phate_model.pkl not found — z_recon PHATE skipped")

    # Prefer precomputed .npy (aligned), fall back to transform()
    for model, emb_file, col1, col2, label in [
        (umap_model,  eval_dir / "umap_emb.npy",  "UMAP_1",  "UMAP_2",  "UMAP"),
        (phate_model, eval_dir / "phate_emb.npy", "PHATE_1", "PHATE_2", "PHATE"),
    ]:
        if model is None:
            continue
        if emb_file.exists():
            arr = np.load(str(emb_file))
            if len(arr) == len(train_df):
                train_df[col1] = arr[:, 0]
                train_df[col2] = arr[:, 1]
                print(f"[crossds]   loaded {emb_file.name} ({len(arr)} pts)")
                continue
            print(f"[crossds] INFO: {emb_file.name} length mismatch — re-transforming")
        print(f"[crossds] transforming {len(train_df)} training pts through z_recon {label} …",
              flush=True)
        try:
            coords = model.transform(z_train)
            train_df[col1] = coords[:, 0]
            train_df[col2] = coords[:, 1]
        except Exception as e:
            print(f"[crossds] WARNING: z_recon {label} transform failed: {e}")

    # ── 3. z_proj UMAP/PHATE — fit fresh (or reload if already saved) ─────────
    umap_proj_model  = None
    phate_proj_model = None

    if p_train is not None and len(p_cols) > 0:
        umap_proj_pkl  = eval_dir / "umap_proj_model.pkl"
        umap_proj_npy  = eval_dir / "umap_proj_emb.npy"
        phate_proj_pkl = eval_dir / "phate_proj_model.pkl"
        phate_proj_npy = eval_dir / "phate_proj_emb.npy"

        # z_proj UMAP
        if umap_proj_pkl.exists():
            umap_proj_model = joblib.load(str(umap_proj_pkl))
            print("[crossds]   loaded umap_proj_model.pkl")
            # Attach training embedding from npy or transform
            if umap_proj_npy.exists():
                arr = np.load(str(umap_proj_npy))
                if len(arr) == len(train_df):
                    train_df["UMAP_proj_1"] = arr[:, 0]
                    train_df["UMAP_proj_2"] = arr[:, 1]
                    print(f"[crossds]   loaded umap_proj_emb.npy ({len(arr)} pts)")
                else:
                    coords = umap_proj_model.transform(p_train)
                    train_df["UMAP_proj_1"] = coords[:, 0]
                    train_df["UMAP_proj_2"] = coords[:, 1]
            else:
                coords = umap_proj_model.transform(p_train)
                train_df["UMAP_proj_1"] = coords[:, 0]
                train_df["UMAP_proj_2"] = coords[:, 1]
        else:
            umap_proj_model, emb = _fit_umap(
                p_train, umap_proj_pkl, umap_proj_npy, "z_proj")
            train_df["UMAP_proj_1"] = emb[:, 0]
            train_df["UMAP_proj_2"] = emb[:, 1]

        # z_proj PHATE
        if phate_proj_pkl.exists():
            phate_proj_model = joblib.load(str(phate_proj_pkl))
            print("[crossds]   loaded phate_proj_model.pkl")
            if phate_proj_npy.exists():
                arr = np.load(str(phate_proj_npy))
                if len(arr) == len(train_df):
                    train_df["PHATE_proj_1"] = arr[:, 0]
                    train_df["PHATE_proj_2"] = arr[:, 1]
                    print(f"[crossds]   loaded phate_proj_emb.npy ({len(arr)} pts)")
                else:
                    coords = phate_proj_model.transform(p_train)
                    train_df["PHATE_proj_1"] = coords[:, 0]
                    train_df["PHATE_proj_2"] = coords[:, 1]
            else:
                coords = phate_proj_model.transform(p_train)
                train_df["PHATE_proj_1"] = coords[:, 0]
                train_df["PHATE_proj_2"] = coords[:, 1]
        else:
            phate_proj_model, emb = _fit_phate(
                p_train, phate_proj_pkl, phate_proj_npy, "z_proj")
            if emb is not None:
                train_df["PHATE_proj_1"] = emb[:, 0]
                train_df["PHATE_proj_2"] = emb[:, 1]
    else:
        print("[crossds] no p_* cols in latents.csv — z_proj embeddings skipped")

    # ── 4. Encode external datasets ───────────────────────────────────────────
    train_ds_set = _training_datasets(result_dir)
    patch_tpl    = _patch_template(result_dir)
    external_ds  = sorted(KNOWN_DATASETS - train_ds_set)

    ext_rows = []

    if external_ds and patch_tpl:
        ckpt = result_dir / "model_final.pt"
        if not ckpt.exists():
            ckpt = result_dir / "model_best.pt"
        if not ckpt.exists():
            print("[crossds] WARNING: no checkpoint — external datasets skipped")
        else:
            print(f"[crossds] loading model …", flush=True)
            model = torch.load(str(ckpt), map_location=device, weights_only=False)
            model.eval()
            div      = _input_divisor(result_dir)
            has_proj = hasattr(model, "projector")

            for ds_name in external_ds:
                for cond in ALL_CONDITIONS:
                    patch_dir = root_folder / patch_tpl.format(ds_name, cond)
                    if not patch_dir.exists():
                        print(f"  [skip] {ds_name}/{cond}: {patch_dir} not found")
                        continue
                    n_tif = len(list(patch_dir.glob("*.tif")))
                    print(f"  encoding {ds_name}/{cond}  ({n_tif} patches) …",
                          flush=True)
                    names, z_arr, p_arr = _encode(model, patch_dir, device,
                                                   batch_size, div)
                    if len(names) == 0:
                        continue

                    row: dict = {
                        "name":           names,
                        "dataset":        ds_name,
                        "condition_name": cond,
                        "split":          "test",
                    }
                    for ci, col in enumerate(z_cols):
                        row[col] = z_arr[:, ci]
                    if p_arr is not None and has_proj:
                        p_col_names = p_cols if p_cols else \
                            [f"p_{i}" for i in range(p_arr.shape[1])]
                        for ci, col in enumerate(p_col_names):
                            row[col] = p_arr[:, ci]

                    # z_recon UMAP / PHATE
                    if umap_model is not None:
                        try:
                            coords = umap_model.transform(z_arr)
                            row["UMAP_1"] = coords[:, 0]
                            row["UMAP_2"] = coords[:, 1]
                        except Exception as e:
                            print(f"  [warn] z_recon UMAP transform failed: {e}")
                    if phate_model is not None:
                        try:
                            coords = phate_model.transform(z_arr)
                            row["PHATE_1"] = coords[:, 0]
                            row["PHATE_2"] = coords[:, 1]
                        except Exception as e:
                            print(f"  [warn] z_recon PHATE transform failed: {e}")

                    # z_proj UMAP / PHATE
                    if p_arr is not None and umap_proj_model is not None:
                        try:
                            coords = umap_proj_model.transform(p_arr)
                            row["UMAP_proj_1"] = coords[:, 0]
                            row["UMAP_proj_2"] = coords[:, 1]
                        except Exception as e:
                            print(f"  [warn] z_proj UMAP transform failed: {e}")
                    if p_arr is not None and phate_proj_model is not None:
                        try:
                            coords = phate_proj_model.transform(p_arr)
                            row["PHATE_proj_1"] = coords[:, 0]
                            row["PHATE_proj_2"] = coords[:, 1]
                        except Exception as e:
                            print(f"  [warn] z_proj PHATE transform failed: {e}")

                    ext_rows.append(pd.DataFrame(row))

    elif not external_ds:
        print("[crossds] model trained on all 4 datasets — no external encoding needed")
    else:
        print("[crossds] patch template not found — external datasets skipped")

    # ── 5. Combine and save CSV ───────────────────────────────────────────────
    emb_cols = ["UMAP_1", "UMAP_2", "PHATE_1", "PHATE_2",
                "UMAP_proj_1", "UMAP_proj_2", "PHATE_proj_1", "PHATE_proj_2"]
    keep = (["name", "dataset", "condition_name", "split",
             "annotation_label_name"] + z_cols + p_cols + emb_cols)
    keep = [c for c in keep if c in train_df.columns]
    all_dfs = [train_df[keep].copy()] + ext_rows
    df_all = pd.concat(all_dfs, ignore_index=True)

    out_csv = eval_dir / "cross_dataset_latents.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"[crossds] saved {len(df_all)} patches → {out_csv.name}")
    print(f"  per dataset: {df_all.groupby('dataset').size().to_dict()}")

    # ── 6. Scatter plots ──────────────────────────────────────────────────────
    mn = result_dir.name
    for c1, c2, prefix, label in [
        ("UMAP_1",      "UMAP_2",      "umap_4ds",      "UMAP (z_recon)"),
        ("PHATE_1",     "PHATE_2",     "phate_4ds",     "PHATE (z_recon)"),
        ("UMAP_proj_1", "UMAP_proj_2", "umap_proj_4ds", "UMAP (z_proj)"),
        ("PHATE_proj_1","PHATE_proj_2","phate_proj_4ds","PHATE (z_proj)"),
    ]:
        _save_scatter_group(df_all, c1, c2, prefix, label, eval_dir, mn)

    # ── 7. K-means cluster UMAP (z_proj) ────────────────────────────────────
    cluster_csv = eval_dir / "cluster_panels" / "cluster_labels.csv"
    if cluster_csv.exists() and "UMAP_proj_1" in df_all.columns:
        try:
            cl_df = pd.read_csv(cluster_csv, usecols=["name", "cluster"])
            merged = df_all[["name", "UMAP_proj_1", "UMAP_proj_2"]].merge(
                cl_df, on="name", how="left")
            valid = merged["cluster"].notna()
            if valid.any():
                emb_km = merged.loc[valid, ["UMAP_proj_1", "UMAP_proj_2"]].values
                km_labels = merged.loc[valid, "cluster"].astype(int).tolist()
                k_ids = sorted(set(km_labels))
                out_km = eval_dir / "umap_proj_4ds_kmeans.png"
                fig, ax = plt.subplots(figsize=(7, 5))
                cmap = plt.get_cmap("tab20" if len(k_ids) > 10 else "tab10")
                n = max(len(k_ids) - 1, 1)
                for i, kid in enumerate(k_ids):
                    mask = np.array(km_labels) == kid
                    ax.scatter(emb_km[mask, 0], emb_km[mask, 1],
                               label=f"k{kid}", s=3, alpha=0.4,
                               color=cmap(i / n))
                ax.set_title(f"UMAP (z_proj) — K-means cluster ID  ({mn})", fontsize=10)
                ax.legend(markerscale=3, fontsize=7, loc="best", ncol=3)
                ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
                fig.tight_layout()
                fig.savefig(str(out_km), dpi=150)
                plt.close(fig)
                print(f"[crossds]   {out_km.name}")
        except Exception as e:
            print(f"[crossds] WARNING: k-means UMAP failed: {e}")

    print("[crossds] done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--root-folder", type=Path, default=ROOT_FOLDER)
    ap.add_argument("--batch-size",  type=int,  default=512)
    ap.add_argument("--device",      default="cpu")
    args = ap.parse_args()
    run(args.result_dir.resolve(), args.root_folder.resolve(),
        args.batch_size, args.device)


if __name__ == "__main__":
    main()
