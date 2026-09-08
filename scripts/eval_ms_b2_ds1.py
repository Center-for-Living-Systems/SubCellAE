#!/usr/bin/env python3
"""
eval_ms_b2_ds1.py

Evaluate multiscale B2 DS1 5-fold CV runs using LightGBM on latent features.

For each fold fv{k}:
  - Train latents come from latents.csv (already saved by the AE pipeline)
  - Test latents are obtained by running model_best.pt on the held-out fold patches
  - LightGBM classifies No adhesion vs adhesion
  - Reports balanced accuracy per fold + mean ± std

Accepts a run_tag (default: ms_b2_ds1_ps64_lat64p32) so any multiscale run
can be evaluated with the same script.

Usage:
    python scripts/eval_ms_b2_ds1.py [--run-tag ms_b2_ds1_ps64_lat64p32]
"""
from __future__ import annotations

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score

REPO      = Path(__file__).resolve().parents[1]
DATA      = Path("/net/projects/CLS/lding/data/fa_data_analysis")
FRAME_BASE = DATA / "ae_results" / "source_frames" / "cio_mode_prt" / "vinc"

sys.path.insert(0, str(REPO))
from subcellae.modelling.autoencoders import ContrastiveAE
from subcellae.modelling.dataset import CoordCropDataset

N_FOLDS = 5
LABEL_ORDER = ["No adhesion", "adhesion"]
LGBM_PARAMS = dict(
    n_estimators=300, num_leaves=31, learning_rate=0.05,
    min_child_samples=1, class_weight="balanced",
    n_jobs=4, random_state=42, verbose=-1,
)


def coord_key(row) -> str:
    return f"{row['condition_name']}_f{int(row['frame_idx']):04d}_cx{int(row['cx'])}_cy{int(row['cy'])}"


def encode_fold(run_dir: Path, test_df: pd.DataFrame, patch_size: int,
                latent_dim: int, proj_dim: int, device: str) -> pd.DataFrame:
    """Load model_best.pt and encode the test-fold patches. Returns DataFrame
    with columns: coord_key, z_0..z_{latent_dim-1}, label."""

    # Build temp coord CSV for the test fold
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    test_df.to_csv(tmp.name, index=False)
    tmp.close()

    results = []
    try:
        for cond_name, cond_id in [("control", 0), ("ycomp", 1)]:
            sub = test_df[test_df["condition_name"] == cond_name]
            if len(sub) == 0:
                continue
            ds = CoordCropDataset(
                frame_dir=str(FRAME_BASE / cond_name),
                coord_csv=tmp.name,
                patch_size=patch_size,
                channel="pax",
                condition=cond_id,
                condition_name=cond_name,
                annotation_label_col="label",
                label_order=LABEL_ORDER,
            )
            loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

            model = ContrastiveAE(
                latent_dim=latent_dim, proj_dim=proj_dim,
                input_ps=patch_size, no_ch=1,
                noise_prob=0.0, BN_flag=False, output_sigmoid=False,
            ).to(device)
            ckpt = torch.load(run_dir / "model_best.pt",
                              map_location=device, weights_only=False)
            state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
            model.load_state_dict(state)
            model.eval()

            with torch.no_grad():
                for batch in loader:
                    x     = batch[0].to(device)
                    ann   = batch[2].tolist()
                    keys  = batch[4]
                    _, z  = model(x)
                    z_np  = z.cpu().numpy()
                    for key, label_int, zrow in zip(keys, ann, z_np):
                        results.append({"key": key, "label_int": label_int, **{f"z_{i}": v for i, v in enumerate(zrow)}})
    finally:
        os.unlink(tmp.name)

    return pd.DataFrame(results).set_index("key")


def run_fold(k: int, run_dir: Path, fold_splits: pd.DataFrame,
             patch_size: int, latent_dim: int, proj_dim: int, device: str):
    train_rows = fold_splits[fold_splits["fold"] != k]
    test_rows  = fold_splits[fold_splits["fold"] == k]

    # --- Train latents from saved CSV ---
    lat_path = run_dir / "latents.csv"
    lat_raw  = pd.read_csv(lat_path)
    z_cols   = [c for c in lat_raw.columns if c.startswith("z_") and "_proj" not in c]
    lat_raw  = lat_raw.set_index("filename")

    train_keys = [coord_key(r) for _, r in train_rows.iterrows()]
    X_tr = lat_raw.reindex(train_keys)[z_cols].values
    y_tr = train_rows["label"].values

    # --- Test latents via model inference ---
    test_enc = encode_fold(run_dir, test_rows.reset_index(drop=True),
                           patch_size, latent_dim, proj_dim, device)
    test_keys = [coord_key(r) for _, r in test_rows.iterrows()]
    X_te = test_enc.reindex(test_keys)[[c for c in test_enc.columns if c.startswith("z_")]].values
    y_te = test_rows["label"].values

    # Drop rows with any NaN (missing latents)
    tr_ok = ~np.isnan(X_tr).any(axis=1)
    te_ok = ~np.isnan(X_te).any(axis=1)
    X_tr, y_tr = X_tr[tr_ok], y_tr[tr_ok]
    X_te, y_te = X_te[te_ok], y_te[te_ok]

    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = balanced_accuracy_score(y_te, y_pred)

    return acc, len(y_tr), len(y_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag",    default="ms_b2_ds1_ps64_lat64p32")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--proj-dim",   type=int, default=32)
    ap.add_argument("--device",     default="cpu")
    args = ap.parse_args()

    LAB_DIR  = DATA / "labelling" / args.run_tag
    RUN_ROOT = DATA / "ae_results" / "multiscale" / args.run_tag
    fs = pd.read_csv(LAB_DIR / "fold_splits.csv")

    print(f"\nEvaluating: {args.run_tag}  (ps={args.patch_size}  lat={args.latent_dim}  proj={args.proj_dim})")
    print(f"Total patches: {len(fs)}  folds: {N_FOLDS}")
    print()

    accs = []
    for k in range(N_FOLDS):
        run_dir = RUN_ROOT / f"{args.run_tag}_fv{k}"
        if not (run_dir / "latents.csv").exists():
            print(f"  fv{k}: MISSING latents.csv — skipping")
            continue
        acc, n_tr, n_te = run_fold(k, run_dir, fs,
                                   args.patch_size, args.latent_dim,
                                   args.proj_dim, args.device)
        accs.append(acc)
        print(f"  fv{k}  train={n_tr}  test={n_te}  bal_acc={acc:.4f}")

    if accs:
        print(f"\n  Mean ± std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Per-fold:   {[f'{a:.4f}' for a in accs]}")


if __name__ == "__main__":
    main()
