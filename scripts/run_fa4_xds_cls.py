#!/usr/bin/env python3
"""
run_fa4_xds_cls.py
==================
Cross-dataset 4-class FA subtype classification experiments.

Modes
-----
  encode   : Encode vinc/ycomp and pfak/ctrl patches using the Stage-2 AE model.
             Saves latent CSVs to RUN_DIR/fa4_xds_eval/.  Requires GPU (or CPU fallback).
  classify : Run LightGBM label-efficiency experiments across five eval scenarios
             (within-dataset and cross-dataset).  CPU-only.
  plot     : Load results CSVs and generate efficiency-curve PNGs.  CPU-only.
  all      : Run encode → classify → plot in sequence.

Options
-------
  --option A  (default) uses annabel_vinc_supcon2_stage2_s3v1 model/latents.
  --option B  uses annabel_vinc_supcon2_stage2_combined model/latents.

Usage examples
--------------
  python scripts/run_fa4_xds_cls.py --mode encode --device auto
  python scripts/run_fa4_xds_cls.py --mode classify --option A
  python scripts/run_fa4_xds_cls.py --mode plot
  python scripts/run_fa4_xds_cls.py --mode all --option A --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Must add repo root before any subcellae imports (and before torch.load unpickling)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR   = DATA_ROOT / "ae_results" / "contrastive_run"
LABEL_DIR = DATA_ROOT / "labelling"

PATCH_DIRS = {
    "vinc_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio"    / "vinc" / "control" / "tiff_patches32_mr10",
    "vinc_ycomp": DATA_ROOT / "ae_results" / "patches" / "cio"    / "vinc" / "ycomp"   / "tiff_patches32_mr10",
    "pfak_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "pfak" / "control" / "tiff_patches32_mr10",
    "ppax_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio"    / "ppax" / "control" / "tiff_patches32_mr10",
}

LABEL_FILES = {
    "vinc_ctrl":  LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv",
    "vinc_ycomp": LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv",   # ycomp_ prefix rows
    "pfak_ctrl":  LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv",
    "ppax_ctrl":  LABEL_DIR / "ppax_combined_label_Ernest_latest.csv",      # 4-class FA only, no No-adhesion
}

# Stage 1 binary GBM (for ppax zero-shot Stage 1 gate)
STAGE1_AE_DIR  = RUN_DIR / "annabel_vinc_supcon2_s3v1"
STAGE1_GBM_PKL = STAGE1_AE_DIR / "fa_cls_zrecon" / "model.pkl"

# Stage-2 model / latents per option
MODEL_DIRS = {
    "A": RUN_DIR / "annabel_vinc_supcon2_stage2_s3v1",
    "B": RUN_DIR / "annabel_vinc_supcon2_stage2_combined",
    "C": RUN_DIR / "annabel_vinc_supcon2_stage2_2ch_s3v1",   # 2ch pax+actin
}

# Option C: 2-channel (pax + actin) patch dirs — (pax_dir, actin_dir) per dataset
PATCH_DIRS_2CH = {
    "vinc_ctrl":  (
        DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "vinc"     / "control" / "tiff_patches32_mr10",
        DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "vinc_ch3" / "control" / "tiff_patches32_mr10",
    ),
    "vinc_ycomp": (
        DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "vinc"     / "ycomp"   / "tiff_patches32_mr10",
        DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "vinc_ch3" / "ycomp"   / "tiff_patches32_mr10",
    ),
    "pfak_ctrl":  (
        DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "pfak"     / "control" / "tiff_patches32_mr10",
        DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "pfak_ch3" / "control" / "tiff_patches32_mr10",
    ),
}

# Option A: vinc/ctrl latents are already encoded in the model directory
PREENCODED_LATENTS = {
    "A": MODEL_DIRS["A"] / "latents.csv",
    "B": MODEL_DIRS["B"] / "latents.csv",
}

FA_LABEL_ORDER_4 = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]

# Short names used for per-class metric columns
LABEL_SHORT = {
    "Nascent Adhesion":    "NA",
    "focal complex":       "FC",
    "focal adhesion":      "FA",
    "fibrillar adhesion":  "Fib",
}

Z_COLS  = [f"z_{i}"  for i in range(12)]   # z_recon (12-d)
ZP_COLS = [f"zp_{i}" for i in range(8)]    # z_proj  (8-d)
SEED    = 42


def _feat_cols(variant: str) -> list[str]:
    if variant == "zproj":  return ZP_COLS
    if variant == "both":   return Z_COLS + ZP_COLS
    return Z_COLS  # "zrecon" (default)

# Training-fraction / repeat schedule
FRACS_REPEATS = [
    (0.10, 10),
    (0.25,  4),
    (0.50,  4),
    (0.75,  4),
]

# Scenario definitions: (name, train_datasets, test_datasets, cross_dataset)
# cross_dataset=True → test pool is always the FULL other-dataset pool
SCENARIOS = [
    ("vinc_only",   ["vinc_ctrl", "vinc_ycomp"], ["vinc_ctrl", "vinc_ycomp"], False),
    ("pfak_only",   ["pfak_ctrl"],               ["pfak_ctrl"],               False),
    ("ctrl->ycomp", ["vinc_ctrl"],               ["vinc_ycomp"],              True),
    ("vinc->pfak",  ["vinc_ctrl", "vinc_ycomp"], ["pfak_ctrl"],               True),
    ("pfak->vinc",  ["pfak_ctrl"],               ["vinc_ctrl", "vinc_ycomp"], True),
    ("combined",    ["vinc_ctrl", "vinc_ycomp", "pfak_ctrl"],
                    ["vinc_ctrl", "vinc_ycomp", "pfak_ctrl"],                 False),
]

# ---------------------------------------------------------------------------
# Eval output directory helper
# ---------------------------------------------------------------------------

def eval_dir(option: str) -> Path:
    name = "fa4_xds_eval_2ch" if option == "C" else "fa4_xds_eval"
    d = RUN_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def make_classifier():
    """Return LGBMClassifier if available, else GradientBoostingClassifier."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=3,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
        )
    except ImportError:
        print("[info] lightgbm not found; falling back to GradientBoostingClassifier")
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight as _csw

        class _GBDT:
            """Thin wrapper to expose fit/predict like LGBMClassifier."""
            def __init__(self):
                self._clf = GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=SEED,
                )

            def fit(self, X, y):
                w = _csw("balanced", y)
                self._clf.fit(X, y, sample_weight=w)
                return self

            def predict(self, X):
                return self._clf.predict(X)

            def predict_proba(self, X):
                return self._clf.predict_proba(X)

        return _GBDT()


# ===========================================================================
# MODE: encode
# ===========================================================================

def _load_model(option: str, device: str):
    """Load Stage-2 AE model_best.pt for the given option."""
    import torch
    model_path = MODEL_DIRS[option] / "model_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"[encode] Loading model: {model_path}")
    model = torch.load(str(model_path), map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def _encode_patches(model, patch_dir: Path, device: str,
                    batch_size: int = 256) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Encode all .tif patches in *patch_dir*.

    Returns
    -------
    filenames : list[str]    (basename only)
    latents   : np.ndarray   (N, 12)  z_recon
    proj      : np.ndarray   (N, 8)   z_proj
    """
    import torch
    import tifffile

    patch_paths = sorted(patch_dir.glob("*.tif"))
    if not patch_paths:
        raise FileNotFoundError(f"No .tif files in {patch_dir}")
    print(f"[encode]   {len(patch_paths)} patches in {patch_dir.name}")

    all_latents:   list[np.ndarray] = []
    all_proj:      list[np.ndarray] = []
    all_filenames: list[str] = []

    for start in range(0, len(patch_paths), batch_size):
        batch_paths = patch_paths[start : start + batch_size]
        imgs = []
        for p in batch_paths:
            arr = tifffile.imread(str(p)).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            elif arr.ndim == 3 and arr.shape[-1] <= 4:
                arr = np.moveaxis(arr, -1, 0)
            imgs.append(arr)
        x = torch.from_numpy(np.stack(imgs, axis=0)).to(device)

        with torch.no_grad():
            z = model.encode(x)
            p = model.project(z)

        all_latents.append(z.cpu().numpy())
        all_proj.append(p.cpu().numpy())
        all_filenames.extend([p.name for p in batch_paths])

        if (start // batch_size) % 10 == 0:
            print(f"[encode]     processed {min(start + batch_size, len(patch_paths))}"
                  f" / {len(patch_paths)}")

    return all_filenames, np.concatenate(all_latents, axis=0), np.concatenate(all_proj, axis=0)


def _encode_patches_2ch(model, pax_dir: Path, actin_dir: Path, device: str,
                         batch_size: int = 256) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Encode 2-channel (pax + actin) patches stacked as (2, H, W).

    Returns filenames, z_recon (N,12), z_proj (N,8) — same shape as single-ch.
    Only filenames present in BOTH directories are encoded (intersection).
    """
    import torch
    import tifffile

    pax_paths   = {p.name: p for p in sorted(pax_dir.glob("*.tif"))}
    actin_paths = {p.name: p for p in sorted(actin_dir.glob("*.tif"))}
    common = sorted(pax_paths.keys() & actin_paths.keys())
    if not common:
        raise FileNotFoundError(f"No matching patches between {pax_dir} and {actin_dir}")
    print(f"[encode]   {len(common)} matched patches (pax ∩ actin)")

    all_latents:   list[np.ndarray] = []
    all_proj:      list[np.ndarray] = []
    all_filenames: list[str] = []

    for start in range(0, len(common), batch_size):
        batch_names = common[start : start + batch_size]
        imgs = []
        for name in batch_names:
            pax   = tifffile.imread(str(pax_paths[name])).astype(np.float32)
            actin = tifffile.imread(str(actin_paths[name])).astype(np.float32)
            if pax.ndim == 3:
                pax = pax.squeeze()
            if actin.ndim == 3:
                actin = actin.squeeze()
            imgs.append(np.stack([pax, actin], axis=0))   # (2, H, W)
        x = torch.from_numpy(np.stack(imgs, axis=0)).to(device)

        with torch.no_grad():
            z = model.encode(x)
            p = model.project(z)

        all_latents.append(z.cpu().numpy())
        all_proj.append(p.cpu().numpy())
        all_filenames.extend(batch_names)

        if (start // batch_size) % 10 == 0:
            print(f"[encode]     processed {min(start + batch_size, len(common))}"
                  f" / {len(common)}")

    return all_filenames, np.concatenate(all_latents, axis=0), np.concatenate(all_proj, axis=0)


def run_encode(option: str, device: str):
    """Encode patches; save CSVs with both z_recon (Z_COLS) and z_proj (ZP_COLS)."""
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[encode] Using device: {device}")

    model = _load_model(option, device)
    out   = eval_dir(option)

    if option == "C":
        # 2-channel encoding: pax + actin stacked
        to_encode = {
            "vinc_ctrl":  "encoded_vinc_ctrl.csv",
            "vinc_ycomp": "encoded_vinc_ycomp.csv",
            "pfak_ctrl":  "encoded_pfak_ctrl.csv",
        }
        for ds_key, csv_name in to_encode.items():
            out_csv = out / csv_name
            pax_dir, actin_dir = PATCH_DIRS_2CH[ds_key]
            print(f"[encode] Encoding 2ch {ds_key} → {out_csv.name}")
            filenames, latents, proj = _encode_patches_2ch(model, pax_dir, actin_dir, device)
            df = pd.DataFrame(latents, columns=Z_COLS)
            df[ZP_COLS] = proj
            df.insert(0, "filename", filenames)
            df["dataset"] = ds_key
            df.to_csv(out_csv, index=False)
            print(f"[encode]   saved {len(df)} rows → {out_csv}")
        return

    # Options A / B: single-channel encoding
    to_encode = {
        "vinc_ctrl":  ("encoded_vinc_ctrl.csv",       PATCH_DIRS["vinc_ctrl"]),
        "vinc_ycomp": ("encoded_vinc_ycomp.csv",       PATCH_DIRS["vinc_ycomp"]),
        "pfak_ctrl":  ("encoded_pfak_ctrl.csv",        PATCH_DIRS["pfak_ctrl"]),
        "ppax_ctrl":  ("encoded_ppax_ctrl_stage2.csv", PATCH_DIRS["ppax_ctrl"]),
    }

    for ds_key, (csv_name, patch_dir) in to_encode.items():
        out_csv = out / csv_name
        print(f"[encode] Encoding {ds_key} → {out_csv.name}")
        filenames, latents, proj = _encode_patches(model, patch_dir, device)
        df = pd.DataFrame(latents, columns=Z_COLS)
        df[ZP_COLS] = proj
        df.insert(0, "filename", filenames)
        df["dataset"] = ds_key
        df.to_csv(out_csv, index=False)
        print(f"[encode]   saved {len(df)} rows → {out_csv}")


# ===========================================================================
# MODE: classify
# ===========================================================================

def _load_labels(ds_key: str) -> pd.DataFrame:
    """Return DataFrame(filename, label) for the given dataset key."""
    path = LABEL_FILES[ds_key]
    df = pd.read_csv(path)

    if ds_key == "vinc_ycomp":
        # Combined file: keep only ycomp_ prefixed rows; filenames match disk as-is
        df = df[df["filename"].str.startswith("ycomp_")].copy()
    elif ds_key == "vinc_ctrl":
        # Only keep non-ycomp rows (safety; should be all rows)
        df = df[~df["filename"].str.startswith("ycomp_")].copy()

    return df[["filename", "label"]].copy()


def _load_encoded_latents(option: str, ds_key: str) -> pd.DataFrame:
    """Return DataFrame with Z_COLS + ZP_COLS for the given dataset key.

    Prefers encoded_*.csv (which has both zrecon + zproj columns).
    Falls back to stage2 latents.csv for vinc/ctrl Option A (zrecon only).
    """
    out = eval_dir(option)

    csv_map = {
        "vinc_ctrl":  out / "encoded_vinc_ctrl.csv",
        "vinc_ycomp": out / "encoded_vinc_ycomp.csv",
        "pfak_ctrl":  out / "encoded_pfak_ctrl.csv",
    }
    src = csv_map.get(ds_key)
    if src is None:
        raise ValueError(f"Unknown dataset key: {ds_key}")

    if src.exists():
        return pd.read_csv(src)

    # Fallback: Option A vinc/ctrl pre-encoded latents (z_recon only, no zp_*)
    if ds_key == "vinc_ctrl" and option == "A":
        fallback = PREENCODED_LATENTS["A"]
        if fallback.exists():
            print(f"[warn] Using fallback latents.csv for vinc_ctrl "
                  f"(no zp_* columns — re-run encode for zproj/both variants)")
            df = pd.read_csv(fallback, usecols=["filename"] + Z_COLS)
            df["dataset"] = "vinc_ctrl"
            return df

    raise FileNotFoundError(
        f"{src.name} not found; run --mode encode --option {option} first."
    )


def _build_dataset(option: str, ds_keys: list[str]) -> pd.DataFrame:
    """Merge latents with labels; keep only FA-4 labelled patches."""
    frames = []
    for ds_key in ds_keys:
        lat  = _load_encoded_latents(option, ds_key)
        labs = _load_labels(ds_key)
        merged = lat.merge(labs, on="filename", how="inner")
        merged["dataset"] = ds_key
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(FA_LABEL_ORDER_4)].copy()
    df["label"] = pd.Categorical(df["label"], categories=FA_LABEL_ORDER_4, ordered=True)
    return df


def _metrics_one_split(y_true, y_pred, classes) -> dict:
    """Compute balanced accuracy, per-class F1, macro F1."""
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_recall_fscore_support,
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    row: dict = {"bal_acc": bal_acc, "macro_f1": macro_f1}
    for cls, f in zip(classes, f1):
        short = LABEL_SHORT.get(cls, cls)
        row[f"f1_{short}"] = float(f)
    return row


def _apply_smote(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to match the majority class count.

    Uses imblearn SMOTE if available; otherwise falls back to random oversampling
    with small Gaussian jitter (noise scale = 0.05 * per-feature std).
    """
    try:
        from imblearn.over_sampling import SMOTE
        counts = dict(zip(*np.unique(y_tr, return_counts=True)))
        k = max(1, min(5, min(counts.values()) - 1))
        return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X_tr, y_tr)
    except ImportError:
        pass  # fall through to numpy fallback

    rng = np.random.default_rng(SEED)
    classes, counts = np.unique(y_tr, return_counts=True)
    n_target = counts.max()
    feat_std  = X_tr.std(axis=0).clip(min=1e-6)

    X_parts = [X_tr]
    y_parts = [y_tr]
    for cls, cnt in zip(classes, counts):
        if cnt >= n_target:
            continue
        idx = np.where(y_tr == cls)[0]
        n_need = n_target - cnt
        chosen = rng.choice(idx, size=n_need, replace=True)
        noise  = rng.normal(0, 0.05, size=(n_need, X_tr.shape[1])) * feat_std
        X_parts.append(X_tr[chosen] + noise)
        y_parts.append(np.full(n_need, cls))

    return np.vstack(X_parts), np.concatenate(y_parts)


def run_classify(option: str, variant: str = "zrecon", smote: bool = False):
    """Run all label-efficiency scenarios; save per-run and summary CSVs.

    Parameters
    ----------
    variant : "zrecon" | "zproj" | "both"
        Which latent features to use for the GBM classifier.
    smote   : bool
        If True, apply SMOTE oversampling to the training set before each GBM fit.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    fcols = _feat_cols(variant)
    suffix = f"{option}_{variant}" + ("_smote" if smote else "")
    out = eval_dir(option)
    print(f"[classify] Option {option}  variant={variant}  smote={smote}")
    print(f"[classify] Features: {fcols}  output suffix: {suffix}")

    # Pre-load all unique datasets needed
    all_ds_keys = set()
    for _, train_ds, test_ds, _ in SCENARIOS:
        all_ds_keys.update(train_ds + test_ds)

    ds_data: dict[str, pd.DataFrame] = {}
    for key in sorted(all_ds_keys):
        ds_data[key] = _build_dataset(option, [key])
        print(f"[classify]   {key}: {len(ds_data[key])} FA-labelled patches, "
              f"classes: {dict(ds_data[key]['label'].value_counts())}")

    classes = FA_LABEL_ORDER_4
    all_summaries = []

    for scenario_name, train_keys, test_keys, cross_ds in SCENARIOS:
        print(f"\n[classify] Scenario: {scenario_name}  "
              f"(train={train_keys}, test={test_keys})")

        train_pool = pd.concat([ds_data[k] for k in train_keys], ignore_index=True)
        test_pool  = pd.concat([ds_data[k] for k in test_keys],  ignore_index=True)

        # Validate feature columns exist
        missing = [c for c in fcols if c not in train_pool.columns]
        if missing:
            print(f"  [skip] columns {missing} not in dataset — "
                  f"re-run --mode encode first for variant='{variant}'")
            continue

        records = []

        for frac, n_repeats in FRACS_REPEATS:
            if cross_ds:
                rng = np.random.default_rng(SEED)
                for rep in range(n_repeats):
                    n_sample = max(1, int(len(train_pool) * frac))
                    train_idx = rng.choice(len(train_pool), size=n_sample, replace=False)
                    train_df  = train_pool.iloc[train_idx]
                    test_df   = test_pool

                    vc = train_df["label"].value_counts()
                    for cls in FA_LABEL_ORDER_4:
                        if vc.get(cls, 0) < 3:
                            print(f"  [warn] {scenario_name} frac={frac} rep={rep}: "
                                  f"'{cls}' has only {vc.get(cls,0)} train samples")

                    X_tr = train_df[fcols].values
                    y_tr = train_df["label"].values
                    X_te = test_df[fcols].values
                    y_te = test_df["label"].values

                    if smote:
                        X_tr, y_tr = _apply_smote(X_tr, y_tr)

                    clf = make_classifier()
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)

                    m = _metrics_one_split(y_te, y_pred, classes)
                    m.update({"frac": frac, "repeat": rep,
                              "n_train": len(train_df), "n_test": len(test_df)})
                    records.append(m)
            else:
                X = train_pool[fcols].values
                y = train_pool["label"].values

                sss = StratifiedShuffleSplit(
                    n_splits=n_repeats,
                    test_size=1.0 - frac,
                    random_state=SEED,
                )
                for rep, (tr_idx, te_idx) in enumerate(sss.split(X, y)):
                    X_tr, y_tr = X[tr_idx], y[tr_idx]
                    X_te, y_te = X[te_idx], y[te_idx]

                    unique, counts = np.unique(y_tr, return_counts=True)
                    for cls in FA_LABEL_ORDER_4:
                        if dict(zip(unique, counts)).get(cls, 0) < 3:
                            print(f"  [warn] {scenario_name} frac={frac} rep={rep}: "
                                  f"'{cls}' has only {dict(zip(unique,counts)).get(cls,0)} train samples")

                    if smote:
                        X_tr, y_tr = _apply_smote(X_tr, y_tr)

                    clf = make_classifier()
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)

                    m = _metrics_one_split(y_te, y_pred, classes)
                    m.update({"frac": frac, "repeat": rep,
                              "n_train": len(X_tr), "n_test": len(X_te)})
                    records.append(m)

        df_res = pd.DataFrame(records)
        csv_out = out / f"results_{scenario_name}_{suffix}.csv"
        df_res.to_csv(csv_out, index=False)
        print(f"[classify]   saved {len(df_res)} rows → {csv_out.name}")

        metric_cols = ["bal_acc", "macro_f1"] + [f"f1_{LABEL_SHORT[c]}" for c in FA_LABEL_ORDER_4]
        agg = df_res.groupby("frac")[metric_cols].agg(["mean", "std"]).reset_index()
        agg.columns = ["frac"] + [f"{m}_{s}" for m in metric_cols for s in ("mean", "std")]
        agg["scenario"] = scenario_name
        all_summaries.append(agg)

    summary = pd.concat(all_summaries, ignore_index=True)
    sum_out = out / f"summary_{suffix}.csv"
    summary.to_csv(sum_out, index=False)
    print(f"\n[classify] Summary saved → {sum_out}")


# ===========================================================================
# MODE: plot
# ===========================================================================

SCENARIO_COLORS = {
    "vinc_only":   "#4E79A7",
    "pfak_only":   "#F28E2B",
    "ctrl->ycomp": "#B07AA1",
    "vinc->pfak":  "#E15759",
    "pfak->vinc":  "#76B7B2",
    "combined":    "#59A14F",
}

SCENARIO_LABELS = {
    "vinc_only":   "vinc only (within)",
    "pfak_only":   "pfak only (within)",
    "ctrl->ycomp": "ctrl → ycomp (cross-cond)",
    "vinc->pfak":  "vinc → pfak (cross-ds)",
    "pfak->vinc":  "pfak → vinc (cross-ds)",
    "combined":    "combined (within)",
}


def run_plot(option: str, variant: str = "zrecon", smote: bool = False):
    """Load results CSVs and generate efficiency-curve PNGs per scenario."""
    suffix = f"{option}_{variant}" + ("_smote" if smote else "")
    out = eval_dir(option)

    metric_display = [
        ("bal_acc",   "Balanced accuracy"),
        ("macro_f1",  "Macro F1"),
    ]

    for metric_key, metric_label in metric_display:
        fig, axes = plt.subplots(
            1, len(SCENARIOS), figsize=(4.2 * len(SCENARIOS), 4.8),
            sharey=False, facecolor="white"
        )
        if len(SCENARIOS) == 1:
            axes = [axes]

        for ax, (scenario_name, _, _, _) in zip(axes, SCENARIOS):
            csv_path = out / f"results_{scenario_name}_{suffix}.csv"
            if not csv_path.exists():
                ax.set_title(f"{scenario_name}\n(no data)", fontsize=9)
                ax.axis("off")
                continue

            df = pd.read_csv(csv_path)
            fracs  = sorted(df["frac"].unique())
            means  = [df[df["frac"] == f][metric_key].mean() * 100 for f in fracs]
            stds   = [df[df["frac"] == f][metric_key].std()  * 100 for f in fracs]

            color = SCENARIO_COLORS.get(scenario_name, "#888888")
            x = np.arange(len(fracs))
            ax.errorbar(
                x, means, yerr=stds,
                fmt="o-", color=color,
                capsize=4, linewidth=1.8, markersize=7,
            )
            # Annotate values
            for xi, m, s in zip(x, means, stds):
                ax.text(xi, m + s + 1.0, f"{m:.1f}", ha="center", fontsize=7.5, color=color)

            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(f*100)}%" for f in fracs], fontsize=9)
            ax.set_xlabel("Training fraction", fontsize=9)
            ax.set_ylabel(f"{metric_label} (%)", fontsize=9)
            ax.set_title(SCENARIO_LABELS.get(scenario_name, scenario_name),
                         fontsize=9.5, fontweight="bold")
            ax.set_ylim(0, 110)
            ax.axhline(50, color="#CCCCCC", linestyle="--", linewidth=0.8)
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            f"FA 4-class — Cross-Dataset Label Efficiency  (Option {option}  {variant}"
            + ("  SMOTE" if smote else "") + f")\n"
            f"{metric_label}  ·  Stage-2 SupCon AE latents + LightGBM",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        png_out = out / f"efficiency_{metric_key}_{suffix}.png"
        fig.savefig(str(png_out), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[plot] saved → {png_out}")

    # Per-class F1 summary plot
    fig, axes = plt.subplots(
        len(FA_LABEL_ORDER_4), len(SCENARIOS),
        figsize=(4.0 * len(SCENARIOS), 3.5 * len(FA_LABEL_ORDER_4)),
        sharey=False, facecolor="white",
    )
    if len(FA_LABEL_ORDER_4) == 1:
        axes = [axes]
    if len(SCENARIOS) == 1:
        axes = [[ax] for ax in axes]

    for col, (scenario_name, _, _, _) in enumerate(SCENARIOS):
        csv_path = out / f"results_{scenario_name}_{suffix}.csv"
        for row, cls in enumerate(FA_LABEL_ORDER_4):
            ax = axes[row][col]
            short = LABEL_SHORT[cls]
            col_key = f"f1_{short}"
            if not csv_path.exists():
                ax.axis("off")
                continue

            df = pd.read_csv(csv_path)
            if col_key not in df.columns:
                ax.axis("off")
                continue

            fracs = sorted(df["frac"].unique())
            means = [df[df["frac"] == f][col_key].mean() * 100 for f in fracs]
            stds  = [df[df["frac"] == f][col_key].std()  * 100 for f in fracs]
            color = SCENARIO_COLORS.get(scenario_name, "#888888")
            x = np.arange(len(fracs))
            ax.errorbar(x, means, yerr=stds, fmt="o-", color=color,
                        capsize=3, linewidth=1.5, markersize=5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(f*100)}%" for f in fracs], fontsize=7)
            ax.set_ylim(0, 110)
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(SCENARIO_LABELS.get(scenario_name, scenario_name),
                             fontsize=8, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"F1 ({cls})", fontsize=8)

    fig.suptitle(
        f"FA 4-class per-class F1  (Option {option}  {variant}"
        + ("  SMOTE" if smote else "") + ")\nStage-2 SupCon AE + LightGBM",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    png_out = out / f"efficiency_perclass_{suffix}.png"
    fig.savefig(str(png_out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] saved → {png_out}")


# ===========================================================================
# MODE: ppax_zeroshot
# ===========================================================================

def run_ppax_zeroshot(option: str, device: str, variant: str = "zrecon", smote: bool = False):
    """Zero-shot ppax evaluation: Stage 1 binary gate → Stage 2 4-class.

    Ernest labeled only FA subtypes (no 'No adhesion'), so:
      - Stage 1 false negatives: Ernest-labeled patches predicted as No-adhesion
      - Stage 2 accuracy: computed on patches that pass Stage 1 AND have Ernest labels

    Steps
    -----
    1. Encode ppax/ctrl patches with Stage-1 AE (annabel_vinc_supcon2_s3v1)
    2. Apply Stage-1 binary GBM → get predicted-adhesion filenames
    3. Load Stage-2 latents for ppax (from encoded_ppax_ctrl_stage2.csv)
    4. Train a fresh 4-class GBM on ALL training data (vinc + pfak, 100%)
    5. Apply to ppax Stage-2 latents for predicted-adhesion patches
    6. Compute Stage-1 recall + Stage-2 accuracy vs Ernest labels
    """
    import torch
    import joblib

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out = eval_dir(option)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: encode ppax with Stage-1 AE ──────────────────────────────────
    stage1_enc_csv = out / "encoded_ppax_ctrl_stage1.csv"
    if not stage1_enc_csv.exists():
        print(f"[ppax_zs] Encoding ppax/ctrl with Stage-1 AE → {stage1_enc_csv.name}")
        if not STAGE1_AE_DIR.exists():
            raise FileNotFoundError(f"Stage-1 AE dir not found: {STAGE1_AE_DIR}")
        s1_model = torch.load(str(STAGE1_AE_DIR / "model_best.pt"),
                              map_location=device, weights_only=False)
        s1_model = s1_model.to(device)
        s1_model.eval()
        fns, zs = _encode_patches(s1_model, PATCH_DIRS["ppax_ctrl"], device)
        df_s1 = pd.DataFrame(zs, columns=[f"z_{i}" for i in range(zs.shape[1])])
        df_s1.insert(0, "filename", fns)
        df_s1.to_csv(stage1_enc_csv, index=False)
        print(f"[ppax_zs]   saved {len(df_s1)} rows → {stage1_enc_csv.name}")
        del s1_model
    else:
        print(f"[ppax_zs] Stage-1 encodings exist: {stage1_enc_csv.name}")
        df_s1 = pd.read_csv(stage1_enc_csv)

    # ── Step 2: apply Stage-1 binary GBM ─────────────────────────────────────
    if not STAGE1_GBM_PKL.exists():
        raise FileNotFoundError(f"Stage-1 GBM not found: {STAGE1_GBM_PKL}")
    s1_gbm = joblib.load(str(STAGE1_GBM_PKL))
    s1_z_cols = [c for c in df_s1.columns if c.startswith("z_")]
    s1_preds = s1_gbm.predict(df_s1[s1_z_cols].values)
    df_s1["stage1_pred"] = s1_preds

    total_ppax = len(df_s1)
    n_adh_pred = (df_s1["stage1_pred"] == 1).sum()
    adh_fns = set(df_s1.loc[df_s1["stage1_pred"] == 1, "filename"])
    print(f"[ppax_zs] Stage-1 gate: {n_adh_pred}/{total_ppax} patches predicted adhesion")

    # ── Step 3: Ernest labels ─────────────────────────────────────────────────
    ernest = pd.read_csv(LABEL_FILES["ppax_ctrl"])
    ernest = ernest[ernest["label"].isin(FA_LABEL_ORDER_4)].copy()
    print(f"[ppax_zs] Ernest labels: {len(ernest)} patches  "
          f"({dict(ernest['label'].value_counts())})")

    ernest_fns = set(ernest["filename"])
    # Stage 1 recall on Ernest-labeled patches
    labeled_in_adh  = ernest_fns & adh_fns
    n_fn = len(ernest_fns) - len(labeled_in_adh)
    s1_recall = len(labeled_in_adh) / len(ernest_fns) if ernest_fns else float("nan")
    print(f"[ppax_zs] Stage-1 recall on Ernest patches: {s1_recall:.3f} "
          f"({len(labeled_in_adh)}/{len(ernest_fns)} passed, {n_fn} false negatives)")

    # ── Step 4: load Stage-2 latents for ppax ────────────────────────────────
    stage2_enc_csv = out / "encoded_ppax_ctrl_stage2.csv"
    if not stage2_enc_csv.exists():
        raise FileNotFoundError(
            f"encoded_ppax_ctrl_stage2.csv not found; run --mode encode first."
        )
    df_s2 = pd.read_csv(stage2_enc_csv)

    # Filter to Stage-1 predicted-adhesion patches
    df_s2_adh = df_s2[df_s2["filename"].isin(adh_fns)].copy()
    print(f"[ppax_zs] Stage-2 latents for predicted-adhesion ppax: {len(df_s2_adh)}")

    # Merge with Ernest labels (intersection: passed Stage 1 AND have labels)
    eval_df = df_s2_adh.merge(ernest[["filename", "label"]], on="filename", how="inner")
    print(f"[ppax_zs] Evaluation set (passed S1 + Ernest labels): {len(eval_df)} patches")

    if len(eval_df) == 0:
        print("[ppax_zs] No patches to evaluate. Stopping.")
        return

    # ── Step 5: train 4-class GBM on all available training data ─────────────
    all_train_keys = ["vinc_ctrl", "vinc_ycomp", "pfak_ctrl"]
    train_parts = []
    for key in all_train_keys:
        try:
            lat = _load_encoded_latents(option, key)
            labs = _load_labels(key)
            merged = lat.merge(labs, on="filename", how="inner")
            merged = merged[merged["label"].isin(FA_LABEL_ORDER_4)].copy()
            merged["dataset"] = key
            train_parts.append(merged)
            print(f"[ppax_zs]   train {key}: {len(merged)} patches")
        except FileNotFoundError as e:
            print(f"[ppax_zs]   [skip] {key}: {e}")

    if not train_parts:
        print("[ppax_zs] No training data found. Stopping.")
        return

    fcols = _feat_cols(variant)
    train_df = pd.concat(train_parts, ignore_index=True)
    missing = [c for c in fcols if c not in train_df.columns]
    if missing:
        print(f"[ppax_zs] [warn] columns {missing} missing — falling back to zrecon")
        fcols = Z_COLS

    X_tr = train_df[fcols].values
    y_tr = train_df["label"].values
    print(f"[ppax_zs] Training 4-class GBM on {len(train_df)} patches "
          f"from {all_train_keys}  (variant={variant}, smote={smote})")

    if smote:
        X_tr, y_tr = _apply_smote(X_tr, y_tr)

    clf = make_classifier()
    clf.fit(X_tr, y_tr)

    # ── Step 6: evaluate on ppax eval set ────────────────────────────────────
    ppax_missing = [c for c in fcols if c not in eval_df.columns]
    if ppax_missing:
        print(f"[ppax_zs] [warn] ppax eval missing {ppax_missing} — falling back to zrecon")
        fcols = Z_COLS
    X_ev = eval_df[fcols].values
    y_ev = eval_df["label"].values
    y_pred = clf.predict(X_ev)

    from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                  confusion_matrix, ConfusionMatrixDisplay)

    bal_acc = balanced_accuracy_score(y_ev, y_pred)
    report  = classification_report(y_ev, y_pred, labels=FA_LABEL_ORDER_4, zero_division=0)
    print(f"\n[ppax_zs] 4-class balanced accuracy on ppax eval set: {bal_acc:.3f}")
    print(report)

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_ev, y_pred, labels=FA_LABEL_ORDER_4),
        display_labels=["NA", "FC", "FA", "Fib"],
    ).plot(ax=ax, colorbar=False)
    suffix_zs = f"{option}_{variant}" + ("_smote" if smote else "")
    ax.set_title(f"ppax zero-shot 4-class  (Option {option}  {variant}"
                 + ("  SMOTE" if smote else "") +
                 f")\nStage-1 recall={s1_recall:.2f}  Stage-2 bal_acc={bal_acc:.3f}")
    fig.tight_layout()
    fig.savefig(str(out / f"ppax_zeroshot_confusion_{suffix_zs}.png"), dpi=150)
    plt.close(fig)

    # Save summary CSV
    summary = {
        "option":      option,
        "variant":     variant,
        "smote":       smote,
        "n_ernest":    len(ernest_fns),
        "n_passed_s1": len(labeled_in_adh),
        "n_false_neg": n_fn,
        "s1_recall":   s1_recall,
        "n_eval":      len(eval_df),
        "bal_acc_s2":  bal_acc,
    }
    pd.DataFrame([summary]).to_csv(out / f"ppax_zeroshot_summary_{suffix_zs}.csv", index=False)
    print(f"[ppax_zs] Results saved → {out}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-dataset 4-class FA subtype classification experiments."
    )
    p.add_argument(
        "--mode",
        choices=["encode", "classify", "plot", "ppax_zeroshot", "all"],
        default="all",
        help="Which step(s) to run (default: all; ppax_zeroshot runs after encode+classify).",
    )
    p.add_argument(
        "--option",
        choices=["A", "B", "C"],
        default="A",
        help=(
            "A = stage2_s3v1 model (default); "
            "B = stage2_combined model; "
            "C = stage2_2ch_s3v1 model (pax+actin, 2-channel)."
        ),
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Torch device for encoding (default: auto).",
    )
    p.add_argument(
        "--variant",
        choices=["zrecon", "zproj", "both"],
        default="zrecon",
        help="Feature columns for GBM classifier (default: zrecon). "
             "zproj and both require encoded CSVs with zp_* columns (re-run encode).",
    )
    p.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE oversampling to training set before each GBM fit.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode in ("encode", "all"):
        run_encode(args.option, args.device)

    if args.mode in ("classify", "all"):
        run_classify(args.option, args.variant, args.smote)

    if args.mode in ("plot", "all"):
        run_plot(args.option, args.variant, args.smote)

    if args.mode in ("ppax_zeroshot", "all"):
        run_ppax_zeroshot(args.option, args.device, args.variant, args.smote)

    print("\n[done]")


if __name__ == "__main__":
    main()
