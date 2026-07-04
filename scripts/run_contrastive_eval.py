#!/usr/bin/env python3
"""
run_contrastive_eval.py
=======================
Comprehensive post-training evaluation for a single contrastive AE result directory.

Steps
-----
1. UMAP + PHATE embeddings from latents.csv, scatter plots coloured by
   condition / FA-type annotation / split / KMeans cluster
2. KMeans clustering (k=5 by default)
3. KNN classification rate on vinc: trained on labelled train patches,
   evaluated on labelled val patches → accuracy + confusion matrices
4. Model inference on ppax patches → extract latents
5. KNN applied to ppax latents → accuracy + confusion matrices vs ppax labels

Output: all artefacts written to  <result_dir>/eval/

Usage
-----
    python scripts/run_contrastive_eval.py <result_dir>
    python scripts/run_contrastive_eval.py <result_dir> \\
        --kmeans_k 5 --knn_k 5 --batch_size 512 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subcellae.modelling.dataset import PatchDataset, MultiChannelPatchDataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = "/net/projects/CLS/lding/data/fa_data_analysis"

PPAX_PATCH_DIRS = {
    "control": f"{ROOT}/ae_results/patches/cio_rb/ppax/control/tiff_patches32_mr10",
    "ycomp":   f"{ROOT}/ae_results/patches/cio_rb/ppax/ycomp/tiff_patches32_mr10",
}
PPAX_PATCH_DIRS_CH3 = {
    "control": f"{ROOT}/ae_results/patches/cio_rb/ppax_ch3/control/tiff_patches32_mr10",
    "ycomp":   f"{ROOT}/ae_results/patches/cio_rb/ppax_ch3/ycomp/tiff_patches32_mr10",
}
PPAX_LABELS_CSV  = f"{ROOT}/labelling/labels_ppax_20260521.csv"
VINC_LABELS_CSV  = f"{ROOT}/labelling/labels_vinc_20260521.csv"

FA_LABEL_ORDER = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
    "No adhesion",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_input_divisor(result_dir: Path) -> float:
    """Read input_divisor from the config YAML copied into result_dir."""
    yamls = list(result_dir.glob("*.yaml"))
    if not yamls:
        return 1.0
    with open(yamls[0]) as f:
        raw = yaml.safe_load(f)
    return float((raw.get("enlarged_crop") or {}).get("input_divisor", 1.0))


def _scatter(emb: np.ndarray, labels, label_order: list, title: str,
             out_path: Path, cmap: str = "tab10"):
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = plt.get_cmap(cmap)
    n = max(len(label_order) - 1, 1)
    for i, cat in enumerate(label_order):
        mask = np.array(labels) == cat
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   label=str(cat), s=3, alpha=0.5,
                   color=palette(i / n))
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=3, fontsize=7, loc="best")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _confusion_png(y_true, y_pred, labels: list, title: str, out_path: Path,
                   normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues",
              values_format=".2f" if normalize else "d")
    ax.set_title(title, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _encode_patches(model, patch_dir: str, input_divisor: float,
                    device: str, batch_size: int) -> tuple[np.ndarray, list[str]]:
    """Load all patches in *patch_dir*, encode with *model*, return (latents, filenames)."""
    ds = PatchDataset(patch_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"))
    latents, filenames = [], []
    with torch.no_grad():
        for batch in loader:
            x      = batch[0].to(device)
            paths  = batch[4]
            if x.ndim == 3:          # [N, H, W] → [N, 1, H, W]
                x = x.unsqueeze(1)
            if input_divisor != 1.0:
                x = x / input_divisor
            z = model.encode(x)
            latents.append(z.cpu().numpy())
            filenames.extend([Path(p).name for p in paths])
    return np.concatenate(latents, axis=0), filenames


def _encode_patches_2ch(model, ch1_dir: str, ch3_dir: str,
                        device: str, batch_size: int) -> tuple[np.ndarray, list[str]]:
    """Load 2-channel patches (ch1+ch3 stacked) and encode; return (latents, filenames)."""
    ds = MultiChannelPatchDataset([ch1_dir, ch3_dir])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"))
    latents, filenames = [], []
    with torch.no_grad():
        for batch in loader:
            x     = batch[0].to(device)   # (B, 2, H, W)
            paths = batch[4]
            z = model.encode(x)
            latents.append(z.cpu().numpy())
            filenames.extend([Path(p).name for p in paths])
    return np.concatenate(latents, axis=0), filenames


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def run_eval(result_dir: Path, kmeans_k: int = 5, knn_k: int = 5,
             batch_size: int = 512, device: str = "auto",
             two_channel: bool = False):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    out_dir = result_dir / "eval"
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load latents.csv                                                  #
    # ------------------------------------------------------------------ #
    csv_path = result_dir / "latents.csv"
    if not csv_path.exists():
        log.error("latents.csv not found in %s – aborting", result_dir)
        return

    df = pd.read_csv(csv_path)
    latent_cols = [c for c in df.columns if c.startswith("z_")]
    latents     = df[latent_cols].values.astype(np.float32)
    log.info("Loaded %d rows, %d latent dims from %s", len(df), len(latent_cols), result_dir.name)

    cond_order  = sorted(df["condition_name"].dropna().unique().tolist())
    split_order = ["train", "val"]

    # Always try to merge annotations from the canonical vinc label CSV so that
    # old runs (which stored all -1 in latents.csv) still get KNN evaluation.
    has_ann = False
    if Path(VINC_LABELS_CSV).exists():
        vinc_lbl = pd.read_csv(VINC_LABELS_CSV)
        vinc_lbl["unique_ID"] = vinc_lbl["unique_ID"].astype(str).apply(lambda p: Path(p).name)
        uid_map = dict(zip(vinc_lbl["unique_ID"], vinc_lbl["classification"].astype(str)))
        # patch filename → unique_ID: replace first underscore with hyphen
        df["_uid"] = df["filename"].apply(lambda f: Path(f).name.replace("_", "-", 1))
        df["annotation_label_name"] = df["_uid"].map(uid_map)
        df.drop(columns=["_uid"], inplace=True)
        has_ann = df["annotation_label_name"].notna().any()
        lbl_to_int = {l: i for i, l in enumerate(FA_LABEL_ORDER)}
        df["annotation_label"] = df["annotation_label_name"].map(lbl_to_int).fillna(-1).astype(int)
    elif ("annotation_label_name" in df.columns and "annotation_label" in df.columns
          and (df["annotation_label"] != -1).any()):
        has_ann = True
    ann_mask = df["annotation_label"] != -1 if has_ann else np.zeros(len(df), bool)

    summary: dict = {"result_dir": result_dir.name}

    # ------------------------------------------------------------------ #
    # 2. UMAP                                                              #
    # ------------------------------------------------------------------ #
    log.info("Step 2: UMAP …")
    from umap import UMAP
    umap_reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_emb = umap_reducer.fit_transform(latents)
    joblib.dump(umap_reducer, str(out_dir / "umap_model.pkl"))
    np.save(str(out_dir / "umap_emb.npy"), umap_emb)

    _scatter(umap_emb, df["condition_name"], cond_order,
             "UMAP – condition", out_dir / "umap_condition.png")
    _scatter(umap_emb, df["split"], split_order,
             "UMAP – split", out_dir / "umap_split.png", cmap="Set1")
    if has_ann:
        _scatter(umap_emb[ann_mask], df.loc[ann_mask, "annotation_label_name"],
                 FA_LABEL_ORDER, "UMAP – FA type (labelled)",
                 out_dir / "umap_annotation.png")

    # ------------------------------------------------------------------ #
    # 3. PHATE                                                             #
    # ------------------------------------------------------------------ #
    phate_emb = None
    try:
        import phate as phate_lib
        log.info("Step 3: PHATE …")
        ph = phate_lib.PHATE(k=5, random_state=42, n_jobs=-1, verbose=0)
        phate_emb = ph.fit_transform(latents)
        joblib.dump(ph, str(out_dir / "phate_model.pkl"))
        np.save(str(out_dir / "phate_emb.npy"), phate_emb)

        _scatter(phate_emb, df["condition_name"], cond_order,
                 "PHATE – condition", out_dir / "phate_condition.png")
        _scatter(phate_emb, df["split"], split_order,
                 "PHATE – split", out_dir / "phate_split.png", cmap="Set1")
        if has_ann:
            _scatter(phate_emb[ann_mask], df.loc[ann_mask, "annotation_label_name"],
                     FA_LABEL_ORDER, "PHATE – FA type (labelled)",
                     out_dir / "phate_annotation.png")
    except ImportError:
        log.warning("Step 3: phate not installed – skipping PHATE")

    # ------------------------------------------------------------------ #
    # 4. KMeans                                                            #
    # ------------------------------------------------------------------ #
    log.info("Step 4: KMeans k=%d …", kmeans_k)
    km = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(latents).astype(str)
    joblib.dump(km, str(out_dir / "kmeans_model.pkl"))

    cluster_order = [str(i) for i in range(kmeans_k)]
    _scatter(umap_emb, cluster_labels, cluster_order,
             f"UMAP – KMeans k={kmeans_k}", out_dir / "umap_kmeans.png")
    if phate_emb is not None:
        _scatter(phate_emb, cluster_labels, cluster_order,
                 f"PHATE – KMeans k={kmeans_k}", out_dir / "phate_kmeans.png")

    # ------------------------------------------------------------------ #
    # 5. KNN classification on vinc (train → val)                         #
    # ------------------------------------------------------------------ #
    if not has_ann:
        log.warning("Step 5: no annotation labels in latents.csv – skipping KNN")
    else:
        log.info("Step 5: KNN classification on vinc (k=%d) …", knn_k)
        labeled = df[df["annotation_label"] != -1].copy()

        train_sel = labeled["split"] == "train"
        val_sel   = labeled["split"] == "val"
        X_tr = latents[labeled.index[train_sel]]
        y_tr = labeled.loc[train_sel, "annotation_label_name"].values
        X_va = latents[labeled.index[val_sel]]
        y_va = labeled.loc[val_sel,   "annotation_label_name"].values

        log.info("  Train labelled: %d   Val labelled: %d", len(X_tr), len(X_va))

        knn = KNeighborsClassifier(n_neighbors=knn_k, metric="euclidean")
        knn.fit(X_tr, y_tr)
        joblib.dump(knn, str(out_dir / "knn_vinc.pkl"))

        y_pred = knn.predict(X_va)
        vinc_acc = accuracy_score(y_va, y_pred)
        log.info("  Vinc val KNN accuracy: %.4f", vinc_acc)
        summary["vinc_val_knn_acc"] = round(vinc_acc, 4)
        summary["vinc_val_n"] = len(y_va)

        present = [l for l in FA_LABEL_ORDER if l in y_va or l in y_pred]
        _confusion_png(y_va, y_pred, present,
                       f"Vinc val  KNN k={knn_k}  acc={vinc_acc:.3f}",
                       out_dir / "confusion_vinc_val.png")
        _confusion_png(y_va, y_pred, present,
                       f"Vinc val  KNN k={knn_k}  row-normalised",
                       out_dir / "confusion_vinc_val_norm.png", normalize="true")

        rep = classification_report(y_va, y_pred, labels=present,
                                    output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(str(out_dir / "cls_report_vinc.csv"))

        # ------------------------------------------------------------------ #
        # 6. ppax inference + KNN                                             #
        # ------------------------------------------------------------------ #
        log.info("Step 6: ppax inference …")
        model_path = result_dir / "model_best.pt"
        if not model_path.exists():
            model_path = result_dir / "model_final.pt"

        if not model_path.exists():
            log.error("  No model file found – skipping ppax step")
        else:
            input_divisor = _get_input_divisor(result_dir)
            log.info("  Loading model from %s  (input_divisor=%.1f)", model_path.name, input_divisor)

            model = torch.load(str(model_path), map_location=device, weights_only=False)
            model.eval()

            all_latents, all_filenames = [], []
            if two_channel:
                for cond in PPAX_PATCH_DIRS:
                    z, fnames = _encode_patches_2ch(
                        model,
                        PPAX_PATCH_DIRS[cond],
                        PPAX_PATCH_DIRS_CH3[cond],
                        device, batch_size,
                    )
                    all_latents.append(z)
                    all_filenames.extend(fnames)
            else:
                for cond, patch_dir in PPAX_PATCH_DIRS.items():
                    z, fnames = _encode_patches(model, patch_dir, input_divisor,
                                                device, batch_size)
                    all_latents.append(z)
                    all_filenames.extend(fnames)
            ppax_latents   = np.concatenate(all_latents, axis=0)
            ppax_filenames = all_filenames

            # Match filenames to ppax labels
            # Patch basename: "control_f0000x0176y0336ps32.tif"
            # Label unique_ID: "control-f0000x0176y0336ps32.tif"
            ppax_label_df  = pd.read_csv(PPAX_LABELS_CSV)
            uid_to_label   = dict(zip(
                ppax_label_df["unique_ID"].astype(str),
                ppax_label_df["classification"].astype(str),
            ))
            ppax_uid  = [f.replace("_", "-", 1) for f in ppax_filenames]
            ppax_true = np.array([uid_to_label.get(u, "unlabelled") for u in ppax_uid])

            labelled = (ppax_true != "unlabelled") & (ppax_true != "Uncertain")
            log.info("  ppax patches: %d total, %d labelled",
                     len(ppax_filenames), labelled.sum())
            summary["ppax_total"] = len(ppax_filenames)
            summary["ppax_labelled"] = int(labelled.sum())

            if labelled.sum() > 0:
                X_pp = ppax_latents[labelled]
                y_pp_true = ppax_true[labelled]
                y_pp_pred = knn.predict(X_pp)

                ppax_acc = accuracy_score(y_pp_true, y_pp_pred)
                log.info("  ppax KNN accuracy: %.4f", ppax_acc)
                summary["ppax_knn_acc"] = round(ppax_acc, 4)

                ppax_present = [l for l in FA_LABEL_ORDER
                                if l in y_pp_true or l in y_pp_pred]
                _confusion_png(y_pp_true, y_pp_pred, ppax_present,
                               f"ppax  KNN k={knn_k}  acc={ppax_acc:.3f}",
                               out_dir / "confusion_ppax.png")
                _confusion_png(y_pp_true, y_pp_pred, ppax_present,
                               f"ppax  KNN k={knn_k}  row-normalised",
                               out_dir / "confusion_ppax_norm.png", normalize="true")

                pp_rep = classification_report(y_pp_true, y_pp_pred,
                                               labels=ppax_present,
                                               output_dict=True, zero_division=0)
                pd.DataFrame(pp_rep).T.to_csv(str(out_dir / "cls_report_ppax.csv"))

            # Save full ppax latents + predictions CSV
            ppax_df = pd.DataFrame(ppax_latents,
                                   columns=[f"z_{i}" for i in range(ppax_latents.shape[1])])
            ppax_df.insert(0, "filename",    ppax_filenames)
            ppax_df.insert(1, "unique_ID",   ppax_uid)
            ppax_df.insert(2, "true_label",  ppax_true)
            ppax_df.insert(3, "pred_label",  knn.predict(ppax_latents))
            ppax_df.to_csv(str(out_dir / "ppax_latents.csv"), index=False)

            # UMAP scatter for ppax (transform with fitted reducer)
            ppax_umap = umap_reducer.transform(ppax_latents)
            _scatter(ppax_umap[labelled], y_pp_true,
                     ppax_present, "UMAP – ppax labelled (true label)",
                     out_dir / "umap_ppax_true.png")
            _scatter(ppax_umap[labelled], y_pp_pred,
                     ppax_present, "UMAP – ppax labelled (KNN pred)",
                     out_dir / "umap_ppax_pred.png")

    # ------------------------------------------------------------------ #
    # 7. Save summary                                                      #
    # ------------------------------------------------------------------ #
    pd.DataFrame([summary]).to_csv(str(out_dir / "eval_summary.csv"), index=False)
    log.info("Eval complete → %s", out_dir)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser(
        description="Evaluate a contrastive AE result dir: UMAP, PHATE, KMeans, KNN cls rate, ppax.")
    p.add_argument("result_dir", type=Path, help="Path to the AE result directory")
    p.add_argument("--kmeans_k",   type=int,   default=5,   help="KMeans clusters (default 5)")
    p.add_argument("--knn_k",      type=int,   default=5,   help="KNN neighbours (default 5)")
    p.add_argument("--batch_size", type=int,   default=512, help="Inference batch size")
    p.add_argument("--device",      default="auto", help="cuda | cpu | auto")
    p.add_argument("--two-channel", action="store_true",
                   help="Use 2-channel (pax+actin) inference for ppax cross-dataset step")
    args = p.parse_args()

    run_eval(args.result_dir,
             kmeans_k=args.kmeans_k,
             knn_k=args.knn_k,
             batch_size=args.batch_size,
             device=args.device,
             two_channel=args.two_channel)


if __name__ == "__main__":
    main()
