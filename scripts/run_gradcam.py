#!/usr/bin/env python3
"""
run_gradcam.py
==============
GradCAM visualisation: which input pixels drive the MLP classifier's class decisions?

Pipeline
--------
  patch (32×32) ──► CNN encoder ──► z (12-dim) ──► PyTorch-MLP ──► class logit
                         ▲                                              │
                         └──────────── GradCAM gradients ◄─────────────┘

The sklearn MLP weights are ported to a PyTorch nn.Module so that autograd
can propagate from class logit back through z, through the encoder FC layers,
and into the last conv-layer feature maps (128 × 4 × 4).

GradCAM is computed at `encoder[-1] conv` (Conv2d 64→128, output 4×4),
upsampled to 32×32 and overlaid on the original patch.

Usage
-----
  python scripts/run_gradcam.py \\
      --ae-dir   ae_results/contrastive_run/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1 \\
      --cls-dir  ae_results/contrastive_run/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1/fa_cls_zrecon_mlp \\
      --out-dir  ae_results/gradcam/supcon_vinc_ppax_enlcrop_sc2_l1 \\
      --n-per-class 16

Output
------
  out_dir/
    gradcam_class_{name}.png   — grid: original | GradCAM overlay, for each true class
    gradcam_all_classes.png    — combined figure across all FA types
    gradcam_summary.csv        — per-patch mean GradCAM activation
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.modelling.autoencoders import ContrastiveAE, AE, SemiSupAE
from subcellae.modelling.dataset import PatchDataset

ROOT = "/net/projects/CLS/lding/data/fa_data_analysis"

FA_LABELS = {
    0: "Nascent Adhesion",
    1: "focal complex",
    2: "focal adhesion",
    3: "fibrillar adhesion",
    4: "No adhesion",
}
FA_COLORS = {
    "Nascent Adhesion":    "#e41a1c",
    "focal complex":       "#377eb8",
    "focal adhesion":      "#4daf4a",
    "fibrillar adhesion":  "#984ea3",
    "No adhesion":         "#888888",
}


# ── PyTorch MLP wrapper ───────────────────────────────────────────────────────

class SklearnMLPTorch(nn.Module):
    """Wrap a fitted sklearn MLPClassifier as a differentiable PyTorch module."""

    def __init__(self, sklearn_mlp):
        super().__init__()
        coefs = sklearn_mlp.coefs_
        intercepts = sklearn_mlp.intercepts_
        layers = []
        for i, (W, b) in enumerate(zip(coefs, intercepts)):
            lin = nn.Linear(W.shape[0], W.shape[1])
            lin.weight = nn.Parameter(torch.tensor(W.T, dtype=torch.float32))
            lin.bias   = nn.Parameter(torch.tensor(b,   dtype=torch.float32))
            layers.append(lin)
            if i < len(coefs) - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# ── model loading ─────────────────────────────────────────────────────────────

def load_ae(ae_dir: Path, device: str):
    yaml_files = list(ae_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config in {ae_dir}")
    import yaml
    cfg = yaml.safe_load(yaml_files[0].read_text())

    model_cfg = cfg.get("model", {})
    mtype      = model_cfg.get("model_type", "ae")
    latent_dim = model_cfg.get("latent_dim", 12)
    proj_dim   = model_cfg.get("proj_dim", 8)
    no_ch      = model_cfg.get("no_ch", 1)
    bn_flag    = model_cfg.get("BN_flag", False)
    dropout    = model_cfg.get("dropout_flag", False)
    sigmoid    = model_cfg.get("output_sigmoid", True)

    if mtype in ("contrastive", "supcon"):
        model = ContrastiveAE(
            latent_dim=latent_dim, proj_dim=proj_dim,
            no_ch=no_ch, BN_flag=bn_flag, output_sigmoid=sigmoid,
        )
    elif mtype == "semisup":
        model = SemiSupAE(
            latent_dim=latent_dim, no_ch=no_ch, BN_flag=bn_flag,
            dropout_flag=dropout, output_sigmoid=sigmoid,
        )
    else:
        model = AE(
            latent_dim=latent_dim, no_ch=no_ch, BN_flag=bn_flag,
            dropout_flag=dropout, output_sigmoid=sigmoid,
        )

    ckpt = ae_dir / "model_best.pt"
    if not ckpt.exists():
        ckpt = ae_dir / "model_final.pt"
    loaded = torch.load(str(ckpt), map_location=device, weights_only=False)
    if isinstance(loaded, nn.Module):
        # checkpoint is the full model object (ContrastiveAE saved directly)
        model = loaded
    else:
        model.load_state_dict(loaded)
    model.to(device).eval()
    return model


def find_last_conv(encoder: nn.Sequential):
    """Return the last Conv2d layer inside encoder (before Flatten)."""
    last = None
    for m in encoder.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


# ── GradCAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(ae_model, mlp_torch, x, class_idx, target_conv,
                    use_projector: bool = False):
    """
    Compute GradCAM heatmap for one patch tensor x (1,1,H,W).

    use_projector : if True, route through ae_model.project(z) before MLP,
                   so gradient path is: logit -> MLP(p) -> projector -> z -> conv.
                   if False (default): logit -> MLP(z) -> z -> conv.

    Returns heatmap as np.ndarray (H,W), normalised 0–1.
    """
    activations = {}
    gradients   = {}

    def fwd_hook(m, inp, out):
        activations["feat"] = out

    def bwd_hook(m, grad_in, grad_out):
        gradients["feat"] = grad_out[0]

    h1 = target_conv.register_forward_hook(fwd_hook)
    h2 = target_conv.register_full_backward_hook(bwd_hook)

    x = x.requires_grad_(False)
    z = ae_model.encode(x)                      # (1, latent_dim)
    if use_projector:
        features = ae_model.project(z)           # (1, proj_dim)
    else:
        features = z
    logits = mlp_torch(features)                 # (1, n_classes)
    score  = logits[0, class_idx]

    ae_model.zero_grad()
    mlp_torch.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    feat = activations["feat"].detach()   # (1, C, h, w)
    grad = gradients["feat"].detach()     # (1, C, h, w)

    weights = grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
    cam = (weights * feat).sum(dim=1, keepdim=True)  # (1, 1, h, w)
    cam = F.relu(cam)

    # Upsample to input size
    cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                        mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalise
    vmin, vmax = cam.min(), cam.max()
    if vmax > vmin:
        cam = (cam - vmin) / (vmax - vmin)
    else:
        cam = np.zeros_like(cam)
    return cam


# ── visualisation ─────────────────────────────────────────────────────────────

def overlay_gradcam(patch_np, cam_np, alpha=0.5):
    """Blend patch (H,W) with GradCAM heatmap, return RGB (H,W,3)."""
    patch_rgb = np.stack([patch_np] * 3, axis=-1)
    patch_rgb = np.clip(patch_rgb, 0, 1)
    heatmap   = cm.jet(cam_np)[..., :3]
    blended   = (1 - alpha) * patch_rgb + alpha * heatmap
    return np.clip(blended, 0, 1)


def make_class_grid(patches, cams, pred_labels, true_label_name,
                    class_names, filenames=None, n_cols=8):
    """
    Make a figure with rows: [original | gradcam_class0 | gradcam_class1 | ...]
    patches   : list of (H,W) np arrays
    cams      : list of dict {class_idx: cam_np}
    filenames : list of str, shown as column titles
    """
    n_patches  = len(patches)
    n_classes  = len(class_names)
    n_rows_fig = 1 + n_classes
    n_pages    = (n_patches + n_cols - 1) // n_cols

    row_labels = ["original"] + list(class_names.values())

    figs = []
    for page in range(n_pages):
        start = page * n_cols
        end   = min(start + n_cols, n_patches)
        batch_patches = patches[start:end]
        batch_cams    = cams[start:end]
        batch_preds   = pred_labels[start:end]
        batch_names   = (filenames[start:end] if filenames is not None
                         else [f"#{start+i}" for i in range(end - start)])
        nc = len(batch_patches)

        # Extra left margin for row labels
        fig, axes = plt.subplots(n_rows_fig, nc,
                                 figsize=(nc * 1.5 + 1.5, n_rows_fig * 1.5),
                                 squeeze=False)
        fig.suptitle(f"True class: {true_label_name}", fontsize=10, y=1.01)

        for col, (patch, cam_dict, pred, fname) in enumerate(
                zip(batch_patches, batch_cams, batch_preds, batch_names)):
            # Row 0: original patch — title = filename + prediction
            ax = axes[0][col]
            ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
            short_name = fname[-22:] if len(fname) > 22 else fname
            ax.set_title(f"{short_name}\npred: {pred}", fontsize=4.5,
                         wrap=False)
            ax.axis("off")

            # Rows 1+: GradCAM per class
            for row, (cidx, cname) in enumerate(class_names.items()):
                ax = axes[row + 1][col]
                cam = cam_dict.get(cidx, np.zeros_like(patch))
                ax.imshow(overlay_gradcam(patch, cam), vmin=0, vmax=1)
                ax.axis("off")

        # Row labels on left edge using ax.text (axis("off") hides set_ylabel)
        for row_i, label in enumerate(row_labels):
            ax = axes[row_i][0]
            ax.text(-0.12, 0.5, label, transform=ax.transAxes,
                    fontsize=6, rotation=0, ha="right", va="center",
                    clip_on=False,
                    fontweight="bold" if row_i == 0 else "normal")

        plt.tight_layout()
        figs.append(fig)
    return figs


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-dir",  required=True)
    parser.add_argument("--cls-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-per-class", type=int, default=16,
                        help="Max patches per true FA class to visualise")
    parser.add_argument("--input-divisor", type=float, default=None,
                        help="Scale patches by 1/x before encoding (e.g. 2.0 for sc2)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-projector", action="store_true",
                        help="Route through projector head (p_) before MLP")
    args = parser.parse_args()

    ae_dir  = Path(args.ae_dir)
    cls_dir = Path(args.cls_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device: {device}")

    # ── detect input divisor from yaml if not given ──────────────────────────
    input_divisor = args.input_divisor
    if input_divisor is None:
        import yaml
        yf = list(ae_dir.glob("*.yaml"))
        if yf:
            cfg = yaml.safe_load(yf[0].read_text())
            elc = cfg.get("enlarged_crop", {})
            input_divisor = float(elc.get("input_divisor", 1.0))
        else:
            input_divisor = 1.0
    print(f"Input divisor: {input_divisor}")

    # ── load AE ──────────────────────────────────────────────────────────────
    print("Loading AE model...")
    ae_model = load_ae(ae_dir, device)
    ae_model.eval()

    # ── find target conv layer ───────────────────────────────────────────────
    target_conv = find_last_conv(ae_model.encoder)
    print(f"GradCAM target layer: {target_conv}")

    # ── load + wrap sklearn MLP ───────────────────────────────────────────────
    print("Loading MLP classifier...")
    sklearn_mlp = joblib.load(cls_dir / "model.pkl")
    mlp_torch   = SklearnMLPTorch(sklearn_mlp).to(device)
    mlp_torch.eval()
    class_names  = {i: name for i, name in enumerate(sklearn_mlp.classes_)}
    # map sklearn class int → FA label string
    label_map = FA_LABELS  # assume 0-4 → FA type

    # ── load latents + annotations ───────────────────────────────────────────
    print("Loading latents...")
    lat_csv = ae_dir / "latents.csv"
    lat_df  = pd.read_csv(lat_csv)

    # Load vinc annotations
    vinc_lbl = pd.read_csv(f"{ROOT}/labelling/labels_vinc_20260521.csv")
    vinc_lbl["_uid"] = vinc_lbl["unique_ID"].astype(str).apply(
        lambda p: Path(p).name)
    uid_map = dict(zip(vinc_lbl["_uid"], vinc_lbl["classification"].astype(str)))
    lat_df["_uid"] = lat_df["filename"].apply(
        lambda f: Path(f).name.replace("_", "-", 1))
    lat_df["true_label"] = lat_df["_uid"].map(uid_map)

    # Keep vinc-only val-split labelled patches.
    # Multi-dataset models (e.g. vinc+ppax) have ppax patches with the same
    # filenames as vinc patches; filtering by filepath avoids ppax rows
    # inheriting vinc labels and showing up as wrongly-looking patches.
    lat_df = lat_df[lat_df["filepath"].str.contains("/vinc/")].copy()
    lat_df = lat_df[lat_df["true_label"].notna()].copy()
    lat_df = lat_df[lat_df["split"] == "val"].copy()
    lat_df = lat_df[lat_df["true_label"] != "Uncertain"].copy()
    print(f"Labelled val patches: {len(lat_df)}")

    # ── run GradCAM per patch ────────────────────────────────────────────────
    label_to_idx = {v: k for k, v in FA_LABELS.items()}
    results = []

    all_patches_by_class   = {}
    all_cams_by_class      = {}
    all_preds_by_class     = {}
    all_filenames_by_class = {}

    import tifffile

    for true_label in FA_LABELS.values():
        # Deduplicate by filepath so the same patch can't appear twice
        subset = (lat_df[lat_df["true_label"] == true_label]
                  .drop_duplicates("filepath")
                  .head(args.n_per_class))
        if len(subset) == 0:
            continue
        print(f"  {true_label}: {len(subset)} patches")

        patches_list, cams_list, preds_list, filenames_list = [], [], [], []
        for _, row in subset.iterrows():
            patch_path = row["filepath"]
            if not Path(patch_path).exists():
                continue

            patch_np = tifffile.imread(str(patch_path)).astype(np.float32)
            if patch_np.ndim == 3:
                patch_np = patch_np[0]

            x = torch.tensor(patch_np[None, None], dtype=torch.float32,
                              device=device)
            if input_divisor != 1.0:
                x = x / input_divisor

            # GradCAM for every class
            cam_dict = {}
            with torch.enable_grad():
                for cidx in FA_LABELS:
                    cam_dict[cidx] = compute_gradcam(
                        ae_model, mlp_torch, x, cidx, target_conv,
                        use_projector=args.use_projector)

            # Predicted class
            with torch.no_grad():
                z      = ae_model.encode(x)
                feats  = ae_model.project(z) if args.use_projector else z
                logits = mlp_torch(feats)
                pred_idx = int(logits.argmax(dim=1).item())
            pred_name = FA_LABELS[pred_idx]

            # Normalize patch for display: 1st–99th percentile → [0, 1]
            # matches the contrast stretch used by the labeller interface
            p_disp = patch_np.copy()
            lo, hi = np.percentile(p_disp, 1), np.percentile(p_disp, 99)
            if hi > lo:
                p_disp = np.clip((p_disp - lo) / (hi - lo), 0, 1)
            else:
                p_disp = np.zeros_like(p_disp)

            patches_list.append(p_disp)
            cams_list.append(cam_dict)
            preds_list.append(pred_name)
            filenames_list.append(Path(patch_path).name)

            results.append({
                "filepath"      : patch_path,
                "true_label"    : true_label,
                "pred_label"    : pred_name,
                "correct"       : true_label == pred_name,
                "mean_cam_true" : cam_dict.get(label_to_idx.get(true_label, -1),
                                               np.zeros((32,32))).mean(),
            })

        all_patches_by_class[true_label]   = patches_list
        all_cams_by_class[true_label]      = cams_list
        all_preds_by_class[true_label]     = preds_list
        all_filenames_by_class[true_label] = filenames_list

    # ── save per-class grids ─────────────────────────────────────────────────
    print("Saving visualisations...")
    for true_label in all_patches_by_class:
        figs = make_class_grid(
            all_patches_by_class[true_label],
            all_cams_by_class[true_label],
            all_preds_by_class[true_label],
            true_label,
            FA_LABELS,
            filenames=all_filenames_by_class[true_label],
            n_cols=8,
        )
        safe_name = true_label.replace(" ", "_").replace("/", "_")
        for i, fig in enumerate(figs):
            out_path = out_dir / f"gradcam_{safe_name}_p{i}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")

    # ── combined summary figure ───────────────────────────────────────────────
    n_classes_present = len(all_patches_by_class)
    if n_classes_present > 0:
        fig, axes = plt.subplots(
            n_classes_present, 1,
            figsize=(16, 3 * n_classes_present),
            squeeze=False,
        )
        fig.suptitle("GradCAM — mean heatmap per true FA class", fontsize=12)

        for row_idx, true_label in enumerate(FA_LABELS.values()):
            if true_label not in all_patches_by_class:
                continue
            ax = axes[row_idx][0]
            cams = all_cams_by_class[true_label]
            # Average GradCAM for the "true class" channel across all patches
            true_cidx = label_to_idx.get(true_label, 0)
            mean_cam = np.mean(
                [c[true_cidx] for c in cams if true_cidx in c], axis=0)
            ax.imshow(mean_cam, cmap="jet", vmin=0, vmax=1)
            ax.set_title(f"{true_label} (n={len(cams)})", fontsize=9)
            ax.axis("off")
            fig.colorbar(
                cm.ScalarMappable(cmap="jet"),
                ax=ax, fraction=0.03, pad=0.02)

        plt.tight_layout()
        fig.savefig(out_dir / "gradcam_mean_per_class.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # ── save summary CSV ──────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(out_dir / "gradcam_summary.csv", index=False)
    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
