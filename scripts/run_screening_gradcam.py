#!/usr/bin/env python3
"""
run_screening_gradcam.py
========================
Generate Grad-CAM visualizations for a trained binary adhesion screener.

For each labeled patch (train + val) a side-by-side PNG is saved:
  left:  raw 32×32 grayscale patch (original TIFF, intensity-clipped to [0,1])
  right: Grad-CAM heatmap overlaid on the patch (jet colormap, alpha=0.5)

The PNG filename matches the patch basename.  Outputs are organised as:
  {out_dir}/{split}/{patch_basename}.png      (train / val)

The script reads the saved predictions_all.csv from the run directory to
determine patch paths and split assignments.  The YAML config in the run
directory is used to reconstruct model architecture / size.

Usage
-----
python scripts/run_screening_gradcam.py \
    --model_dir /path/to/diversity/jitter_mc_efficientnet_b0_sz224 \
    [--out_dir   /path/to/gradcam_output]   # default: {model_dir}/gradcam
    [--device    auto]                       # auto | cpu | cuda
    [--log_level INFO]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import yaml
from scipy.ndimage import zoom as _zoom

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from subcellae.screening.model import ScreeningClassifier
from subcellae.screening.dataset import build_transforms

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Minimal Grad-CAM for a single-logit binary classifier.

    Registers forward and backward hooks on *target_layer*.  Call with a
    single-sample tensor (batch size 1) to get the normalised CAM array.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_act)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _mod, _inp, output):
        self._activations = output.detach()

    def _save_grad(self, _mod, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Return normalised CAM as float32 array in [0, 1].

        Parameters
        ----------
        x : torch.Tensor, shape (1, 3, H, W)
            Pre-processed input (ImageNet-normalised, on the correct device).
        """
        self.model.zero_grad()
        x = x.requires_grad_(True)
        logit = self.model(x)           # scalar logit, shape (1,)
        logit.backward()                # grad w.r.t. the single output unit

        # weights: global average of gradients over spatial dims → (C,)
        weights = self._gradients.mean(dim=(2, 3)).squeeze(0)
        acts    = self._activations.squeeze(0)   # (C, H, W)

        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (H, W)
        cam_np = cam.cpu().float().numpy()

        # EfficientNet zero-padding concentrates artefacts in the 1-pixel border
        # of the spatial feature map.  Zero it out before normalisation.
        h, w = cam_np.shape
        if h > 2 and w > 2:
            cam_np[0, :]  = 0.0
            cam_np[-1, :] = 0.0
            cam_np[:, 0]  = 0.0
            cam_np[:, -1] = 0.0

        # If the interior has no meaningful activation (e.g. a confidently
        # no-adhesion patch where all signal was border artefact), return a
        # flat-zero map so it renders as uniform blue rather than stretching noise.
        if cam_np.max() < 1e-6:
            return cam_np.astype(np.float32)

        # Percentile-based normalisation (2–98 %) for robustness.
        lo = float(np.percentile(cam_np, 2))
        hi = float(np.percentile(cam_np, 98))
        cam_np = np.clip((cam_np - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        return cam_np.astype(np.float32)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model: ScreeningClassifier) -> nn.Module:
    """Return the last conv layer to use as the Grad-CAM target."""
    name = model.backbone_name
    bb   = model.backbone

    if name.startswith("efficientnet"):
        # timm EfficientNet: conv_head is the 1×1 pointwise conv before global pool
        # Output shape: (B, 1280, 7, 7) for 224px input
        return bb.conv_head

    if name.startswith("resnet") or name.startswith("resnext"):
        # timm ResNet: last conv in layer4's final block
        return bb.layer4[-1].conv3 if hasattr(bb.layer4[-1], "conv3") else bb.layer4[-1].conv2

    if name.startswith("mobilenet"):
        return bb.conv_head

    # Generic fallback: walk backwards to find the last Conv2d
    for _, module in reversed(list(bb.named_modules())):
        if isinstance(module, nn.Conv2d):
            log.warning("Using fallback target layer for backbone '%s'", name)
            return module

    raise ValueError(f"Cannot determine Grad-CAM target layer for backbone: {name}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_gradcam_figure(
    raw_patch: np.ndarray,
    cam:       np.ndarray,
    title:     str = "",
) -> plt.Figure:
    """Create a side-by-side figure: raw patch (left) | CAM overlay (right).

    Parameters
    ----------
    raw_patch : np.ndarray, shape (H, W), float in [0, 1]
        Original TIFF values clipped to [0, 1].
    cam : np.ndarray, shape (H_cam, W_cam), float in [0, 1]
        Grad-CAM output (any spatial size — upsampled to match raw_patch).
    title : str
        Suptitle text.
    """
    h, w = raw_patch.shape

    # Upsample CAM to match raw patch size
    cam_up = _zoom(cam, (h / cam.shape[0], w / cam.shape[1]), order=1)
    cam_up = np.clip(cam_up, 0.0, 1.0)

    # Jet colourmap overlay
    jet_rgba = plt.cm.jet(cam_up)          # (H, W, 4)
    overlay = raw_patch[:, :, None] * np.ones((1, 1, 3))   # greyscale → RGB
    alpha   = 0.5
    blended = (1 - alpha) * overlay + alpha * jet_rgba[:, :, :3]
    blended = np.clip(blended, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))

    axes[0].imshow(raw_patch, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Raw patch", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title("Grad-CAM", fontsize=8)
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=7, y=1.01)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------

def _load_run_config(model_dir: Path) -> dict:
    """Load the YAML config saved alongside the model checkpoint."""
    yamls = sorted(model_dir.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(f"No YAML config found in {model_dir}")
    if len(yamls) > 1:
        log.warning("Multiple YAML files found — using %s", yamls[0])
    raw = yamls[0].read_text()
    return yaml.safe_load(raw)


def _eval_value(val, root_folder: str):
    """Resolve simple 'root_folder + "..."' string expressions from YAML."""
    if isinstance(val, str) and "root_folder" in val:
        # e.g. root_folder + "/labelling/labels_vinc_20260521.csv"
        val = val.replace("root_folder", repr(root_folder))
        try:
            return eval(val)
        except Exception:
            return val
    return val


def _resolve_config(cfg_raw: dict) -> dict:
    """Recursively resolve root_folder references in the config dict."""
    root = cfg_raw.get("root_folder", "")
    if isinstance(root, str) and root.startswith('"') and root.endswith('"'):
        root = root[1:-1]

    def _recurse(obj):
        if isinstance(obj, dict):
            return {k: _recurse(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_recurse(i) for i in obj]
        if isinstance(obj, str):
            return _eval_value(obj, root)
        return obj

    return _recurse(cfg_raw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM for adhesion screener")
    p.add_argument("--model_dir", required=True,
                   help="Run directory containing model_best.pt and a YAML config")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: {model_dir}/gradcam)")
    p.add_argument("--device", default="auto",
                   help="Device: auto | cpu | cuda")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    model_dir = Path(args.model_dir).resolve()
    out_dir   = Path(args.out_dir).resolve() if args.out_dir else model_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("model_dir : %s", model_dir)
    log.info("out_dir   : %s", out_dir)

    # ------------------------------------------------------------------ device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("device    : %s", device)

    # ---------------------------------------------------------- load run config
    cfg_raw  = _load_run_config(model_dir)
    cfg      = _resolve_config(cfg_raw)

    backbone   = cfg.get("model", {}).get("backbone",   "efficientnet_b0")
    input_size = cfg.get("model", {}).get("input_size",  224)
    dropout    = cfg.get("model", {}).get("dropout",     0.3)

    log.info("backbone   : %s  input_size=%d", backbone, input_size)

    # ------------------------------------------------------------- load model
    model_path = model_dir / "model_best.pt"
    log.info("Loading model from %s", model_path)
    model = ScreeningClassifier(
        backbone=backbone,
        pretrained=False,
        dropout=dropout,
        img_size=input_size,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # -------------------------------------------------- Grad-CAM target layer
    target_layer = get_target_layer(model)
    log.info("GradCAM target layer: %s", target_layer.__class__.__name__)
    gradcam = GradCAM(model, target_layer)

    # --------------------------------------------- load patch list from CSV
    pred_csv = model_dir / "predictions_all.csv"
    if not pred_csv.exists():
        log.error("predictions_all.csv not found in %s", model_dir)
        sys.exit(1)

    df = pd.read_csv(pred_csv)
    # Fix legacy truncation: np.full("val") truncated "train" → "tra"
    if "split" in df.columns:
        df["split"] = df["split"].replace({"tra": "train"})
    log.info("Loaded %d patches from predictions_all.csv", len(df))

    required_cols = {"filepath", "split", "true_label", "pred_label",
                     "prob_adhesion", "classification"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error("Missing columns in CSV: %s", missing)
        sys.exit(1)

    # ---------------------------------------------- preprocessing transform
    val_transform = build_transforms(input_size, augment=False)

    # ----------------------------------------------------------------- loop
    label_names = {0: "no-ad", 1: "ad"}

    n_total = len(df)
    n_done  = 0
    n_err   = 0

    for _, row in df.iterrows():
        patch_path = Path(str(row["filepath"]))
        if not patch_path.exists():
            log.warning("Patch not found: %s", patch_path)
            n_err += 1
            continue

        split       = str(row.get("split", "unknown"))
        true_lbl    = int(row["true_label"])
        pred_lbl    = int(row["pred_label"])
        prob_ad     = float(row["prob_adhesion"])
        fine_label  = str(row.get("classification", ""))

        # Raw patch (for display)
        raw = tifffile.imread(str(patch_path)).astype(np.float32)
        raw_display = np.clip(raw, 0.0, 1.0)

        # Preprocessed tensor for model
        raw_clipped = np.clip(raw, 0.0, 1.0)
        tensor = torch.from_numpy(raw_clipped).unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)
        tensor = val_transform(tensor).unsqueeze(0).to(device)               # (1,3,H,W)

        # Grad-CAM
        try:
            cam = gradcam(tensor)  # (H_cam, W_cam)
        except Exception as exc:
            log.warning("GradCAM failed for %s: %s", patch_path.name, exc)
            n_err += 1
            continue

        # Title
        correct = "OK" if true_lbl == pred_lbl else "ERR"
        title = (
            f"{patch_path.stem}  |  "
            f"true={label_names[true_lbl]} ({fine_label})  "
            f"pred={label_names[pred_lbl]}  p={prob_ad:.2f}  [{correct}]"
        )

        # Save
        save_dir = out_dir / split
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / (patch_path.stem + ".png")

        fig = make_gradcam_figure(raw_display, cam, title=title)
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
        plt.close(fig)

        n_done += 1
        if n_done % 200 == 0:
            log.info("  %d / %d patches done  (%d errors)", n_done, n_total, n_err)

    gradcam.remove_hooks()

    log.info("Done.  %d saved,  %d errors.  Output → %s", n_done, n_err, out_dir)


if __name__ == "__main__":
    main()
