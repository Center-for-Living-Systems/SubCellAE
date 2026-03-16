"""
Wrapper script for the SubCellAE analysis pipeline.

Usage
-----
    python scripts/run_analysis_pipeline.py --config configs/analysis_config.yaml

The script:
  1. Reads the YAML config
  2. Loads the trained model and dataloader
  3. Calls subcellae/pipeline/analysis_pipeline.py::run_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Make sure the repo root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.autoencoders import AE
from core.dataset import TIFFDataset
from subcellae.pipeline.analysis_pipeline import run_analysis
from torch.utils.data import DataLoader


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloader(cfg: dict, device: torch.device) -> DataLoader:
    dataset = TIFFDataset(**cfg["dataset"])
    return DataLoader(
        dataset,
        batch_size=cfg["dataloader"].get("batch_size", 64),
        shuffle=False,
        num_workers=cfg["dataloader"].get("num_workers", 0),
    )


def load_model(cfg: dict, device: torch.device):
    model = AE(**cfg["model"]["params"])
    state = torch.load(cfg["model"]["checkpoint_path"], map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run SubCellAE analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model      = load_model(cfg, device)
    dataloader = build_dataloader(cfg, device)

    run_analysis(
        model=model,
        dataloader=dataloader,
        device=device,
        out_dir=cfg["out_dir"],
        label_csv=cfg.get("label_csv"),
        embedding=cfg.get("embedding", {"methods": ["UMAP"]}),
        clustering=cfg.get("clustering", {"kmeans": {"enabled": False}, "dbscan": {"enabled": False}}),
        label_orders=cfg.get("label_orders"),
        latent_source=cfg.get("latent_source", "mu"),
    )


if __name__ == "__main__":
    main()
