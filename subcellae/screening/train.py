"""
train.py
========
Training loop for the EfficientNet binary screening classifier.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def compute_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight = n_neg / n_pos to handle class imbalance."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds  = (logits.detach() > 0).long()
        correct += (preds == labels.long()).sum().item()
        total  += len(labels)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_probs:  list[float] = []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).cpu().tolist()
        all_probs  += probs
        all_labels += labels.long().cpu().tolist()

    all_labels_arr = np.array(all_labels)
    all_probs_arr  = np.array(all_probs)
    preds          = (all_probs_arr > 0.5).astype(int)
    accuracy       = (preds == all_labels_arr).mean()

    return {
        "loss":        total_loss / len(all_labels),
        "accuracy":    float(accuracy),
        "all_labels":  all_labels_arr,
        "all_probs":   all_probs_arr,
        "all_preds":   preds,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    lr_scheduler: str = "cosine",
    pos_weight: Optional[torch.Tensor] = None,
    device: torch.device,
    save_best_path: Optional[str] = None,
    patience: int = 15,
) -> dict:
    """Train the model and return the best val results.

    Parameters
    ----------
    patience : int
        Stop early if val loss does not improve for this many epochs.
    save_best_path : str, optional
        Path to save the best model checkpoint (by val loss).

    Returns
    -------
    dict with keys ``history``, ``best_epoch``, ``best_val``.
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
    elif lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    else:
        scheduler = None

    history: list[dict] = []
    best_val_loss  = float("inf")
    best_epoch     = 0
    best_val       = {}
    no_improve     = 0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate_epoch(model, val_loader, criterion, device)

        if lr_scheduler == "cosine" and scheduler is not None:
            scheduler.step()
        elif lr_scheduler == "plateau" and scheduler is not None:
            scheduler.step(va["loss"])

        history.append({
            "epoch":         epoch,
            "train_loss":    tr["loss"],
            "train_acc":     tr["accuracy"],
            "val_loss":      va["loss"],
            "val_acc":       va["accuracy"],
        })

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.3f  "
            "val_loss=%.4f  val_acc=%.3f  lr=%.2e",
            epoch, epochs,
            tr["loss"], tr["accuracy"],
            va["loss"], va["accuracy"],
            optimizer.param_groups[0]["lr"],
        )

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            best_epoch    = epoch
            best_val      = va
            no_improve    = 0
            if save_best_path:
                torch.save(model.state_dict(), save_best_path)
                log.info("  → Best model saved (epoch %d, val_loss=%.4f)", epoch, best_val_loss)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping: no improvement for %d epochs.", patience)
                break

    return {
        "history":    history,
        "best_epoch": best_epoch,
        "best_val":   best_val,
    }
