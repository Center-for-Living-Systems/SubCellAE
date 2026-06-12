"""
evaluate.py
===========
Metrics, plots, and predictions CSV for the binary screening classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)

LABEL_NAMES = ["no adhesion", "adhesion"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: Sequence[str] = LABEL_NAMES,
) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_binary":         f1_score(y_true, y_pred, average="binary", zero_division=0),
        "roc_auc":           roc_auc_score(y_true, y_proba),
        "avg_precision":     average_precision_score(y_true, y_proba),
        "report":            classification_report(
                                 y_true, y_pred, target_names=class_names, zero_division=0
                             ),
        "report_dict":       report,
        "confusion_matrix":  confusion_matrix(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] = LABEL_NAMES,
    *,
    normalize: bool = False,
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(all="ignore"):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(5, 4))
    if _SNS_AVAILABLE:
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5)
    else:
        im = ax.imshow(cm_plot, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = f"{cm_plot[i, j]:.2f}" if normalize else str(int(cm_plot[i, j]))
                ax.text(j, i, val, ha="center", va="center", fontsize=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or ("Confusion matrix (normalised)" if normalize else "Confusion matrix (counts)"))
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    auc: float,
    save_path: Optional[str | Path] = None,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (adhesion vs no adhesion)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    ap: float,
    save_path: Optional[str | Path] = None,
) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_prob_histogram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Sequence[str] = LABEL_NAMES,
    *,
    save_path: Optional[str | Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls_id, cls_name in enumerate(class_names):
        mask = y_true == cls_id
        if mask.any():
            ax.hist(y_proba[mask], bins=40, alpha=0.6, label=f"true: {cls_name}", density=True)
    ax.set_xlabel("P(adhesion)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted probability distribution by true class")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_training_curves(
    history: list[dict],
    *,
    save_path: Optional[str | Path] = None,
) -> None:
    epochs     = [h["epoch"]     for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_acc  = [h["train_acc"]  for h in history]
    val_acc    = [h["val_acc"]    for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss,   label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc,   label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curves")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def save_metrics_txt(metrics: dict, out_dir: Path, split: str = "val") -> None:
    lines = [
        f"# ── {split} metrics ──────────────────────────────────────",
        f"accuracy          : {metrics['accuracy']:.4f}",
        f"balanced_accuracy : {metrics['balanced_accuracy']:.4f}",
        f"f1_macro          : {metrics['f1_macro']:.4f}",
        f"f1_binary         : {metrics['f1_binary']:.4f}",
        f"roc_auc           : {metrics['roc_auc']:.4f}",
        f"avg_precision     : {metrics['avg_precision']:.4f}",
        "",
        "Classification report:",
        metrics["report"],
    ]
    (out_dir / "metrics.txt").write_text("\n".join(lines))


def save_metrics_csv(metrics: dict, out_dir: Path) -> None:
    report_dict = metrics["report_dict"]
    rows = [
        {
            "class":     cls,
            "precision": report_dict[cls]["precision"],
            "recall":    report_dict[cls]["recall"],
            "f1":        report_dict[cls]["f1-score"],
            "support":   report_dict[cls]["support"],
        }
        for cls in LABEL_NAMES
        if cls in report_dict
    ]
    pd.DataFrame(rows).to_csv(str(out_dir / "metrics.csv"), index=False)


def save_predictions_csv(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
    split_col: Optional[str] = None,
) -> None:
    out = df[["_patch_path", "unique_ID", "condition", "czi_filename",
              "_binary_label", "classification"]].copy()
    out = out.rename(columns={"_patch_path": "filepath", "_binary_label": "true_label"})
    out["pred_label"]       = y_pred
    out["pred_label_name"]  = [LABEL_NAMES[p] for p in y_pred]
    out["prob_adhesion"]    = y_proba
    if split_col and split_col in df.columns:
        out["split"] = df[split_col].values
    out.to_csv(str(out_dir / "predictions_all.csv"), index=False)
