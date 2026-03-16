"""
Analysis pipeline for SubCellAE.

Runs the full post-training analysis sequence:
  1. Extract latent vectors + compute reconstruction MSE from a dataloader
  2. Compute 2-D embedding (UMAP and/or PHATE)
  3. Cluster latents (KMeans and/or DBSCAN)
  4. Build merged label + latent DataFrame
  5. Latent dimension correlation (heatmap + CSV)
  6. Latent dimensions by label (box/violin plots)
  7. Class distribution bar chart
  8. Crosstab heatmap (classification vs position)
  9. Per-class latent dim statistics (mean/std CSV + heatmap)
  10. Reconstruction MSE per sample (distribution plot + per-class violin)

All artefacts (models, CSVs, plots) are saved to out_dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from utils.feature_analysis import (
    dataloader_model_latents,
    compute_2d_embedding,
    build_label_latent_df,
)
from utils.clustering import kmeans_cluster, DBSCAN_cluster
from utils.plotting_utils import (
    plot_2d_scatter,
    plot_latent_pairwise_correlation,
    plot_latent_vs_label_boxplots,
    plot_labels_on_embedding,
    plot_class_distribution,
    plot_crosstab_heatmap,
)


def run_analysis(
    model,
    dataloader,
    device,
    out_dir: str | Path,
    *,
    label_csv: Optional[str | Path] = None,
    embedding: dict,
    clustering: dict,
    label_orders: Optional[dict] = None,
    latent_source: str = "mu",
) -> dict:
    """
    Run the full analysis pipeline and save all outputs.

    Parameters
    ----------
    model : trained AE/VAE model
    dataloader : DataLoader — batch format (x, group_id, _)
    device : torch.device
    out_dir : str or Path
        All artefacts (pkl, csv, png) are written here.
    label_csv : str or Path, optional
        Path to combined labels CSV. Steps 4-8 are skipped if not provided.
    embedding : dict
        Controls dim-reduction. Keys:
          - ``methods``  : list of str, e.g. ``["UMAP", "PHATE"]``
          - ``umap``     : dict of kwargs forwarded to UMAP constructor
          - ``phate``    : dict of kwargs forwarded to PHATE constructor
    clustering : dict
        Controls clustering. Keys:
          - ``kmeans`` : dict with ``enabled`` (bool) and ``n_clusters`` (int)
          - ``dbscan`` : dict with ``enabled`` (bool), ``eps`` (float), ``min_samples`` (int)
          - ``boxplot_kind`` : str, ``"box"`` or ``"violin"`` (default: ``"box"``)
    label_orders : dict, optional
        Ordered label lists used for plots. Keys:
          - ``classification`` : list of str
          - ``position``       : list of str
        If not provided, unique values from the data are used (unordered).
    latent_source : str
        For VAE outputs: ``"mu"`` or ``"z"``; ignored for plain AE.

    Returns
    -------
    dict with keys: ``latents``, ``embeddings``, ``cluster_labels``, ``merged_df``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── Step 1: Extract latents ───────────────────────────────────────────────
    print("Step 1: Extracting latents and computing reconstruction MSE...")
    latents, images, group_ids = dataloader_model_latents(
        model, dataloader, device, latent_source=latent_source
    )

    # Compute per-sample MSE against reconstructions
    mse_list = []
    model.eval()
    with torch.no_grad():
        for x, _, _ in dataloader:
            x = x.to(device)
            out = model(x)
            recon = out[0]
            mse = ((recon - x) ** 2).mean(dim=[1, 2, 3])
            mse_list.append(mse.cpu().numpy())
    sample_mse = np.concatenate(mse_list, axis=0)

    joblib.dump(latents, out_dir / "latents.pkl")
    np.save(out_dir / "sample_mse.npy", sample_mse)
    print(f"  Latents shape: {latents.shape}")
    print(f"  MSE — mean: {sample_mse.mean():.4f}, std: {sample_mse.std():.4f}")
    results["latents"] = latents
    results["sample_mse"] = sample_mse

    latent_cols = [f"lat_d{i}" for i in range(latents.shape[1])]
    latents_df  = pd.DataFrame(latents, columns=latent_cols)

    # ── Step 2: 2-D embeddings ────────────────────────────────────────────────
    print("Step 2: Computing embeddings...")
    embeddings = {}
    for method in embedding.get("methods", ["UMAP"]):
        print(f"  {method}...")
        embedding_2d = compute_2d_embedding(
            latents,
            method=method,
            save_model_path=out_dir / f"{method.lower()}_model.pkl",
            **embedding.get(method.lower(), {}),
        )
        joblib.dump(embedding_2d, out_dir / f"{method.lower()}_2d.pkl")
        embeddings[method] = embedding_2d

        plot_2d_scatter(
            embedding_2d,
            labels=np.zeros(len(embedding_2d)),
            xlabel=f"{method}1", ylabel=f"{method}2",
            title=f"{method} Embedding",
            save_path=out_dir / f"{method.lower()}_scatter.png",
        )
    results["embeddings"] = embeddings

    # ── Step 3: Clustering ────────────────────────────────────────────────────
    print("Step 3: Clustering...")
    cluster_labels = {}

    kmeans_cfg = clustering.get("kmeans", {})
    if kmeans_cfg.get("enabled", False):
        n_clusters = kmeans_cfg["n_clusters"]
        print(f"  KMeans (k={n_clusters})...")
        _, labels = kmeans_cluster(latents, n_clusters, str(out_dir), "kmeans_model")
        cluster_labels["kmeans"] = labels
        for method, embedding_2d in embeddings.items():
            plot_2d_scatter(
                embedding_2d, labels=labels,
                xlabel=f"{method}1", ylabel=f"{method}2",
                title=f"{method} — KMeans (k={n_clusters})",
                save_path=out_dir / f"{method.lower()}_kmeans.png",
            )

    dbscan_cfg = clustering.get("dbscan", {})
    if dbscan_cfg.get("enabled", False):
        eps, min_samples = dbscan_cfg["eps"], dbscan_cfg["min_samples"]
        print(f"  DBSCAN (eps={eps}, min_samples={min_samples})...")
        _, labels = DBSCAN_cluster(latents, eps, min_samples, str(out_dir), "dbscan_model")
        cluster_labels["dbscan"] = labels
        for method, embedding_2d in embeddings.items():
            plot_2d_scatter(
                embedding_2d, labels=labels,
                xlabel=f"{method}1", ylabel=f"{method}2",
                title=f"{method} — DBSCAN",
                save_path=out_dir / f"{method.lower()}_dbscan.png",
            )

    results["cluster_labels"] = cluster_labels

    # ── Steps 4-8 require labels ──────────────────────────────────────────────
    if label_csv is None:
        print("Steps 4-8: Skipped (no label_csv provided).")
        results["merged_df"] = None
        return results

    # ── Step 4: Build merged label + latent DataFrame ─────────────────────────
    print("Step 4: Building merged label + latent DataFrame...")
    primary_method    = embedding.get("methods", ["UMAP"])[0]
    primary_embedding = embeddings[primary_method]
    merged_df = build_label_latent_df(
        label_csv, latents_df, primary_embedding, dim_method=primary_method,
    )
    merged_df.to_csv(out_dir / "merged_label_latent.csv", index=False)
    print(f"  Labeled rows: {merged_df['classification'].notna().sum()}")
    results["merged_df"] = merged_df

    # Resolve label orders — use config if provided, else infer from data
    label_orders   = label_orders or {}
    classif_order  = label_orders.get(
        "classification",
        sorted(merged_df["classification"].dropna().unique().tolist())
    )
    has_position   = "Position" in merged_df.columns
    position_order = label_orders.get(
        "position",
        sorted(merged_df["Position"].dropna().unique().tolist()) if has_position else []
    )

    # Overlay labels on embedding for each method
    for method, embedding_2d in embeddings.items():
        emb_df = merged_df.copy()
        emb_df[f"{method}_d0"] = embedding_2d[merged_df.index, 0]
        emb_df[f"{method}_d1"] = embedding_2d[merged_df.index, 1]
        plot_labels_on_embedding(
            emb_df,
            embedding_d0_col=f"{method}_d0",
            embedding_d1_col=f"{method}_d1",
            color_by_col="classification",
            label_order=classif_order,
            dim_method=method,
            save_path=out_dir / f"{method.lower()}_labels.png",
        )

    # ── Step 5: Latent dimension correlation ──────────────────────────────────
    print("Step 5: Latent dimension correlation...")
    corr_matrix = merged_df[latent_cols].corr()
    corr_matrix.to_csv(out_dir / "latent_correlation.csv")
    plot_latent_pairwise_correlation(
        merged_df[latent_cols],
        save_path=out_dir / "latent_correlation.png",
    )

    # ── Step 6: Latent dims by label ──────────────────────────────────────────
    print("Step 6: Latent dimensions by label...")
    boxplot_kind = clustering.get("boxplot_kind", "box")
    plot_latent_vs_label_boxplots(
        merged_df, merged_df,
        label_col="classification",
        label_order=classif_order,
        latent_cols=latent_cols,
        merge_on="unique_ID",
        kind=boxplot_kind,
        save_path=out_dir / "latent_by_classification_boxplot.png",
    )
    if position_order:
        plot_latent_vs_label_boxplots(
            merged_df, merged_df,
            label_col="Position",
            label_order=position_order,
            latent_cols=latent_cols,
            merge_on="unique_ID",
            kind=boxplot_kind,
            save_path=out_dir / "latent_by_position_boxplot.png",
        )

    # ── Step 7: Class distribution ────────────────────────────────────────────
    print("Step 7: Class distribution...")
    plot_class_distribution(
        merged_df, label_col="classification", label_order=classif_order,
        save_path=out_dir / "class_distribution.png",
    )
    if position_order:
        plot_class_distribution(
            merged_df, label_col="Position", label_order=position_order,
            save_path=out_dir / "position_distribution.png",
        )

    # ── Step 8: Crosstab heatmap ──────────────────────────────────────────────
    if position_order:
        print("Step 8: Crosstab heatmap (classification vs position)...")
        plot_crosstab_heatmap(
            merged_df,
            row_col="classification", col_col="Position",
            row_order=classif_order,  col_order=position_order,
            save_path=out_dir / "crosstab_classification_vs_position.png",
        )
    else:
        print("Step 8: Skipped (no Position column found).")

    # ── Step 9: Per-class latent dim statistics ───────────────────────────────
    print("Step 9: Per-class latent dim statistics...")
    stats_rows = []
    for label in classif_order:
        subset = merged_df[merged_df["classification"] == label][latent_cols]
        row = {"classification": label}
        for col in latent_cols:
            row[f"{col}_mean"] = subset[col].mean()
            row[f"{col}_std"]  = subset[col].std()
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows).set_index("classification")
    stats_df.to_csv(out_dir / "latent_stats_per_class.csv")

    # Heatmap of per-class latent means
    mean_cols = [f"{c}_mean" for c in latent_cols]
    mean_matrix = stats_df[mean_cols].copy()
    mean_matrix.columns = latent_cols
    fig, ax = plt.subplots(figsize=(len(latent_cols) * 1.2, len(classif_order) * 0.8 + 1))
    sns.heatmap(mean_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, linewidths=0.5)
    ax.set_title("Mean latent value per class")
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Classification")
    fig.tight_layout()
    fig.savefig(out_dir / "latent_mean_per_class_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Step 10: Reconstruction MSE per sample ────────────────────────────────
    print("Step 10: Reconstruction MSE per sample...")

    # Attach MSE to merged_df by positional index alignment
    mse_series = pd.Series(results["sample_mse"], name="recon_mse")
    merged_df["recon_mse"] = mse_series.reindex(merged_df.index).values
    merged_df.to_csv(out_dir / "merged_label_latent.csv", index=False)  # re-save with MSE column

    # MSE distribution overall
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(results["sample_mse"], bins=50, edgecolor="none", alpha=0.8)
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction MSE distribution (all samples)")
    fig.tight_layout()
    fig.savefig(out_dir / "recon_mse_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # MSE per class — violin plot
    mse_labeled = merged_df[merged_df["classification"].notna() & merged_df["recon_mse"].notna()]
    fig, ax = plt.subplots(figsize=(max(6, len(classif_order) * 1.5), 4))
    sns.violinplot(data=mse_labeled, x="classification", y="recon_mse",
                   order=classif_order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Reconstruction MSE by class")
    ax.set_xlabel("Classification")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(out_dir / "recon_mse_by_class.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDone. All outputs saved to: {out_dir}")
    return results
