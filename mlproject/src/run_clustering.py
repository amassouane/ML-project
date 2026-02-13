from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import DEFAULT_DATASET, FIGURES_DIR, METRICS_DIR, RANDOM_SEED, TARGET_COL
from src.evaluation import ensure_dir
from src.preprocessing import build_preprocessor, load_data


def run(
    data_path: str | Path,
    target_col: str = TARGET_COL,
    random_state: int = RANDOM_SEED,
):
    df = load_data(data_path, target_col=target_col)
    X = df.drop(columns=[target_col])

    figures_dir = ensure_dir(FIGURES_DIR)
    metrics_dir = ensure_dir(METRICS_DIR)

    preprocess, _, _ = build_preprocessor(df, target_col=target_col)
    X_scaled = preprocess.fit_transform(X)

    k_values = list(range(2, 11))
    inertias = []
    silhouettes = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(k_values, inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_elbow.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(k_values, silhouettes, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("K-Means Silhouette")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_silhouette.png", dpi=150)
    plt.close(fig)

    best_k = k_values[int(np.argmax(silhouettes))]
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    agglom = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agglom_labels = agglom.fit_predict(X_scaled)

    eps_candidates = [0.3, 0.5, 0.7, 1.0]
    best_dbscan = None
    best_dbscan_score = -1
    for eps in eps_candidates:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        if len(set(labels)) > 1 and (labels >= 0).any():
            score = silhouette_score(X_scaled, labels)
            if score > best_dbscan_score:
                best_dbscan_score = score
                best_dbscan = dbscan

    dbscan_labels = None
    if best_dbscan is not None:
        dbscan_labels = best_dbscan.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)

    def plot_clusters(labels, title, filename):
        fig, ax = plt.subplots(figsize=(5, 4))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=ax)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=150)
        plt.close(fig)

    plot_clusters(kmeans_labels, f"K-Means Clusters (k={best_k})", "kmeans_clusters.png")
    plot_clusters(agglom_labels, "Agglomerative Clusters", "agglomerative_clusters.png")
    if dbscan_labels is not None:
        plot_clusters(dbscan_labels, "DBSCAN Clusters", "dbscan_clusters.png")

    linked = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(6, 4))
    dendrogram(linked, ax=ax, truncate_mode="lastp", p=12)
    ax.set_title("Agglomerative Clustering Dendrogram")
    fig.tight_layout()
    fig.savefig(figures_dir / "dendrogram.png", dpi=150)
    plt.close(fig)

    records = []
    records.append(
        {
            "model": "kmeans",
            "k": best_k,
            "silhouette": silhouette_score(X_scaled, kmeans_labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, kmeans_labels),
            "inertia": kmeans.inertia_,
        }
    )
    records.append(
        {
            "model": "agglomerative",
            "k": best_k,
            "silhouette": silhouette_score(X_scaled, agglom_labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, agglom_labels),
            "inertia": None,
        }
    )
    if dbscan_labels is not None:
        records.append(
            {
                "model": "dbscan",
                "k": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                "silhouette": silhouette_score(X_scaled, dbscan_labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, dbscan_labels),
                "inertia": None,
            }
        )

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(metrics_dir / "clustering_metrics.csv", index=False)

    return metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        target_col=args.target_col,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
