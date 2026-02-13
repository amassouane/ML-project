from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE

from src.config import DEFAULT_DATASET, FIGURES_DIR, RANDOM_SEED, TARGET_COL
from src.evaluation import ensure_dir
from src.preprocessing import build_preprocessor, load_data


def run(
    data_path: str | Path,
    target_col: str = TARGET_COL,
    random_state: int = RANDOM_SEED,
):
    df = load_data(data_path, target_col=target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col] if target_col in df.columns else None

    figures_dir = ensure_dir(FIGURES_DIR)

    preprocess, _, _ = build_preprocessor(df, target_col=target_col)
    X_scaled = preprocess.fit_transform(X)

    pca = PCA(n_components=min(10, X_scaled.shape[1]), random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Explained Variance")
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_variance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    if y is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        fig.colorbar(scatter, ax=ax, label="Target")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    ax.set_title("PCA 2D Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_2d.png", dpi=150)
    plt.close(fig)

    tsne_2d = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
    X_tsne = tsne_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(5, 4))
    if y is not None:
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        fig.colorbar(scatter, ax=ax, label="Target")
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    ax.set_title("t-SNE 2D Projection")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    fig.savefig(figures_dir / "tsne_2d.png", dpi=150)
    plt.close(fig)

    tsne_3d = TSNE(n_components=3, random_state=random_state, init="pca", learning_rate="auto")
    X_tsne_3d = tsne_3d.fit_transform(X_scaled)
    fig_3d = px.scatter_3d(
        x=X_tsne_3d[:, 0],
        y=X_tsne_3d[:, 1],
        z=X_tsne_3d[:, 2],
        color=y if y is not None else None,
        title="t-SNE 3D Projection",
        opacity=0.7,
    )
    fig_3d.write_html(figures_dir / "tsne_3d.html")

    preprocess_nmf, _, _ = build_preprocessor(df, target_col=target_col, for_nmf=True)
    X_nmf_input = preprocess_nmf.fit_transform(X)
    nmf = NMF(n_components=5, init="nndsvda", random_state=random_state, max_iter=1000)
    X_nmf = nmf.fit_transform(X_nmf_input)

    fig, ax = plt.subplots(figsize=(5, 4))
    if y is not None:
        scatter = ax.scatter(X_nmf[:, 0], X_nmf[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        fig.colorbar(scatter, ax=ax, label="Target")
    else:
        ax.scatter(X_nmf[:, 0], X_nmf[:, 1], alpha=0.7)
    ax.set_title("NMF 2D Projection")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(figures_dir / "nmf_2d.png", dpi=150)
    plt.close(fig)

    return {
        "pca": X_pca,
        "tsne": X_tsne,
        "nmf": X_nmf,
    }


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
