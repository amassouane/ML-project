from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import DEFAULT_DATASET, FIGURES_DIR, METRICS_DIR, TARGET_COL
from src.evaluation import ensure_dir
from src.preprocessing import load_data


def count_outliers_iqr(df: pd.DataFrame, columns, factor: float = 1.5) -> pd.Series:
    numeric = df[columns]
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    counts = {}
    for col in columns:
        counts[col] = int(((numeric[col] < lower[col]) | (numeric[col] > upper[col])).sum())
    return pd.Series(counts)


def run(
    data_path: str | Path,
    target_col: str = TARGET_COL,
):
    figures_dir = ensure_dir(FIGURES_DIR)
    metrics_dir = ensure_dir(METRICS_DIR)

    df = load_data(data_path, target_col=target_col, clip_outliers=False)

    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values per Feature")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "missing_values.png", dpi=150)
    plt.close(fig)

    if target_col in df.columns:
        counts = df[target_col].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 4))
        counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_title("Target Distribution")
        ax.set_xlabel("Target")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1, str(v), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(figures_dir / "target_distribution.png", dpi=150)
        plt.close(fig)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if numeric_cols:
        df[numeric_cols].hist(figsize=(10, 8), bins=20)
        plt.tight_layout()
        plt.savefig(figures_dir / "numeric_histograms.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        melted = df[numeric_cols].melt(var_name="feature", value_name="value")
        sns.boxplot(data=melted, x="feature", y="value", ax=ax)
        ax.set_title("Boxplots of Numeric Features")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(figures_dir / "numeric_boxplots.png", dpi=150)
        plt.close(fig)

    if "age" in df.columns and "thalach" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df, x="age", y="thalach", hue=target_col, palette="coolwarm", ax=ax)
        ax.set_title("Age vs Max Heart Rate")
        fig.tight_layout()
        fig.savefig(figures_dir / "age_vs_thalach.png", dpi=150)
        plt.close(fig)

    if "chol" in df.columns and "trestbps" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df, x="chol", y="trestbps", hue=target_col, palette="coolwarm", ax=ax)
        ax.set_title("Cholesterol vs Resting BP")
        fig.tight_layout()
        fig.savefig(figures_dir / "chol_vs_trestbps.png", dpi=150)
        plt.close(fig)

    desc = df.describe(include="all").transpose()
    desc["missing"] = missing

    outlier_counts = pd.Series(dtype=int)
    if numeric_cols:
        outlier_counts = count_outliers_iqr(df, numeric_cols)
    desc["outliers_iqr"] = outlier_counts
    desc.to_csv(metrics_dir / "eda_summary.csv")

    return desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--target-col", default=TARGET_COL)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        target_col=args.target_col,
    )


if __name__ == "__main__":
    main()
