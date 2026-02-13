from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str | Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], save_path: str | Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def extract_feature_importance(model, feature_names: Iterable[str]):
    estimator = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        return np.array(importances), list(feature_names)

    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
        return np.abs(coef), list(feature_names)

    return None, None


def plot_feature_importance(importances, feature_names, title: str, save_path: str | Path, top_n: int = 20):
    if importances is None or feature_names is None:
        return
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(feature_names)[idx], np.array(importances)[idx])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_permutation_importance(
    model,
    X,
    y,
    feature_names,
    title: str,
    save_path: str | Path,
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 20,
):
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    importances = result.importances_mean
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(feature_names)[idx], np.array(importances)[idx])
    ax.set_title(title)
    ax.set_xlabel("Permutation Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
