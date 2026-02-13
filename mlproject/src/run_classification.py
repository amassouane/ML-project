from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.config import DEFAULT_DATASET, FIGURES_DIR, METRICS_DIR, RANDOM_SEED, TARGET_COL
from src.evaluation import (
    classification_metrics,
    ensure_dir,
    extract_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance,
    plot_roc_curves,
)
from src.models import build_classification_pipelines, get_param_grids
from src.preprocessing import build_preprocessor, get_feature_names, load_data, split_data


def run(
    data_path: str | Path,
    target_col: str = TARGET_COL,
    random_state: int = RANDOM_SEED,
    test_size: float = 0.2,
    cv: int = 5,
    scoring: str = "roc_auc",
    experiment_name: str = "heart-disease-classification",
    use_mlflow: bool = True,
):
    df = load_data(data_path, target_col=target_col)
    preprocess, _, _ = build_preprocessor(df, target_col=target_col)

    X_train, X_test, y_train, y_test = split_data(
        df, target_col=target_col, test_size=test_size, random_state=random_state
    )

    pipelines = build_classification_pipelines(preprocess, random_state=random_state)
    param_grids = get_param_grids()

    figures_dir = ensure_dir(FIGURES_DIR)
    metrics_dir = ensure_dir(METRICS_DIR)

    if use_mlflow:
        mlflow.set_experiment(experiment_name)

    results = []
    roc_curves = {}

    for name, pipeline in pipelines.items():
        grid = param_grids.get(name, {})
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        else:
            y_score = best_model.decision_function(X_test)

        metrics = classification_metrics(y_test, y_pred, y_score=y_score)
        results.append({"model": name, **metrics})
        roc_curves[name] = (np.array(y_test), np.array(y_score))

        cm_path = figures_dir / f"confusion_{name}.png"
        plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {name}", cm_path)

        feature_names = get_feature_names(best_model.named_steps["preprocess"])
        importances, names = extract_feature_importance(best_model, feature_names)
        fi_path = figures_dir / f"feature_importance_{name}.png"
        plot_feature_importance(importances, names, f"Feature Importance - {name}", fi_path)

        perm_path = None
        if name == "RandomForest":
            perm_path = figures_dir / "permutation_importance_RandomForest.png"
            plot_permutation_importance(
                best_model,
                X_test,
                y_test,
                feature_names,
                "Permutation Importance - RandomForest",
                perm_path,
                random_state=random_state,
            )

        if use_mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(search.best_params_)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.sklearn.log_model(best_model, "model")
                mlflow.log_artifact(str(cm_path))
                if fi_path.exists():
                    mlflow.log_artifact(str(fi_path))
                if perm_path and perm_path.exists():
                    mlflow.log_artifact(str(perm_path))

    roc_path = figures_dir / "roc_curves.png"
    plot_roc_curves(roc_curves, roc_path)
    if use_mlflow:
        with mlflow.start_run(run_name="roc_curves"):
            mlflow.log_artifact(str(roc_path))

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    results_path = metrics_dir / "classification_metrics.csv"
    results_df.to_csv(results_path, index=False)

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", default="roc_auc")
    parser.add_argument("--experiment-name", default="heart-disease-classification")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        target_col=args.target_col,
        random_state=args.random_state,
        test_size=args.test_size,
        cv=args.cv,
        scoring=args.scoring,
        experiment_name=args.experiment_name,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
