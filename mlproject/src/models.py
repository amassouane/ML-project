from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_classification_pipelines(preprocess, random_state: int = 42):
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "DecisionTree": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", DecisionTreeClassifier(random_state=random_state, class_weight="balanced")),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", SVC(probability=True, random_state=random_state, class_weight="balanced")),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "AdaBoost": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", AdaBoostClassifier(random_state=random_state)),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def get_param_grids():
    return {
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
        },
        "DecisionTree": {
            "model__max_depth": [None, 3, 5, 7, 9],
            "model__min_samples_split": [2, 5, 10],
        },
        "SVM": {
            "model__C": [0.5, 1.0, 5.0],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
        "AdaBoost": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.5, 1.0],
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
    }
