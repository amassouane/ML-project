from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

DEFAULT_CATEGORICAL = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clip_outliers_iqr(
    df: pd.DataFrame,
    columns: Iterable[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    clipped = df.copy()
    numeric = clipped[columns]
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    for col in columns:
        clipped[col] = numeric[col].clip(lower=lower[col], upper=upper[col])
    return clipped


def load_data(
    path: str | Path,
    target_col: str = "target",
    drop_duplicates: bool = True,
    clip_outliers: bool = True,
    outlier_factor: float = 1.5,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_columns(df)

    if target_col not in df.columns and "num" in df.columns:
        df = df.rename(columns={"num": target_col})

    df = df.replace("?", np.nan)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() > 0.8:
                df[col] = coerced

    if target_col in df.columns and df[target_col].nunique() > 2:
        df[target_col] = (df[target_col] > 0).astype(int)

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    if clip_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if numeric_cols:
            df = clip_outliers_iqr(df, numeric_cols, factor=outlier_factor)

    return df


def get_default_categorical_cols(df: pd.DataFrame, target_col: str = "target") -> List[str]:
    cols = [c for c in DEFAULT_CATEGORICAL if c in df.columns]
    extra = [c for c in df.columns if c != target_col and df[c].dtype == "object"]
    return sorted(set(cols + extra))


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str = "target",
    categorical_cols: List[str] | None = None,
    scale_numeric: bool = True,
    for_nmf: bool = False,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    if categorical_cols is None:
        categorical_cols = get_default_categorical_cols(df, target_col)

    numeric_cols = [c for c in df.columns if c != target_col and c not in categorical_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        scaler = MinMaxScaler() if for_nmf else StandardScaler()
        num_steps.append(("scaler", scaler))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocess, numeric_cols, categorical_cols


def split_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_feature_names(preprocess: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []
    for name, transformer, cols in preprocess.transformers_:
        if name == "remainder":
            continue
        if isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]
        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out(cols)
                feature_names.extend(list(names))
            except TypeError:
                feature_names.extend(list(cols))
        else:
            feature_names.extend(list(cols))
    return feature_names
