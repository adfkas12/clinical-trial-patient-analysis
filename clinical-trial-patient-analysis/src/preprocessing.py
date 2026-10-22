"""Shared leakage-safe preprocessing helpers."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BASELINE_CATEGORICAL_FEATURES, BASELINE_NUMERIC_FEATURES, BASELINE_FEATURES


def select_baseline_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return only the four allowed baseline predictors, in stable order."""
    missing = sorted(set(BASELINE_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing baseline feature columns: {', '.join(missing)}.")
    return data.loc[:, BASELINE_FEATURES].copy()


def build_preprocessor() -> ColumnTransformer:
    """Build a transformer that scales numeric data and encodes sex."""
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, BASELINE_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, BASELINE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
