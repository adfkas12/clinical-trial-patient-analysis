"""Leakage-safe baseline Logistic Regression training and evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import BASELINE_FEATURES, DEFAULT_RANDOM_SEED
from src.preprocessing import build_preprocessor, select_baseline_features


@dataclass
class ModelResults:
    """Fitted pipeline and reproducible held-out evaluation outputs."""

    pipeline: Pipeline
    metrics: dict[str, float | None]
    confusion_matrix: np.ndarray
    classification_report: dict[str, object]
    coefficients: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_probability: np.ndarray


def train_and_evaluate(
    data: pd.DataFrame, random_seed: int = DEFAULT_RANDOM_SEED, test_size: float = 0.25
) -> ModelResults:
    """Split first, then fit preprocessing and Logistic Regression on training data."""
    if len(data) < 20:
        raise ValueError("At least 20 records are required for a train/test evaluation.")
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5.")
    if "treatment_response" not in data.columns:
        raise ValueError("Dataset must contain treatment_response.")
    y = data["treatment_response"]
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Both response classes need at least two records.")
    X = select_baseline_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_seed)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": auc,
    }
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": pipeline.named_steps["classifier"].coef_[0],
        }
    ).sort_values("coefficient", ascending=False, ignore_index=True)
    return ModelResults(
        pipeline=pipeline,
        metrics=metrics,
        confusion_matrix=confusion_matrix(y_test, predictions, labels=[0, 1]),
        classification_report=classification_report(
            y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
        coefficients=coefficients,
        y_test=y_test,
        y_pred=predictions,
        y_probability=probabilities,
    )
