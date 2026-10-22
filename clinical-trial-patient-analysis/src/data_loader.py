"""Loading and strict validation for the synthetic dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    RESPONSE_THRESHOLD,
    SEVERITY_THRESHOLDS,
    SEX_CATEGORIES,
    TREATMENT_GROUPS,
)


def validate_dataset(data: pd.DataFrame) -> dict[str, object]:
    """Validate schema, types, ranges, labels, and derived values."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}.")
    if data.empty:
        raise ValueError("Dataset contains no participant records.")
    missing_values = int(data[REQUIRED_COLUMNS].isna().sum().sum())
    if missing_values:
        raise ValueError(f"Dataset contains {missing_values} missing values in required columns.")
    if data["patient_id"].duplicated().any():
        raise ValueError("patient_id values must be unique.")

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{column} must contain numeric values.")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values.")

    if not data["age"].between(18, 85).all() or not np.equal(data["age"], np.floor(data["age"])).all():
        raise ValueError("age values must be integers between 18 and 85.")
    for column in ("baseline_score", "biomarker_level", "week12_score"):
        if not data[column].between(0, 100).all():
            raise ValueError(f"{column} values must be between 0 and 100.")
    if not data["sex"].isin(SEX_CATEGORIES).all():
        raise ValueError(f"sex must contain only: {', '.join(SEX_CATEGORIES)}.")
    if not data["treatment_group"].isin(TREATMENT_GROUPS).all():
        raise ValueError(f"treatment_group must contain only: {', '.join(TREATMENT_GROUPS)}.")
    if not data["treatment_response"].isin([0, 1]).all():
        raise ValueError("treatment_response must contain only 0 and 1.")

    expected_improvement = data["baseline_score"] - data["week12_score"]
    if not np.allclose(data["improvement"], expected_improvement, atol=0.011, rtol=0):
        raise ValueError("improvement must equal baseline_score minus week12_score.")
    expected_response = (data["improvement"] >= RESPONSE_THRESHOLD).astype(int)
    if not np.array_equal(data["treatment_response"].to_numpy(), expected_response.to_numpy()):
        raise ValueError(f"treatment_response must reflect the {RESPONSE_THRESHOLD:g}-point response rule.")

    expected_severity = pd.Series(
        np.select(
            [
                data["baseline_score"] < SEVERITY_THRESHOLDS[0],
                data["baseline_score"] < SEVERITY_THRESHOLDS[1],
            ],
            ["Mild", "Moderate"],
            default="Severe",
        ),
        index=data.index,
    )
    if not data["disease_severity"].eq(expected_severity).all():
        raise ValueError("disease_severity does not match the documented baseline-score thresholds.")
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_values": missing_values,
        "unique_patients": int(data["patient_id"].nunique()),
        "response_classes": sorted(data["treatment_response"].unique().tolist()),
    }


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a CSV file and return it with validation information."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. From the project folder, run: python generate_data.py"
        )
    data = pd.read_csv(path)
    return data, validate_dataset(data)
