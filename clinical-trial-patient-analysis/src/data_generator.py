"""Deterministic generation of explicitly simulated participant records."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    RESPONSE_THRESHOLD,
    SEVERITY_THRESHOLDS,
    SEX_CATEGORIES,
    TREATMENT_GROUPS,
)
from src.data_loader import validate_dataset


def _severity_from_score(scores: np.ndarray) -> np.ndarray:
    return np.select(
        [scores < SEVERITY_THRESHOLDS[0], scores < SEVERITY_THRESHOLDS[1]],
        ["Mild", "Moderate"],
        default="Severe",
    )


def generate_synthetic_data(
    n_patients: int = 800, random_seed: int = 42
) -> pd.DataFrame:
    """Create reproducible artificial records; no real patient data is used.

    Treatment is assigned independently of baseline variables. Improvement
    includes a modest assumed treatment effect and individual random variation.
    A response is an illustrative reduction of at least ten score points.
    """
    if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients < 20:
        raise ValueError("n_patients must be an integer of at least 20.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer.")

    rng = np.random.default_rng(random_seed)
    age = rng.integers(18, 86, size=n_patients)
    sex = rng.choice(SEX_CATEGORIES, size=n_patients)
    baseline_score = np.clip(rng.normal(55, 17, size=n_patients), 0, 100).round(2)
    biomarker_level = np.clip(rng.gamma(shape=3.0, scale=12.0, size=n_patients), 0, 100).round(2)
    treatment_group = rng.choice(TREATMENT_GROUPS, size=n_patients)
    treated = treatment_group == "Treatment"

    # The 4-point assumed effect is deliberately modest relative to outcome noise.
    expected_improvement = 6.0 + treated.astype(float) * 4.0
    expected_improvement += 0.035 * (baseline_score - 50.0)
    simulated_improvement = expected_improvement + rng.normal(0, 8.5, size=n_patients)
    week12_score = np.clip(baseline_score - simulated_improvement, 0, 100).round(2)
    improvement = (baseline_score - week12_score).round(2)
    response = (improvement >= RESPONSE_THRESHOLD).astype(int)

    data = pd.DataFrame(
        {
            "patient_id": [f"P{index:04d}" for index in range(1, n_patients + 1)],
            "age": age,
            "sex": sex,
            "baseline_score": baseline_score,
            "biomarker_level": biomarker_level,
            "disease_severity": _severity_from_score(baseline_score),
            "treatment_group": treatment_group,
            "week12_score": week12_score,
            "treatment_response": response,
            "improvement": improvement,
        }
    )
    validate_dataset(data)
    if data["treatment_response"].nunique() != 2:
        raise ValueError("Generated data must contain both response classes.")
    return data


def save_synthetic_data(data: pd.DataFrame, output_path: Path) -> Path:
    """Validate and save the generated dataset, creating its parent directory."""
    validate_dataset(data)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return destination
