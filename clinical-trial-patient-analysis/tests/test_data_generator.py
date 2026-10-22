import pandas as pd
import numpy as np
import pytest

from src.config import REQUIRED_COLUMNS
from src.data_generator import generate_synthetic_data
from src.data_loader import validate_dataset


def test_generator_schema_and_size():
    data = generate_synthetic_data(120, 7)
    assert len(data) == 120
    assert list(data.columns) == REQUIRED_COLUMNS
    assert data["patient_id"].is_unique
    assert set(data["treatment_group"]) == {"Treatment", "Placebo"}
    assert set(data["treatment_response"]) == {0, 1}


def test_generator_is_reproducible_and_improvement_is_derived():
    first = generate_synthetic_data(100, 12)
    second = generate_synthetic_data(100, 12)
    pd.testing.assert_frame_equal(first, second)
    assert np.allclose(
        first["improvement"],
        first["baseline_score"] - first["week12_score"],
        atol=0.001,
    )


def test_generator_rejects_invalid_count():
    with pytest.raises(ValueError, match="at least 20"):
        generate_synthetic_data(10)


def test_validator_detects_duplicate_ids():
    data = generate_synthetic_data(50)
    data.loc[1, "patient_id"] = data.loc[0, "patient_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(data)
