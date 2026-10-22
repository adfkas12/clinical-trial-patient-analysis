import pandas as pd

from src.config import BASELINE_FEATURES
from src.data_generator import generate_synthetic_data
from src.preprocessing import build_preprocessor, select_baseline_features


def test_baseline_feature_selection_excludes_outcomes_and_assignment():
    data = generate_synthetic_data(100)
    features = select_baseline_features(data)
    assert list(features.columns) == BASELINE_FEATURES
    assert not {"patient_id", "treatment_response", "week12_score", "improvement", "treatment_group"} & set(features.columns)
    transformed = build_preprocessor().fit_transform(features)
    assert transformed.shape[0] == len(data)


def test_selector_requires_all_baseline_columns():
    import pytest

    with pytest.raises(ValueError, match="Missing baseline"):
        select_baseline_features(pd.DataFrame({"age": [30]}))
