import pytest

from src.data_generator import generate_synthetic_data
from src.modeling import train_and_evaluate


def test_model_trains_and_reports_held_out_metrics():
    result = train_and_evaluate(generate_synthetic_data(300, 11))
    assert 0 <= result.metrics["accuracy"] <= 1
    assert result.metrics["roc_auc"] is not None
    assert result.confusion_matrix.shape == (2, 2)
    assert set(result.coefficients["feature"]) == {"age", "baseline_score", "biomarker_level", "sex_Male"}
    assert len(result.y_test) == 75


def test_model_rejects_small_dataset():
    with pytest.raises(ValueError, match="At least 20"):
        train_and_evaluate(generate_synthetic_data(20).iloc[:19])
