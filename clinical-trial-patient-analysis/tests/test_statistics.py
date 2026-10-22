import pytest

from src.data_generator import generate_synthetic_data
from src.statistics import compare_cluster_response, compare_treatment_groups


def test_treatment_comparison_returns_real_group_summaries():
    result = compare_treatment_groups(generate_synthetic_data(200, 10))
    assert result["group_summary"]["Treatment"]["sample_size"] > 0
    assert result["group_summary"]["Placebo"]["sample_size"] > 0
    assert 0 <= result["p_value"] <= 1
    assert len(result["confidence_interval_95"]) == 2


def test_cluster_comparison_requires_cluster_assignments():
    with pytest.raises(ValueError, match="cluster assignments"):
        compare_cluster_response(generate_synthetic_data(40))
