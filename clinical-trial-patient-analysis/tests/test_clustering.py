import pytest

from src.clustering import cluster_patients
from src.data_generator import generate_synthetic_data


def test_clustering_returns_assignments_profiles_and_sizes():
    data = generate_synthetic_data(100, 4)
    result = cluster_patients(data, n_clusters=3, calculate_silhouette=False)
    assignments = result["assignments"]
    assert assignments["cluster"].nunique() == 3
    assert int(result["cluster_sizes"].sum()) == len(data)
    assert list(result["cluster_profiles"].columns) == ["age", "baseline_score", "biomarker_level"]


def test_clustering_rejects_invalid_cluster_count():
    with pytest.raises(ValueError, match="at least 2"):
        cluster_patients(generate_synthetic_data(30), n_clusters=1)
