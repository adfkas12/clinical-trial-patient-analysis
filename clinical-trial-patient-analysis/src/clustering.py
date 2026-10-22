"""Baseline-only K-means patient stratification."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import CLUSTER_FEATURES, DEFAULT_N_CLUSTERS, DEFAULT_RANDOM_SEED


def cluster_patients(
    data: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    calculate_silhouette: bool = True,
) -> dict[str, object]:
    """Cluster participants on standardized baseline measurements only."""
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise ValueError("n_clusters must be an integer.")
    if n_clusters < 2 or n_clusters >= len(data):
        raise ValueError("n_clusters must be at least 2 and less than the number of rows.")
    missing = sorted(set(CLUSTER_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing clustering features: {', '.join(missing)}.")
    if data[CLUSTER_FEATURES].isna().any().any():
        raise ValueError("Clustering features cannot contain missing values.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[CLUSTER_FEATURES])
    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = model.fit_predict(scaled)
    assignments = data.copy()
    assignments["cluster"] = labels
    counts = assignments["cluster"].value_counts().sort_index().rename("patient_count")
    profiles = assignments.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
    response_rates = None
    if "treatment_response" in assignments.columns:
        response_rates = (
            assignments.groupby("cluster")["treatment_response"]
            .agg(responders="sum", response_rate="mean", patient_count="count")
            .round(4)
        )

    silhouettes: dict[int, float] = {}
    if calculate_silhouette and len(data) > n_clusters + 1:
        candidates = [k for k in (2, 3, 4, 5) if k < len(data) and k != n_clusters]
        for candidate in candidates:
            candidate_labels = KMeans(
                n_clusters=candidate, random_state=random_seed, n_init=10
            ).fit_predict(scaled)
            if len(np.unique(candidate_labels)) > 1:
                silhouettes[candidate] = float(silhouette_score(scaled, candidate_labels))
        silhouettes[n_clusters] = float(silhouette_score(scaled, labels))

    return {
        "assignments": assignments,
        "cluster_sizes": counts,
        "cluster_profiles": profiles,
        "response_rates": response_rates,
        "silhouette_scores": dict(sorted(silhouettes.items())),
    }
