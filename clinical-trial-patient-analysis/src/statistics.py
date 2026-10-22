"""Cautious statistical summaries of the synthetic binary outcomes."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm


def compare_treatment_groups(data: pd.DataFrame) -> dict[str, object]:
    """Compare independent response proportions with a two-sided test."""
    groups = ["Treatment", "Placebo"]
    unknown = set(data["treatment_group"].dropna().unique()) - set(groups)
    if unknown:
        raise ValueError(f"Unexpected treatment_group values: {sorted(unknown)}.")
    summaries = {}
    for group in groups:
        values = data.loc[data["treatment_group"] == group, "treatment_response"]
        summaries[group] = {
            "sample_size": int(values.size),
            "responders": int(values.sum()),
            "non_responders": int(values.size - values.sum()),
            "response_rate": float(values.mean()) if values.size else float("nan"),
        }
    if any(item["sample_size"] == 0 for item in summaries.values()):
        raise ValueError("Both Treatment and Placebo groups must contain participants.")
    treatment_rate = summaries["Treatment"]["response_rate"]
    placebo_rate = summaries["Placebo"]["response_rate"]
    difference = treatment_rate - placebo_rate
    table = np.array(
        [
            [summaries[group]["responders"], summaries[group]["non_responders"]]
            for group in groups
        ]
    )
    expected = chi2_contingency(table, correction=False).expected_freq
    if (expected < 5).any():
        statistic, p_value = fisher_exact(table, alternative="two-sided")
        test_name = "Fisher's exact test (two-sided)"
    else:
        pooled_rate = (table[0, 0] + table[1, 0]) / table.sum()
        standard_error_pooled = np.sqrt(
            pooled_rate * (1 - pooled_rate) * (1 / summaries["Treatment"]["sample_size"] + 1 / summaries["Placebo"]["sample_size"])
        )
        statistic = float(difference / standard_error_pooled) if standard_error_pooled else 0.0
        p_value = float(2 * norm.sf(abs(statistic)))
        test_name = "Two-proportion z-test (two-sided)"

    n_t = summaries["Treatment"]["sample_size"]
    n_p = summaries["Placebo"]["sample_size"]
    standard_error = np.sqrt(
        treatment_rate * (1 - treatment_rate) / n_t
        + placebo_rate * (1 - placebo_rate) / n_p
    )
    ci = (float(difference - 1.96 * standard_error), float(difference + 1.96 * standard_error))
    return {
        "group_summary": summaries,
        "difference_treatment_minus_placebo": float(difference),
        "confidence_interval_95": ci,
        "test_name": test_name,
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "interpretation": _interpret(p_value),
        "limitations": (
            "These are simulated outcomes. Random assignment is part of the simulation; "
            "sampling variation remains, and neither statistical significance nor this "
            "confidence interval is clinical evidence."
        ),
    }


def _interpret(p_value: float) -> str:
    if p_value < 0.05:
        return "The simulated groups differ at the 0.05 level; this is not evidence about real patients."
    return "This simulation does not show a statistically detectable group difference at the 0.05 level."


def compare_cluster_response(
    data: pd.DataFrame, random_seed: int = 42, permutations: int = 1999
) -> dict[str, object]:
    """Compare response by cluster; use permutation inference for sparse tables."""
    if "cluster" not in data.columns:
        raise ValueError("Data must contain cluster assignments.")
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be a positive integer.")
    grouped = data.groupby("cluster")["treatment_response"].agg(
        sample_size="count", responders="sum", response_rate="mean"
    )
    table = np.column_stack(
        [grouped["responders"].to_numpy(), grouped["sample_size"].to_numpy() - grouped["responders"].to_numpy()]
    )
    chi_square_result = chi2_contingency(table, correction=False)
    statistic = chi_square_result.statistic
    expected = chi_square_result.expected_freq
    if (expected < 5).any():
        observed = float(statistic)
        outcomes = data["treatment_response"].to_numpy()
        cluster_labels = data["cluster"].to_numpy()
        rng = np.random.default_rng(random_seed)
        simulated = np.empty(permutations)
        for index in range(permutations):
            shuffled = rng.permutation(outcomes)
            shuffled_table = np.zeros_like(table)
            for row, cluster in enumerate(grouped.index):
                mask = cluster_labels == cluster
                shuffled_table[row, 0] = shuffled[mask].sum()
                shuffled_table[row, 1] = mask.sum() - shuffled_table[row, 0]
            simulated[index] = chi2_contingency(shuffled_table, correction=False)[0]
        p_value = float((1 + np.count_nonzero(simulated >= observed)) / (permutations + 1))
        test_name = f"Permutation chi-square test ({permutations} shuffles)"
    else:
        observed = float(statistic)
        p_value = float(chi_square_result.pvalue)
        test_name = "Chi-square test of independence"
    return {
        "cluster_summary": grouped.round(4),
        "test_name": test_name,
        "test_statistic": observed,
        "p_value": p_value,
        "interpretation": (
            "Cluster response rates are an exploratory comparison; clusters were formed "
            "without outcomes and do not define validated subgroups or treatment benefit."
        ),
    }
