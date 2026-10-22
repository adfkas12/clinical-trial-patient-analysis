"""Matplotlib/Seaborn plots used by the Streamlit dashboard."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _finish(fig: plt.Figure) -> plt.Figure:
    fig.tight_layout()
    return fig


def distribution_plot(data: pd.DataFrame, column: str, title: str, xlabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=data, x=column, ax=ax, color="#3973ac")
    ax.set(title=title, xlabel=xlabel, ylabel="Participants")
    return _finish(fig)


def numeric_distribution(data: pd.DataFrame, column: str, title: str, xlabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=data, x=column, kde=True, ax=ax, color="#3973ac")
    ax.set(title=title, xlabel=xlabel, ylabel="Participants")
    return _finish(fig)


def treatment_response_plot(data: pd.DataFrame) -> plt.Figure:
    rates = data.groupby("treatment_group")["treatment_response"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=rates, x="treatment_group", y="treatment_response", ax=ax, color="#3973ac")
    ax.set(title="Simulated response rate by assigned group", xlabel="Group", ylabel="Response rate")
    ax.set_ylim(0, 1)
    return _finish(fig)


def cluster_scatter_plot(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=data, x="baseline_score", y="biomarker_level", hue="cluster",
        palette="deep", alpha=0.75, ax=ax,
    )
    ax.set(title="Baseline profiles by K-means cluster", xlabel="Baseline score", ylabel="Biomarker level")
    return _finish(fig)


def cluster_response_plot(response_summary: pd.DataFrame) -> plt.Figure:
    plot_data = response_summary.reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=plot_data, x="cluster", y="response_rate", ax=ax, color="#3973ac")
    ax.set(title="Exploratory response rate by cluster", xlabel="Cluster ID (arbitrary)", ylabel="Response rate")
    ax.set_ylim(0, 1)
    return _finish(fig)


def confusion_matrix_plot(matrix: object) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Non-responder", "Responder"],
        yticklabels=["Non-responder", "Responder"],
    )
    ax.set(title="Held-out test confusion matrix", xlabel="Predicted", ylabel="Actual")
    return _finish(fig)


def coefficient_plot(coefficients: pd.DataFrame) -> plt.Figure:
    ordered = coefficients.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(ordered))))
    colors = ["#3973ac" if value >= 0 else "#c44e52" for value in ordered["coefficient"]]
    ax.barh(ordered["feature"], ordered["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Logistic Regression coefficients", xlabel="Coefficient (log-odds scale)", ylabel="Feature")
    return _finish(fig)
