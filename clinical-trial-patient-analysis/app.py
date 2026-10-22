"""Interactive dashboard for simulated treatment-response analysis."""

import streamlit as st

from src.clustering import cluster_patients
from src.config import DATA_PATH, DEFAULT_N_CLUSTERS, DEFAULT_RANDOM_SEED
from src.data_loader import load_dataset
from src.modeling import train_and_evaluate
from src.statistics import compare_cluster_response, compare_treatment_groups
from src.visualization import (
    cluster_response_plot,
    cluster_scatter_plot,
    coefficient_plot,
    confusion_matrix_plot,
    distribution_plot,
    numeric_distribution,
    treatment_response_plot,
)

DISCLAIMER = (
    "Synthetic data for educational and research-method development purposes only. "
    "This application is not intended for clinical decision-making."
)

st.set_page_config(page_title="Clinical Trial Patient Analysis", page_icon="🧪", layout="wide")


@st.cache_data
def load_data(path: str, modified_at: int):
    return load_dataset(path)


@st.cache_data
def get_clusters(data, cluster_count: int):
    return cluster_patients(data, n_clusters=cluster_count)


@st.cache_resource
def get_model(data):
    return train_and_evaluate(data)


st.sidebar.title("Patient Stratification")
st.sidebar.caption(DISCLAIMER)
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Explorer", "Patient Stratification", "Treatment Response Analysis", "Machine Learning", "Methodology and Limitations"],
)
cluster_count = st.sidebar.slider("Number of clusters", min_value=2, max_value=6, value=DEFAULT_N_CLUSTERS)
st.sidebar.caption(f"Default reproducibility seed: {DEFAULT_RANDOM_SEED}")
st.title("Clinical Trial Patient Stratification & Treatment Response Analysis")
st.warning(DISCLAIMER)

try:
    data, validation = load_data(
        str(DATA_PATH), DATA_PATH.stat().st_mtime_ns if DATA_PATH.exists() else 0
    )
except (FileNotFoundError, ValueError, TypeError) as error:
    st.error(str(error))
    st.stop()

if section == "Overview":
    st.header("Overview")
    st.write(
        "This educational project explores a fully simulated clinical-trial dataset: "
        "it groups participants by baseline measurements, compares simulated response "
        "rates, and evaluates a baseline-only Logistic Regression model."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Synthetic participants", f"{validation['rows']:,}")
    col2.metric("Treatment groups", data["treatment_group"].nunique())
    col3.metric("Overall response rate", f"{data['treatment_response'].mean():.1%}")
    col4.metric("Missing values", validation["missing_values"])
    st.caption("The generator assigns treatment independently of baseline characteristics.")
    st.pyplot(treatment_response_plot(data), clear_figure=True)

elif section == "Data Explorer":
    st.header("Data Explorer")
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    left, right = st.columns(2)
    left.subheader("Data types")
    left.dataframe(data.dtypes.astype(str).rename("dtype"), use_container_width=True)
    right.subheader("Missing-value summary")
    right.dataframe(data.isna().sum().rename("missing"), use_container_width=True)
    st.subheader("Descriptive statistics")
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
    a, b = st.columns(2)
    a.pyplot(distribution_plot(data, "treatment_group", "Treatment assignment", "Group"), clear_figure=True)
    b.pyplot(distribution_plot(data, "treatment_response", "Response label (0/1)", "Response"), clear_figure=True)
    a.pyplot(numeric_distribution(data, "baseline_score", "Baseline severity score", "Score"), clear_figure=True)
    b.pyplot(numeric_distribution(data, "biomarker_level", "Simulated biomarker distribution", "Biomarker level"), clear_figure=True)

elif section == "Patient Stratification":
    st.header("Patient Stratification")
    st.info("Clusters use standardized age, baseline score, and biomarker level only. Outcomes and treatment assignments do not affect membership.")
    clusters = get_clusters(data, cluster_count)
    st.pyplot(cluster_scatter_plot(clusters["assignments"]), clear_figure=True)
    left, right = st.columns(2)
    left.subheader("Cluster sizes")
    left.dataframe(clusters["cluster_sizes"], use_container_width=True)
    right.subheader("Baseline feature profiles (means)")
    right.dataframe(clusters["cluster_profiles"], use_container_width=True)
    if clusters["response_rates"] is not None:
        st.subheader("Exploratory response summaries")
        st.dataframe(clusters["response_rates"], use_container_width=True)
        st.pyplot(cluster_response_plot(clusters["response_rates"]), clear_figure=True)
    if clusters["silhouette_scores"]:
        st.subheader("Silhouette scores")
        st.dataframe(
            [{"clusters (k)": key, "silhouette score": value} for key, value in clusters["silhouette_scores"].items()],
            use_container_width=True,
        )
    st.caption("Cluster IDs are arbitrary mathematical groupings, not clinically validated subgroups.")

elif section == "Treatment Response Analysis":
    st.header("Treatment Response Analysis")
    overall = compare_treatment_groups(data)
    left, right = st.columns(2)
    left.pyplot(treatment_response_plot(data), clear_figure=True)
    right.metric("Difference (Treatment − Placebo)", f"{overall['difference_treatment_minus_placebo']:.1%}")
    right.write(f"95% CI: {overall['confidence_interval_95'][0]:.1%} to {overall['confidence_interval_95'][1]:.1%}")
    right.write(f"**Test:** {overall['test_name']}")
    right.write(f"**p-value:** {overall['p_value']:.4f}")
    right.write(overall["interpretation"])
    st.dataframe(overall["group_summary"], use_container_width=True)
    clusters = get_clusters(data, cluster_count)
    cluster_stats = compare_cluster_response(clusters["assignments"])
    st.subheader("Exploratory response comparison across clusters")
    st.dataframe(cluster_stats["cluster_summary"], use_container_width=True)
    st.write(f"**{cluster_stats['test_name']}** — statistic {cluster_stats['test_statistic']:.3f}, p = {cluster_stats['p_value']:.4f}")
    st.caption(cluster_stats["interpretation"])
    st.caption(overall["limitations"])

elif section == "Machine Learning":
    st.header("Baseline-only response prediction")
    st.info("The model uses age, sex, baseline score, and biomarker level. It excludes treatment assignment and all post-treatment outcomes.")
    try:
        result = get_model(data)
    except ValueError as error:
        st.error(f"Unable to train model: {error}")
        st.stop()
    columns = st.columns(5)
    for column, metric in zip(columns, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        value = result.metrics[metric]
        column.metric(metric.replace("_", " ").title(), "N/A" if value is None else f"{value:.3f}")
    left, right = st.columns(2)
    left.pyplot(confusion_matrix_plot(result.confusion_matrix), clear_figure=True)
    right.pyplot(coefficient_plot(result.coefficients), clear_figure=True)
    st.caption("Coefficient direction reflects association in this fitted model, not causality. Held-out performance describes only data simulated by this generator.")

elif section == "Methodology and Limitations":
    st.header("Methodology and Limitations")
    st.markdown(
        """
        - **Data:** Generated locally from a fixed random seed; no real patient records are used.
        - **Outcome:** A responder has a simulated reduction of at least 10 score points from baseline to week 12. This is an illustrative rule, not a validated clinical threshold.
        - **Severity:** Mild is below 35, Moderate is 35 to below 65, and Severe is 65 or higher.
        - **Stratification:** K-means standardizes and groups age, baseline score, and biomarker. It identifies mathematical similarity, not clinically validated subgroups.
        - **Treatment comparison:** A two-sided proportion test (or Fisher's exact test for sparse counts) compares the simulated randomized groups. Cluster response rates are exploratory.
        - **Prediction:** Logistic Regression estimates response association from baseline-only variables. The preprocessing pipeline is fitted on training data only, then evaluated on an untouched stratified test set.
        - **No benefit prediction:** Predicting response is different from estimating which treatment would benefit a particular person; this model does not estimate individualized treatment effects.
        - **Limitations:** Simulated patterns and test metrics are not clinical evidence, causal findings, or a measure of real-world model performance. Statistical significance is not clinical significance.
        """
    )
