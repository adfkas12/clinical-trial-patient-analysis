"""Central project configuration and dataset definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "synthetic_clinical_trial.csv"
DEFAULT_N_PATIENTS = 800
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_CLUSTERS = 3
RESPONSE_THRESHOLD = 10.0
SEVERITY_THRESHOLDS = (35.0, 65.0)

REQUIRED_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "baseline_score",
    "biomarker_level",
    "disease_severity",
    "treatment_group",
    "week12_score",
    "treatment_response",
    "improvement",
]
NUMERIC_COLUMNS = [
    "age",
    "baseline_score",
    "biomarker_level",
    "week12_score",
    "treatment_response",
    "improvement",
]
BASELINE_NUMERIC_FEATURES = ["age", "baseline_score", "biomarker_level"]
BASELINE_CATEGORICAL_FEATURES = ["sex"]
BASELINE_FEATURES = BASELINE_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES
CLUSTER_FEATURES = ["age", "baseline_score", "biomarker_level"]
TREATMENT_GROUPS = ["Treatment", "Placebo"]
SEX_CATEGORIES = ["Female", "Male"]
