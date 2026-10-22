# Clinical Trial Patient Stratification & Treatment Response Analysis

A local, beginner-friendly demonstration of synthetic clinical-trial data generation, baseline patient stratification, response-rate comparisons, and baseline-only Logistic Regression.

> **Synthetic data for educational and research-method development purposes only. This application is not intended for clinical decision-making.**

No real patient data is used. The generated measurements, relationships, subgroup comparisons, and model metrics are simulations—not clinical evidence.

## What the project does

1. Creates 800 reproducible synthetic participant records.
2. Validates the generated CSV and derived labels.
3. Uses K-means to group people by baseline age, score, and biomarker.
4. Compares simulated response rates by randomized treatment assignment and by exploratory cluster.
5. Trains and evaluates one Logistic Regression classifier from baseline characteristics.
6. Presents the analyses in a local Streamlit dashboard.

## Project structure

| File or folder | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard and navigation between analyses. |
| `generate_data.py` | Command-line entry point for generating the CSV. |
| `requirements.txt` | Runtime and test dependencies. |
| `src/config.py` | Paths, feature lists, default seed, response and severity thresholds. |
| `src/data_generator.py` | Deterministic synthetic record generation and CSV saving. |
| `src/data_loader.py` | CSV loading and validation of columns, types, ranges, and derived labels. |
| `src/preprocessing.py` | Reusable numeric scaling, categorical encoding, and baseline-feature selection. |
| `src/clustering.py` | Standardized, baseline-only K-means clustering and optional silhouette scores. |
| `src/statistics.py` | Treatment-group and exploratory cluster response comparisons. |
| `src/modeling.py` | Leakage-safe train/test split, Logistic Regression, metrics, and coefficients. |
| `src/visualization.py` | Reusable charts used by the dashboard. |
| `tests/` | Offline pytest checks for data, preprocessing, clustering, and modelling. |
| `data/` | Generated CSV output; the CSV is ignored by Git. |
| `models/` | Reserved for optional future model artifacts; no model is persisted. |
| `.gitignore` | Excludes virtual environments, caches, generated data, and model artifacts. |

## Synthetic data assumptions

The generator creates a configurable number of participants (800 by default) using NumPy's explicit random generator and seed 42 by default. Re-running it with the same seed and participant count produces the same data.

- `patient_id`: unique synthetic ID, never a model feature.
- `age`: integer from 18 through 85.
- `sex`: randomly simulated Female/Male category. This simplified variable is for demonstration and is not a complete representation of sex or gender.
- `baseline_score`: clipped normal sample on a 0–100 hypothetical severity scale; higher is worse.
- `biomarker_level`: clipped gamma-distributed simulated value on a 0–100 scale.
- `disease_severity`: derived from baseline score: Mild below 35, Moderate from 35 to below 65, Severe at least 65.
- `treatment_group`: randomized Treatment/Placebo assignment, generated independently of baseline characteristics.
- `week12_score`: baseline score less simulated improvement, clipped to 0–100. Expected improvement includes 6 points in Placebo, an additional illustrative 4 points in Treatment, a small baseline-score component, and normally distributed individual variation (standard deviation 8.5).
- `improvement`: baseline score minus week-12 score, after score clipping.
- `treatment_response`: 1 if improvement is at least 10 points, otherwise 0. This is an illustrative modelling threshold and is not medically validated.

The response is not a deterministic function of one baseline feature. Individual variation creates overlap between groups. The generator checks that both outcome classes appear and rejects invalid sample sizes below 20.

## Windows setup and execution

Open PowerShell in the project folder (the folder containing `app.py`) or open that folder in VS Code:

```powershell
cd "C:\Users\aditi\OneDrive\Desktop\Clinical Trial Patient Stratification and Treatement Response Analysis\clinical-trial-patient-analysis"
code .
```

Check Python (3.11 or compatible; this workspace was checked with Python 3.13.3):

```powershell
python --version
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, either allow scripts for the current user as appropriate for your machine, or use the environment's executable directly: `.venv\Scripts\python.exe`.

Install dependencies, generate data, and run tests:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
python -m pytest -q
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

The app opens locally in a browser. If the CSV is absent, the dashboard explains that `python generate_data.py` will create it.

To generate a different reproducible sample, call the generator from Python:

```powershell
python -c "from src.data_generator import generate_synthetic_data, save_synthetic_data; from src.config import DATA_PATH; save_synthetic_data(generate_synthetic_data(1000, 17), DATA_PATH)"
```

The CSV at `data\synthetic_clinical_trial.csv` is generated output and is intentionally not committed.

## Analysis and interpretation

### Stratification

K-means uses standardized `age`, `baseline_score`, and `biomarker_level`. It does not use treatment assignment or any response/post-treatment column. Cluster IDs are arbitrary; inspect their feature profiles rather than treating IDs as meaningful clinical labels. Silhouette scores are an optional geometric diagnostic, not proof that a clinically meaningful grouping exists.

### Statistical comparisons

The overall treatment comparison uses a two-sided two-proportion z-test when expected counts permit, or Fisher's exact test for sparse 2×2 counts. It reports a 95% Wald confidence interval for the difference in response rates. Cluster response comparisons use a chi-square test when expected counts are adequate and a reproducible label-permutation chi-square test if they are sparse. These cluster results are exploratory. Because all records are simulated, neither low p-values nor differences imply a real treatment effect or patient subgroup.

### Prediction

The primary Logistic Regression uses only `age`, `sex`, `baseline_score`, and `biomarker_level`. Treatment group is intentionally excluded: the model estimates response association under the simulated data mechanism and does not estimate individualized benefit from choosing one treatment over another. Numeric scaling and categorical encoding are inside a Scikit-learn pipeline. Data is split before fitting that pipeline, so test records do not influence preprocessing.

Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and coefficients are calculated from actual held-out predictions. Coefficients are on the transformed feature/log-odds scale; their direction is associational, not causal. Test performance only describes held-out samples produced by the same synthetic mechanism and does not establish real-world performance.

## Limitations

- The dataset is synthetic and generated under simple assumptions.
- Simulated treatment assignment is randomized by design; real trials have additional design, conduct, and analysis considerations.
- Scores, biomarkers, response thresholds, and treatment effects are illustrative.
- The model predicts a simulated outcome, not counterfactual treatment benefit.
- Clusters are mathematical partitions, not validated clinical subgroups.
- Subgroup comparisons have multiple sources of uncertainty and are not confirmatory.
- Statistical significance does not equal clinical significance.
- This project must not be used for clinical decisions.

## Interview Preparation

### Core ideas

1. **Problem:** Demonstrate a complete, explainable analysis workflow for stratification and outcome prediction using artificial trial-like records.
2. **Why synthetic data:** It avoids exposing personal health information and makes the assumptions reproducible, while explicitly limiting claims to a coding demonstration.
3. **Generation:** A seeded NumPy generator samples demographics and baseline measures, independently randomizes treatment, simulates noisy improvement with a modest assumed treatment-group difference, clips scores to their valid range, and derives labels.
4. **Response definition:** A participant is a responder when baseline score minus week-12 score is at least 10. The threshold is illustrative, not medically established.
5. **Why Pandas:** DataFrames make tabular generation, validation, grouping, and descriptive summaries readable.
6. **Why scale for K-means:** K-means uses distance. Without scaling, a feature's numeric units or range could dominate the distance.
7. **How K-means works:** It assigns points to the nearest centroid, recomputes each centroid as its group's mean, and iterates to reduce within-cluster squared distances.
8. **Why response is excluded from clustering:** Otherwise the known outcome would define the clusters, making later outcome comparisons circular and invalid as an exploratory check.
9. **Why Logistic Regression:** It is a fast, interpretable binary-classification baseline that outputs probabilities and understandable coefficient directions.
10. **Confusion matrix:** Rows are actual classes and columns predicted classes; it shows correct classifications and the two kinds of errors.
11. **Metrics:** Accuracy is the fraction correct; precision is the fraction of predicted responders who responded; recall is the fraction of responders found; F1 is the harmonic mean of precision and recall.
12. **ROC-AUC:** It summarizes how well the model ranks positive examples above negatives across classification thresholds; 0.5 is chance-level ranking in the usual binary setting.
13. **Hypothesis testing:** It assesses whether observed group differences are surprising under a stated no-difference model, subject to assumptions.
14. **P-value:** Under the null and assumptions, it is the probability of a statistic at least as extreme as observed. It is not the probability the null is true and not a measure of clinical importance.
15. **Leakage prevention:** Split first; fit the imputation/encoding/scaling and model only on training data; evaluate once on the held-out set.
16. **Why no post-treatment inputs:** Week-12 score and improvement occur after treatment and partly define the target; using them for baseline prediction leaks outcome information unavailable at prediction time.
17. **Response versus treatment benefit:** Response prediction estimates outcome under the observed simulated setup. Benefit prediction needs a comparison of potential outcomes under alternative assignments for the same participant, which this model does not provide.
18. **What it cannot establish:** It cannot establish clinical efficacy, validated biomarkers, real patient subgroups, clinical utility, causal effect sizes, or real-world model performance.
19. **Possible improvements:** Add carefully documented simulation scenarios; test robustness across seeds and effect assumptions; use cross-validation for model stability; add uncertainty intervals; improve validation and accessibility; only with suitable ethical approvals and governance, study appropriate real data.

### Likely interviewer questions

- **Why not use treatment group in the classifier?** The primary question is baseline response association, not treatment selection or treatment-benefit estimation.
- **Why can K-means produce different groupings if scaling changes?** Its distance geometry depends on feature scale, so the scaler is part of the clustering definition.
- **What does a positive coefficient mean?** Holding other transformed features fixed in this fitted model, it is associated with higher log-odds of the simulated response; it is not causal.
- **Why have a test set?** It estimates performance on records not used to fit the pipeline, reducing optimistic training-set evaluation.
- **What is a limitation of a single split?** Results can vary with the split. Cross-validation could assess stability, but this small demo uses one reproducible holdout.
- **Does a significant cluster test prove a subgroup benefits from treatment?** No. It only indicates association between cluster membership and simulated response in this exploratory dataset; it does not test individualized treatment benefit.

### Two-minute project explanation

> I built a local Python project that demonstrates a clinical-trial analysis workflow using entirely synthetic data. A seeded generator creates 800 artificial participants with baseline demographics and measurements, randomly assigns Treatment or Placebo independently of those baseline values, and simulates a noisy week-12 score. I define response as an illustrative reduction of at least ten points, which is a modelling assumption rather than a medically validated criterion.
>
> For exploratory stratification, I standardize age, baseline score, and a simulated biomarker, then use K-means to group participants without using their treatment assignments or outcomes. I describe each cluster by its baseline profile and only afterward compare response rates across clusters. Separately, I compare treatment-group response proportions with an appropriate two-sided test and report uncertainty cautiously.
>
> For prediction, I train a Logistic Regression pipeline using baseline features only. I split the data before fitting preprocessing, then evaluate on a held-out test set with accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and coefficients. This predicts a simulated response association; it does not estimate who benefits from a particular treatment.
>
> The dashboard makes the workflow inspectable and the tests check data validity and core functions. Because the data and mechanisms are simulated, none of the patterns or metrics are clinical evidence, and the application is not for clinical decision-making.
