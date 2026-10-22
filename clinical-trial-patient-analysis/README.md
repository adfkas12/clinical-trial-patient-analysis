# Clinical Trial Patient Stratification & Treatment Response Analysis

A local, beginner-friendly Python project demonstrating synthetic clinical-trial data generation, baseline patient stratification, response-rate comparisons, and baseline-only Logistic Regression.

> **Synthetic data for educational and research-method development purposes only. This application is not intended for clinical decision-making.**
>
> All patient records, treatment effects, response labels, statistical results, and model metrics in this project are simulated. They are not clinical evidence.

## Key features

- Generate 800 reproducible synthetic participant records by default.
- Validate the CSV schema, ranges, identifiers, and derived labels.
- Explore patient demographics, baseline measures, assignments, and simulated outcomes.
- Apply K-means to baseline features only; compare outcomes by cluster afterward as exploratory analysis.
- Compare simulated treatment-group response rates with an appropriate statistical test.
- Train and evaluate a leakage-safe Logistic Regression classifier using baseline characteristics.
- Explore the analyses in a local Streamlit dashboard.

## Technology stack

- Python 3.11 or a compatible installed Python version
- Pandas and NumPy for data handling and simulation
- Scikit-learn for preprocessing, clustering, classification, and evaluation
- SciPy for statistical tests
- Matplotlib and Seaborn for charts
- Streamlit for the dashboard
- Pytest for automated tests

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
| `tests/` | Offline pytest checks for data, preprocessing, clustering, modelling, and statistics. |
| `data/` | Generated CSV output; the CSV is ignored by Git. |
| `models/` | Reserved for optional future model artifacts; no model is persisted. |
| `.gitignore` | Excludes virtual environments, caches, generated data, and model artifacts. |

## Synthetic dataset and response-label definition

The generator creates a configurable number of participants (800 by default) using NumPy's explicit random generator and seed 42 by default. Re-running it with the same seed and participant count produces the same data. No real patient information is used.

| Column | Simulated definition |
| --- | --- |
| `patient_id` | Unique synthetic ID; never used as a predictive feature. |
| `age` | Integer from 18 through 85. |
| `sex` | Randomly simulated Female/Male category. This simplified variable is for demonstration and is not a complete representation of sex or gender. |
| `baseline_score` | Clipped normal sample on a 0–100 hypothetical severity scale; higher values represent worse baseline severity. |
| `biomarker_level` | Clipped gamma-distributed simulated value on a 0–100 scale. |
| `disease_severity` | Derived from baseline score: Mild below 35, Moderate from 35 to below 65, Severe at least 65. |
| `treatment_group` | Random Treatment/Placebo assignment, generated independently of baseline characteristics. |
| `week12_score` | Baseline score less simulated improvement, clipped to 0–100. Expected improvement includes 6 points in Placebo, an additional illustrative 4 points in Treatment, a small baseline-score component, and individual normally distributed variation (standard deviation 8.5). |
| `improvement` | Baseline score minus week-12 score, after score clipping. |
| `treatment_response` | 1 if improvement is at least 10 points; otherwise 0. This is an illustrative modelling assumption, not a medically validated threshold. |

Individual variation creates overlapping outcomes, so response is not a deterministic function of one baseline feature. The generator checks that both response classes appear and rejects invalid sample sizes below 20.

## Installation and execution

Open the project folder—the folder containing `app.py`—in VS Code. You can use **File > Open Folder**, or open a PowerShell terminal in that folder and run:

```powershell
code .
```

Run the following commands from the project root (the folder containing `requirements.txt`):

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
python -m pytest -q
python -m streamlit run app.py
```

Python 3.11 or a compatible version is expected; the project was run in this workspace with Python 3.13.3. The app opens locally in a browser. If the CSV is missing, the dashboard explains that `python generate_data.py` creates it.

If PowerShell blocks environment activation, use the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To create a different reproducible sample, run this from the project root:

```powershell
python -c "from src.data_generator import generate_synthetic_data, save_synthetic_data; from src.config import DATA_PATH; save_synthetic_data(generate_synthetic_data(1000, 17), DATA_PATH)"
```

The generated `data\synthetic_clinical_trial.csv` file is output and is intentionally not committed.

## Methodology

### Baseline-only patient stratification

K-means uses standardized `age`, `baseline_score`, and `biomarker_level`. It does not use treatment assignment or response/post-treatment columns. Standardization matters because K-means relies on distances, and otherwise numeric scale can cause a feature to dominate. Cluster IDs are arbitrary; interpret clusters through their feature profiles, not as meaningful clinical labels. Optional silhouette scores are geometric diagnostics, not proof of clinically meaningful groups.

### Statistical testing

The overall treatment comparison uses a two-sided two-proportion z-test when expected counts permit, or Fisher's exact test for sparse 2×2 counts. It reports a 95% Wald confidence interval for the response-rate difference.

Cluster response comparisons use a chi-square test when expected counts are adequate and a reproducible label-permutation chi-square test when they are sparse. These comparisons are exploratory. Clusters are formed without response labels, and response rates are computed only after assignments. Results do not establish a treatment effect in a subgroup.

### Baseline response prediction

The primary Logistic Regression uses only `age`, `sex`, `baseline_score`, and `biomarker_level`. Treatment group is intentionally excluded. This model estimates association with simulated response under the generator's mechanism; it does not estimate an individual's benefit from choosing one treatment over another.

Numeric scaling and categorical encoding are part of a Scikit-learn pipeline. Data is split before fitting that pipeline, so test records do not affect preprocessing. The test set is kept separate from training to provide a held-out evaluation.

## Model evaluation

Metrics are computed from actual held-out predictions: accuracy, precision, recall, F1-score, ROC-AUC (when applicable), confusion matrix, classification report, and fitted coefficients.

- **Accuracy** is the fraction of predictions that are correct.
- **Precision** is the fraction of predicted responders who are responders.
- **Recall** is the fraction of responders the model identifies.
- **F1-score** is the harmonic mean of precision and recall.
- **ROC-AUC** summarizes ranking of responders above non-responders across possible thresholds.
- **Confusion matrix** counts correct predictions and false positives/negatives by class.

Coefficients are on the transformed feature/log-odds scale. Their direction is associational, not causal. Test performance describes held-out data generated by the same synthetic mechanism; it does not establish real-world performance. Predicting response is distinct from estimating individualized treatment benefit, which requires comparing potential outcomes under alternative treatment assignments for a participant.

## Limitations

- All records and all generated patterns—including treatment effects, response labels, statistical results, and model metrics—are simulated, not clinical evidence.
- Simulated treatment assignment is randomized by design; real trials have additional design, conduct, and analysis considerations.
- Scores, biomarkers, response thresholds, and treatment effects are illustrative assumptions.
- The model predicts a simulated outcome, not counterfactual treatment benefit.
- Clusters are mathematical partitions, not validated clinical subgroups.
- Subgroup comparisons have uncertainty and are not confirmatory.
- Statistical significance does not equal clinical significance.
- This project must not be used for clinical decisions.

## Interview preparation

### Core ideas

1. **Problem:** Demonstrate a complete, explainable analysis workflow for stratification and outcome prediction using artificial trial-like records.
2. **Why synthetic data:** It avoids exposing personal health information and makes assumptions reproducible, while limiting claims to a coding demonstration.
3. **Generation:** A seeded NumPy generator samples demographics and baseline measures, independently randomizes treatment, simulates noisy improvement with a modest assumed treatment-group difference, clips scores to their valid range, and derives labels.
4. **Response definition:** A participant is a responder when baseline score minus week-12 score is at least 10. The threshold is illustrative, not medically established.
5. **Why Pandas:** DataFrames make tabular generation, validation, grouping, and descriptive summaries readable.
6. **Why scale for K-means:** K-means uses distance. Without scaling, a feature's numeric units or range could dominate the distance.
7. **How K-means works:** It assigns points to the nearest centroid, recomputes each centroid as its group's mean, and iterates to reduce within-cluster squared distances.
8. **Why response is excluded from clustering:** Otherwise the known outcome would define clusters, making later outcome comparisons circular.
9. **Why Logistic Regression:** It is a fast, interpretable binary-classification baseline that outputs probabilities and understandable coefficient directions.
10. **Confusion matrix:** Rows are actual classes and columns are predicted classes; it shows correct classifications and the two kinds of errors.
11. **Metrics:** Accuracy is the fraction correct; precision is the fraction of predicted responders who responded; recall is the fraction of responders found; F1 is the harmonic mean of precision and recall.
12. **ROC-AUC:** It summarizes how well the model ranks positive examples above negatives across classification thresholds; 0.5 is chance-level ranking in the usual binary setting.
13. **Hypothesis testing:** It assesses whether observed group differences are surprising under a stated no-difference model, subject to assumptions.
14. **P-value:** Under the null and assumptions, it is the probability of a statistic at least as extreme as observed. It is not the probability that the null is true or a measure of clinical importance.
15. **Leakage prevention:** Split first; fit encoding, scaling, and the model on training data; evaluate on the held-out test set.
16. **Why no post-treatment inputs:** Week-12 score and improvement occur after treatment and partly define the target. Using them for baseline prediction leaks information unavailable at prediction time.
17. **Response versus treatment benefit:** Response prediction estimates outcome under the observed simulated setup. Benefit prediction needs a comparison of potential outcomes under alternative assignments for the same participant; this model does not provide it.
18. **What it cannot establish:** It cannot establish clinical efficacy, validated biomarkers, real patient subgroups, clinical utility, causal effect sizes, or real-world model performance.
19. **Possible improvements:** Document multiple simulation scenarios, test robustness across seeds and assumptions, use cross-validation for model stability, add uncertainty intervals, and improve validation and accessibility. Studying real data would require appropriate approvals and governance.

### Likely interviewer questions

- **Why not use treatment group in the classifier?** The primary question is baseline response association, not treatment selection or treatment-benefit estimation.
- **Why can K-means produce different groupings if scaling changes?** Its distance geometry depends on feature scale, so the scaler is part of the clustering definition.
- **What does a positive coefficient mean?** Holding other transformed features fixed in this fitted model, it is associated with higher log-odds of simulated response; it is not causal.
- **Why have a test set?** It estimates performance on records not used to fit the pipeline, reducing optimistic training-set evaluation.
- **What is a limitation of a single split?** Results can vary with the split. Cross-validation could assess stability, but this demo uses one reproducible holdout.
- **Does a significant cluster test prove a subgroup benefits from treatment?** No. It only indicates association between cluster membership and simulated response in this exploratory dataset; it does not test individualized treatment benefit.

### Two-minute project explanation

> I built a local Python project that demonstrates a clinical-trial analysis workflow using entirely synthetic data. A seeded generator creates 800 artificial participants with baseline demographics and measurements, randomly assigns Treatment or Placebo independently of those baseline values, and simulates a noisy week-12 score. I define response as an illustrative reduction of at least ten points, which is a modelling assumption rather than a medically validated criterion.
>
> For exploratory stratification, I standardize age, baseline score, and a simulated biomarker, then use K-means to group participants without using their treatment assignments or outcomes. I describe each cluster by its baseline profile and only afterward compare response rates across clusters. Separately, I compare treatment-group response proportions with an appropriate two-sided test and report uncertainty cautiously.
>
> For prediction, I train a Logistic Regression pipeline using baseline features only. I split the data before fitting preprocessing, then evaluate on a held-out test set with accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and coefficients. This predicts a simulated response association; it does not estimate who benefits from a particular treatment.
>
> The dashboard makes the workflow inspectable and the tests check data validity and core functions. Because the data and mechanisms are simulated, none of the patterns or metrics are clinical evidence, and the application is not for clinical decision-making.

## Future improvements

- Compare model stability across multiple reproducible train/test splits or cross-validation.
- Explore alternative documented synthetic-data assumptions and their impact on analyses.
- Add uncertainty summaries while keeping subgroup findings explicitly exploratory.
- Improve accessibility and expand validation for malformed input files.
