# Predictors of hospital uncompensated care burden: analysis code
Analysis code for "Identifying Predictors of Hospital Uncompensated Care Burden in the United States: Cross-Sectional
Study Integrating Beta Regression and Interpretable Machine Learning" (JMIR Medical Informatics; manuscript under review).

## Analysis

- **Inferential analysis.** Beta regression with standard errors clustered by state, a CAH by Medicaid expansion
  interaction, multiple imputation for the extended predictor sets, and sensitivity analyses.
- **Machine learning.** XGBoost, random forest, multilayer perceptron, and elastic net for the continuous ratio and the
  high-burden indicator, compared with prespecified logistic and beta regression benchmarks in repeated nested
  cross-validation (5 folds x 3 repeats) with the same 3 nested predictor sets; SHAP values computed out of fold.
- **Temporal validation.** Out-of-fold predictions from 2023 characteristics compared with fiscal year 2024 outcomes.
- **No identifiers.** Hospital identifiers, names, addresses, fine-grained geographic codes, survey-process fields, and
  cost report financial fields are blocked from every model by `features.assert_clean()`.

## Contents

| Path | Content |
|---|---|
| `run_all.py` | Runs the analysis steps in order |
| `src/parameters.py` | Analysis parameters reported in the Methods (seed, cutoff, cross-validation, imputation) |
| `src/*.py` | One module per analysis step (see the table below) |

## Running

```
python run_all.py --data-dir <folder with the licensed files> --out-dir <output folder>
python run_all.py --data-dir <folder with the licensed files> --out-dir <output folder> --steps ml,report
```

| Step | Module | What it does |
|---|---|---|
| sample | `src/sample.py` | Links the AHA and cost report data on the AHA identifier, applies the eligibility criteria, derives the outcomes and covariates, and links Medicaid expansion and county data (sample flow in Figure 2) |
| missing | `src/missing.py` | Missing values by variable and type, Little test, nonresponse models, and multiple imputation by chained equations (60 data sets) |
| measures | `src/features.py` | The 7 social needs and health equity measures and their internal consistency (Table 1) |
| inferential | `src/inferential.py` | Beta regression, adjusted means, extended models with imputed data, and sensitivity analyses |
| ml | `src/ml.py` | Repeated nested cross-validation of all learners and benchmarks, with out-of-fold predictions and SHAP values |
| cah | `src/cah_analysis.py` | How much of the discrimination reflects CAH status, and identifier-only models |
| shap | `src/shap_analysis.py` | SHAP feature and feature-group importance and ranking stability (Figures 5 and 6) |
| ao7 | `src/ao7.py` | Dependence of XGBoost regression performance on the model configuration (and on single 80/20 splits, if the optional file is present) |
| temporal | `src/temporal.py` | Temporal validation against fiscal year 2024 outcomes from the public HCRIS files (Figure 4) |
| report | `src/report.py` | Manuscript tables 2-4, Figure 3, and the appendix tables and figures |

The analysis was run with Python 3.10, pandas 2.3, NumPy 2.2, SciPy 1.15, scikit-learn 1.7, statsmodels 0.15,
XGBoost 3.2, and matplotlib 3.10. SHAP values are computed with the TreeSHAP implementation in XGBoost.
