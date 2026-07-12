# Part 2 — Expected-Power Regression & Underperformance Classification

## 1. Project Title
GridSense — Part 2: Supervised Machine Learning (Regression + Classification)

## 2. Dataset
- **Source**: `cleaned_data.csv`, produced by Part 1 (`part1_eda.py`) from the Kaggle Solar Power Generation dataset (Plant 1 generation + weather sensor data).
- **Rows (full cleaned dataset)**: *fill in from your own run — printed as `Shape: (...)` at the top of the console output.*
- **Daytime-only modeling subset** (`IRRADIATION > 0`): *fill in — printed as `Daytime-only subset used for modeling: (...)`.*
- **Targets**:
  - Regression target: `AC_POWER` (used to fit the *expected power* model)
  - Classification target: `UNDERPERFORMING` (binary, derived — see Label Definitions below)

> Fill in the exact row/column counts from your console output before submitting — they depend on your specific run of Part 1's cleaning steps.

## 3. Label Definitions — corrected from a fixed-threshold approach

**Regression label**
```
AC_POWER — predicted from environmental features only
(IRRADIATION, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, TEMP_DIFF,
 SOLAR_HEATING_INDEX, HOUR)
```
This is the "expected power" model: what an inverter *should* produce given today's weather and time of day, using no information about what it actually produced.

**Classification label — Performance Ratio, not a fixed bottom-20% cutoff**

An earlier version of this project defined "underperforming" as simply the bottom 20% of raw `AC_POWER` readings. That confuses naturally low output (sunrise, sunset, overcast, winter) with an actual equipment fault, since a plant produces very little power at those times *regardless of whether every inverter is healthy*.

Instead:
```
PERFORMANCE_RATIO = AC_POWER (actual) / EXPECTED_POWER (from the regression model)

RISK_CATEGORY:
    PR >= 0.90          -> Normal
    0.75 <= PR < 0.90   -> Monitor
    PR < 0.75           -> Underperforming
(rows flagged SHUTDOWN_FLAG=1 in Part 1 are forced to Underperforming
 regardless of PR, since near-zero output at high irradiation is a hard
 fault, not a borderline case)

UNDERPERFORMING (binary) = 1 if RISK_CATEGORY == "Underperforming" else 0
```
`EXPECTED_POWER` itself comes from a `RandomForestRegressor` fit only on non-fault rows, using **out-of-fold cross-validated predictions** (`cross_val_predict`, 5-fold) so that every row's expected value is estimated without that row's own true `AC_POWER` ever being seen by the model that predicted it — this avoids leakage between the "expected" estimate and the label built from it.

- Class balance (`UNDERPERFORMING`): *fill in — printed as "Underperforming share: ...%".*

## 4. Feature Encoding
- **`DAY_PERIOD` → label encoding** (`Night=0, Morning=1, Afternoon=2, Evening=3`): chosen over one-hot because the categories have a genuine natural ordering across the day, which a single ordinal column preserves for tree-based and linear models alike.
- **`SOURCE_KEY_GEN` (inverter ID) → label-encoded to `INVERTER_ID_ENC`**: with ~22 inverters, one-hot encoding would add ~21 sparse columns for a single categorical identity feature; a compact numeric code keeps the feature set manageable for the classifier while still letting tree-based models split on inverter identity where it matters (e.g. a specific unit with a chronic fault).
- Note the **expected-power regression model intentionally excludes** `AC_POWER`, its lags, its rolling stats, `EFFICIENCY`, and inverter identity — it only sees weather + time, because its whole purpose is to say what power *should* look like independent of what any particular inverter actually did.
- The **classifier** is allowed to see `AC_POWER` and everything derived from it (lags, rolling mean/std, `EFFICIENCY`, `EXPECTED_POWER`) — that's not leakage, since the classifier's job is to look at what actually happened and flag it. `PERFORMANCE_RATIO`, `POWER_DROP`, and `RISK_CATEGORY` themselves are excluded from the classifier's feature set since they would trivially encode the label by definition.

## 5. Train-Test Split
```
80% Training / 20% Testing
Random State = 42
Stratified by UNDERPERFORMING for the classification split.

StandardScaler fitted ONLY on training data (separately for the
regression feature set and the classification feature set) to avoid
leaking test-set distribution statistics into scaling.
```

## 6. Regression Results — Expected Power

*Replace the placeholders below with your actual console values (Task 3 / Task 4 / Task 5 sections).*

```
Linear Regression -> MSE: <fill in>   R²: <fill in>
Ridge Regression   -> MSE: <fill in>   R²: <fill in>
RandomForest (out-of-fold, expected power) -> MSE: <fill in>   R²: <fill in>
```

| Model | MSE | R² |
|---|---|---|
| Linear Regression (OLS) | *fill in* | *fill in* |
| Ridge (alpha=1.0) | *fill in* | *fill in* |
| RandomForest expected power (OOF) | *fill in* | *fill in* |

The RandomForest expected-power model is the one actually used downstream to compute `PERFORMANCE_RATIO` — it captures the non-linear irradiation/temperature relationship better than a linear fit, which matters because Performance Ratio is only meaningful if "expected power" is a realistic estimate.

## 7. Coefficient Interpretation (Linear Regression, standardized features)

*Replace with your actual top coefficients from `results/linear_regression_coefficients.csv`.*

Top features by |standardized coefficient|:
1. `IRRADIATION` — *fill in coefficient*
2. *fill in*
3. *fill in*

**Positive coefficient**: increasing that feature (holding others fixed) is associated with higher predicted AC power — expected for `IRRADIATION`.
**Negative coefficient**: increasing that feature is associated with lower predicted AC power — plausible for something like `MODULE_TEMPERATURE` at the margin, since panel efficiency drops slightly as cells heat up, even though higher temperature is correlated with higher irradiation overall.

## 8. Classification Results — Underperformance Detector (Logistic Regression, C=1.0)

*Fill in from your console output / `results/classification_report.csv` / `results/confusion_matrix.csv`.*

```
Accuracy:  <fill in>
Precision: <fill in>
Recall:    <fill in>
F1:        <fill in>
AUC:       <fill in>
```

- Confusion matrix: `figures/confusion_matrix.png`, `results/confusion_matrix.csv`
- ROC curve: `figures/roc_curve.png`

Class imbalance was handled with SMOTE applied to the **training set only** (never the test set), since the underperforming class is the minority class after switching to the Performance-Ratio label.

## 9. Precision / Recall Formulas
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```
For this use case, a **false negative** (a truly underperforming inverter missed by the model) is usually more costly than a false positive (an unnecessary inspection) — a missed fault can mean lost generation for days before anyone notices. That's the basis for preferring recall-favoring thresholds in Section 10.

## 10. Threshold Sensitivity

*Fill in from `results/threshold_results.csv`.*

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | *fill in* | *fill in* | *fill in* |
| 0.40 | *fill in* | *fill in* | *fill in* |
| 0.50 | *fill in* | *fill in* | *fill in* |
| 0.60 | *fill in* | *fill in* | *fill in* |
| 0.70 | *fill in* | *fill in* | *fill in* |

F1-maximizing threshold: *fill in* (F1 = *fill in*).

**Note**: the F1-maximizing threshold is not necessarily the one an operator should use. Since missing a genuine fault is costlier than a false alarm (see Section 9), a plant operator may deliberately choose a **lower** threshold than the F1-optimal one, trading some precision for higher recall.

## 11. Logistic Regularization Comparison

*Fill in from `results/logistic_regularization.csv`.*

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| C=1.0 (baseline) | *fill in* | *fill in* | *fill in* | *fill in* |
| C=0.01 (strong L2) | *fill in* | *fill in* | *fill in* | *fill in* |

`C` is the inverse of the L2 regularization strength in scikit-learn's `LogisticRegression` — a **smaller** `C` means **stronger** regularization (coefficients pulled harder toward zero), which tends to produce a simpler, less confident decision boundary. Comparing `C=1.0` against `C=0.01` shows how much the model's discrimination (AUC) and precision/recall trade-off degrade under heavy regularization.

## 12. Bootstrap Confidence Interval for AUC Difference

*Fill in from your console output (Task 6b).*

```
Mean AUC difference (C=1.0 − C=0.01) over <n> valid bootstrap samples: <fill in>
95% CI: [<fill in>, <fill in>]
Does the 95% CI exclude zero? <fill in>
```

**Interpretation**: if the 95% CI excludes zero, the AUC difference between the two regularization strengths is statistically distinguishable from noise at this sample size — i.e., the weaker-regularized model (`C=1.0`) is reliably better (or worse) than the strongly-regularized one, not just better by chance on this particular test split. If the CI includes zero, the observed AUC difference could plausibly be due to sampling variability alone.

## 13. How to Run
```bash
pip install -r requirements.txt
python part2/part2_models.py
```
Expects `cleaned_data.csv` (from Part 1) to be discoverable — either in the current directory, `/mnt/user-data/uploads/`, `/content/`, `data/`, or anywhere under the current working directory tree (see `find_file()` in the script).

## Outputs produced
```
figures/
    confusion_matrix.png
    roc_curve.png
    threshold_sensitivity.png

results/
    linear_regression_coefficients.csv
    ridge_vs_linear.csv
    classification_report.csv
    confusion_matrix.csv
    threshold_results.csv
    logistic_regularization.csv
    model_summary.csv

modeling_data.csv   <- full feature set + PERFORMANCE_RATIO + RISK_CATEGORY +
                       UNDERPERFORMING, consumed directly by Part 3
```

## Repository structure
```
part2/
    README.md
    part2_models.py
    cleaned_data.csv        (from Part 1, or documented download step)
    modeling_data.csv       (output, consumed by Part 3)
    figures/
        confusion_matrix.png
        roc_curve.png
        threshold_sensitivity.png
    results/
        linear_regression_coefficients.csv
        ridge_vs_linear.csv
        classification_report.csv
        confusion_matrix.csv
        threshold_results.csv
        logistic_regularization.csv
        model_summary.csv
```