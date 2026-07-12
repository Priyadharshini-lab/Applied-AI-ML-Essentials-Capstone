"""
=====================================================================
 PART 2 — Supervised Machine Learning: Regression + Classification
 Input : cleaned_data.csv (produced by Part 1's part1_eda.py)
=====================================================================
Run with:  python3 part2_models.py
Produces:
  - console output for every task
  - figures/roc_curve.png
  - figures/threshold_sensitivity.png
  - figures/confusion_matrix.png
  - results/linear_regression_coefficients.csv
  - results/top3_features.csv
  - results/ridge_vs_linear.csv
  - results/threshold_results.csv
  - results/logistic_regularization.csv
  - results/classification_report.csv
  - results/confusion_matrix.csv
  - results/model_summary.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
)
from imblearn.over_sampling import SMOTE

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
np.random.seed(42)

FIG_DIR = "figures"
RESULTS_DIR = "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def find_file(filename):
    candidates = [filename, f"/mnt/user-data/uploads/{filename}", f"/content/{filename}",
                  f"data/{filename}"]
    for path in candidates:
        matches = glob.glob(path, recursive=True)
        if matches:
            return matches[0]
    matches = glob.glob(f"**/{filename}", recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find '{filename}'.")


# =====================================================================
# LOAD CLEANED DATA FROM PART 1
# =====================================================================
print("=" * 70)
print("LOAD cleaned_data.csv")
print("=" * 70)

path = find_file("cleaned_data.csv")
print(f"Loading: {path}")
df = pd.read_csv(path)
print(f"Shape: {df.shape}")

# ---------------------------------------------------------------------
# Restrict to daytime readings only (IRRADIATION > 0).
# Rationale (see README): ~46% of all rows are exact nighttime zeros
# (IRRADIATION = 0 -> AC_POWER = 0 for every inverter, always). Keeping
# them in would let both models "cheat" by trivially learning
# night = 0, inflating R^2 and making any binary split of AC_POWER
# collapse onto zero. Restricting to real daylight production is also
# what the business case (flagging an underperforming inverter while
# the sun is actually up) requires.
# ---------------------------------------------------------------------
day = df[df["IRRADIATION"] > 0].copy()
print(f"Daytime-only subset used for modeling: {day.shape}")

# =====================================================================
# TASK 1 — DEFINE X, y_reg, y_clf
# =====================================================================
print("\n" + "=" * 70)
print("TASK 1: DEFINE FEATURES AND LABELS")
print("=" * 70)

# y_reg: continuous target = AC_POWER (kW) actually generated
y_reg = day["AC_POWER"].copy()

# y_clf: "underperformance flag" -- 1 if AC_POWER falls in the bottom 20%
# of ALL DAYTIME readings, 0 otherwise. This is the natural binary label
# for GridSense: "is this inverter producing far less than a normal
# daytime reading, given the sun is genuinely up?"
clf_threshold = day["AC_POWER"].quantile(0.20)
y_clf = (day["AC_POWER"] < clf_threshold).astype(int)
print(f"Classification threshold (20th percentile of daytime AC_POWER): {clf_threshold:.2f} kW")
print(f"y_clf class balance:\n{y_clf.value_counts()}")
print(f"Positive (underperforming) class share: {y_clf.mean()*100:.1f}%")

# Feature matrix: exclude anything that leaks the target or carries no
# signal -> AC_POWER (target itself), DC_POWER (near-perfect proxy of
# AC_POWER, corr = 1.00 from Part 1 -> would leak), DAILY_YIELD /
# TOTAL_YIELD (cumulative counters that already encode today's power
# history -> leakage), DATE_TIME (raw timestamp, not usable directly),
# PLANT_ID (single constant value in this file -> zero variance),
# SOURCE_KEY_WX (one constant weather-station id -> zero variance).
drop_cols = ["AC_POWER", "DC_POWER", "DAILY_YIELD", "TOTAL_YIELD",
             "DATE_TIME", "PLANT_ID", "SOURCE_KEY_WX"]
X = day.drop(columns=drop_cols)
print(f"\nFeature columns kept: {list(X.columns)}")

# =====================================================================
# TASK 2 — ENCODE CATEGORICAL COLUMNS
# =====================================================================
print("\n" + "=" * 70)
print("TASK 2: CATEGORICAL ENCODING")
print("=" * 70)

# DAY_PERIOD has a natural order along the course of a day
# (Morning -> Afternoon -> Evening; "Night" was dropped with the
# daytime-only filter). Ordinal/label encoding is justified here because
# the categories represent an inherent progression through the day,
# not an arbitrary label.
day_period_order = {"Night": 0, "Morning": 1, "Afternoon": 2, "Evening": 3}
X["DAY_PERIOD_ENC"] = X["DAY_PERIOD"].map(day_period_order)
X = X.drop(columns=["DAY_PERIOD"])
print("DAY_PERIOD -> label-encoded as Night=0, Morning=1, Afternoon=2, Evening=3 (natural time order).")
print("(A small number of dawn/dusk daytime readings still carry the Night/Evening hour-bucket label; "
      "the ordinal mapping still reflects true chronological progression through the day.)")

# SOURCE_KEY_GEN (which of the 22 physical inverters produced the
# reading) has NO natural order -- the IDs are arbitrary strings.
# Label-encoding them (e.g. inverter "A"=0, "B"=1, ...) would falsely
# imply inverter 21 is "greater than" inverter 0 to a linear/logistic
# model, which is meaningless. One-hot encoding avoids this by giving
# each inverter its own independent 0/1 column, with the first category
# dropped to avoid multicollinearity (the dummy-variable trap).
X = pd.get_dummies(X, columns=["SOURCE_KEY_GEN"], drop_first=True)
print(f"SOURCE_KEY_GEN -> one-hot encoded (drop_first=True). New feature count: {X.shape[1]}")

# Ensure all feature columns are numeric before scaling
X = X.astype(float)
print(f"\nFinal X shape: {X.shape}")

# =====================================================================
# TASK 3 — LEAK-FREE TRAIN/TEST SPLIT AND SCALING
# =====================================================================
print("\n" + "=" * 70)
print("TASK 3: TRAIN/TEST SPLIT AND SCALING")
print("=" * 70)

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f"Train shape: {X_train.shape}   Test shape: {X_test.shape}")
print("stratify=y_clf used so the underperformance class ratio is preserved "
      "identically in both the train and test splits.")

scaler = StandardScaler()
scaler.fit(X_train)                       # fit ONLY on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # test set only ever transformed, never fit

print(
    "\nNOTE (data leakage): the scaler is fit only on X_train. Fitting it on the "
    "full dataset (train+test combined) would let the mean/std used to scale the "
    "training features be influenced by the test set's values -- effectively letting "
    "the model 'see' statistics about data it is supposed to be evaluated on later. "
    "That inflates reported performance and is a classic form of data leakage."
)

# =====================================================================
# TASK 4 — LINEAR REGRESSION
# =====================================================================
print("\n" + "=" * 70)
print("TASK 4: LINEAR REGRESSION")
print("=" * 70)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
r2_lin = r2_score(y_reg_test, y_pred_reg)
print(f"Linear Regression -> MSE: {mse_lin:.3f}   R^2: {r2_lin:.4f}")

coef_table = pd.DataFrame({"feature": X.columns, "coefficient": lin_reg.coef_})
coef_table["abs_coef"] = coef_table["coefficient"].abs()
coef_table = coef_table.sort_values("abs_coef", ascending=False)
print("\nAll coefficients (sorted by |coefficient|):")
print(coef_table.to_string(index=False))

top3 = coef_table.head(3)
print("\nTop 3 features by |coefficient|:")
print(top3.to_string(index=False))

# --- SAVE: top-3 feature table ---
top3.to_csv(f"{RESULTS_DIR}/top3_features.csv", index=False)
print(f"Saved: {RESULTS_DIR}/top3_features.csv")

# --- SAVE: coefficient table ---
coef_table.to_csv(f"{RESULTS_DIR}/linear_regression_coefficients.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/linear_regression_coefficients.csv")

# =====================================================================
# TASK 4b — RIDGE REGRESSION
# =====================================================================
print("\n" + "=" * 70)
print("TASK 4b: RIDGE REGRESSION (alpha=1.0)")
print("=" * 70)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"Ridge Regression  -> MSE: {mse_ridge:.3f}   R^2: {r2_ridge:.4f}")

reg_comparison = pd.DataFrame({
    "Model": ["Linear Regression (OLS)", "Ridge (alpha=1.0)"],
    "MSE": [mse_lin, mse_ridge],
    "R2": [r2_lin, r2_ridge],
})
print("\nOLS vs Ridge comparison:")
print(reg_comparison.to_string(index=False))

# --- SAVE: OLS vs Ridge comparison ---
reg_comparison.to_csv(f"{RESULTS_DIR}/ridge_vs_linear.csv", index=False)
print(f"Saved: {RESULTS_DIR}/ridge_vs_linear.csv")

# =====================================================================
# TASK 5 — LOGISTIC REGRESSION (with class-imbalance handling)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 5: LOGISTIC REGRESSION + CLASS IMBALANCE")
print("=" * 70)

train_counts_before = y_clf_train.value_counts()
minority_share = train_counts_before.min() / train_counts_before.sum()
print(f"y_clf_train class counts BEFORE resampling:\n{train_counts_before}")
print(f"Minority class share: {minority_share*100:.1f}%")

if minority_share < 0.35:
    print("\nMinority class < 35% -> applying SMOTE to the TRAINING set only.")
    smote = SMOTE(random_state=42)
    X_train_bal, y_clf_train_bal = smote.fit_resample(X_train_scaled, y_clf_train)
    train_counts_after = pd.Series(y_clf_train_bal).value_counts()
    print(f"y_clf_train class counts AFTER SMOTE:\n{train_counts_after}")
else:
    X_train_bal, y_clf_train_bal = X_train_scaled, y_clf_train
    print("Classes sufficiently balanced -- no resampling applied.")

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_bal, y_clf_train_bal)

y_pred_clf = log_reg.predict(X_test_scaled)
y_proba_clf = log_reg.predict_proba(X_test_scaled)[:, 1]

# --- Accuracy printed separately (in addition to classification_report) ---
accuracy_base = accuracy_score(y_clf_test, y_pred_clf)
print(f"\nAccuracy (C=1.0): {accuracy_base:.4f}")

cm = confusion_matrix(y_clf_test, y_pred_clf)
print(f"\nConfusion matrix (rows=actual, cols=predicted):\n{cm}")

report = classification_report(y_clf_test, y_pred_clf, digits=3)
print(f"\nClassification report:\n{report}")

# --- SAVE: classification report as csv ---
report_dict = classification_report(y_clf_test, y_pred_clf, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv(f"{RESULTS_DIR}/classification_report.csv")
print(f"Saved: {RESULTS_DIR}/classification_report.csv")

# --- SAVE: confusion matrix values as csv ---
pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]
).to_csv(f"{RESULTS_DIR}/confusion_matrix.csv")
print(f"Saved: {RESULTS_DIR}/confusion_matrix.csv")

# --- SAVE: confusion matrix figure ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Underperforming"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Underperformance Classifier (C=1.0)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=110)
plt.close()
print(f"Confusion matrix figure saved to {FIG_DIR}/confusion_matrix.png")

fpr, tpr, roc_thresholds = roc_curve(y_clf_test, y_proba_clf)
auc_base = roc_auc_score(y_clf_test, y_proba_clf)
print(f"\nAUC (C=1.0, SMOTE-balanced training): {auc_base:.4f}")
print(f"Number of ROC thresholds evaluated: {len(roc_thresholds)}")

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"Logistic Regression (AUC = {auc_base:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Underperformance Classifier")
plt.annotate(f"AUC = {auc_base:.3f}", xy=(0.55, 0.25), fontsize=12,
             bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_curve.png", dpi=110)
plt.close()
print(f"ROC curve saved to {FIG_DIR}/roc_curve.png")

# =====================================================================
# TASK 5b — DECISION-THRESHOLD SENSITIVITY (0.30 -> 0.70)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 5b: THRESHOLD SENSITIVITY")
print("=" * 70)

thresholds_to_test = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_rows = []
for t in thresholds_to_test:
    preds_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, preds_t, zero_division=0)
    r = recall_score(y_clf_test, preds_t, zero_division=0)
    f1 = f1_score(y_clf_test, preds_t, zero_division=0)
    threshold_rows.append((t, p, r, f1))

threshold_table = pd.DataFrame(threshold_rows, columns=["Threshold", "Precision", "Recall", "F1"])
print(threshold_table.to_string(index=False))

best_threshold_row = threshold_table.loc[threshold_table["F1"].idxmax()]
print(f"\nF1-maximizing threshold: {best_threshold_row['Threshold']:.2f} "
      f"(F1 = {best_threshold_row['F1']:.3f})")

# --- SAVE: threshold table ---
threshold_table.to_csv(f"{RESULTS_DIR}/threshold_results.csv", index=False)
print(f"Saved: {RESULTS_DIR}/threshold_results.csv")

plt.figure(figsize=(8, 5))
plt.plot(threshold_table["Threshold"], threshold_table["Precision"], marker="o", label="Precision")
plt.plot(threshold_table["Threshold"], threshold_table["Recall"], marker="o", label="Recall")
plt.plot(threshold_table["Threshold"], threshold_table["F1"], marker="o", label="F1")
plt.title("Precision / Recall / F1 vs Decision Threshold")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/threshold_sensitivity.png", dpi=110)
plt.close()
print(f"Threshold sensitivity plot saved to {FIG_DIR}/threshold_sensitivity.png")

# =====================================================================
# TASK 6 — REGULARIZATION EXPERIMENT (C=1.0 vs C=0.01)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 6: REGULARIZATION EXPERIMENT (C=1.0 vs C=0.01)")
print("=" * 70)

log_reg_strong = LogisticRegression(max_iter=1000, C=0.01, random_state=42)
log_reg_strong.fit(X_train_bal, y_clf_train_bal)

y_pred_strong = log_reg_strong.predict(X_test_scaled)
y_proba_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]

precision_base = precision_score(y_clf_test, y_pred_clf, zero_division=0)
recall_base = recall_score(y_clf_test, y_pred_clf, zero_division=0)

precision_strong = precision_score(y_clf_test, y_pred_strong, zero_division=0)
recall_strong = recall_score(y_clf_test, y_pred_strong, zero_division=0)
accuracy_strong = accuracy_score(y_clf_test, y_pred_strong)
auc_strong = roc_auc_score(y_clf_test, y_proba_strong)

print(f"Accuracy (C=0.01): {accuracy_strong:.4f}")

reg_table = pd.DataFrame({
    "Model": ["C=1.0 (baseline)", "C=0.01 (strong L2)"],
    "Accuracy": [accuracy_base, accuracy_strong],
    "Precision": [precision_base, precision_strong],
    "Recall": [recall_base, recall_strong],
    "AUC": [auc_base, auc_strong],
})
print(reg_table.to_string(index=False))

# --- SAVE: logistic regularization comparison ---
reg_table.to_csv(f"{RESULTS_DIR}/logistic_regularization.csv", index=False)
print(f"Saved: {RESULTS_DIR}/logistic_regularization.csv")

# =====================================================================
# TASK 6b — BOOTSTRAP CONFIDENCE INTERVAL FOR AUC DIFFERENCE
# =====================================================================
print("\n" + "=" * 70)
print("TASK 6b: BOOTSTRAP CI FOR AUC DIFFERENCE (C=1.0 minus C=0.01)")
print("=" * 70)

y_clf_test_arr = np.asarray(y_clf_test)
n = len(y_clf_test_arr)
n_boot = 500

# FIX: use a list and only append valid draws, instead of pre-allocating
# a fixed-size array and leaving some slots uninitialized whenever a
# bootstrap sample happens to contain only one class (AUC is undefined
# in that case). Appending guarantees every entry in `diffs` corresponds
# to an actual computed value.
# FIX: use a list and only append valid draws, instead of pre-allocating
# a fixed-size array and leaving some slots uninitialized whenever a
# bootstrap sample happens to contain only one class (AUC is undefined
# in that case). Appending guarantees every entry in `diffs` corresponds
# to an actual computed value.
diffs_list = []
rng = np.random.default_rng(42)  # NumPy's newer recommended generator
n_skipped = 0
for i in range(n_boot):
    idx = rng.choice(n, size=n, replace=True)
    y_sample = y_clf_test_arr[idx]
    if len(np.unique(y_sample)) < 2:
        n_skipped += 1
        continue
    auc_base_i = roc_auc_score(y_sample, y_proba_clf[idx])
    auc_strong_i = roc_auc_score(y_sample, y_proba_strong[idx])
    diffs_list.append(auc_base_i - auc_strong_i)

print(f"Bootstrap samples used: {len(diffs_list)} / {n_boot}  (skipped {n_skipped} single-class samples)")

# SAFETY: guard against the (extremely unlikely) case where every single
# bootstrap draw was skipped, which would otherwise make .mean() / percentile
# calls below fail on an empty array with a confusing error.
if len(diffs_list) == 0:
    raise ValueError(
        "No valid bootstrap samples were produced (every resample contained "
        "only one class). Increase n_boot or check y_clf_test's class balance."
    )

diffs = np.array(diffs_list)

mean_diff = diffs.mean()
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

print(f"Mean AUC difference (C=1.0 - C=0.01) over {len(diffs)} valid bootstrap samples: {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
excludes_zero = (ci_low > 0) or (ci_high < 0)
print(f"Does the 95% CI exclude zero? {excludes_zero}")

# =====================================================================
# TASK 7 — OVERALL MODEL SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("TASK 7: MODEL SUMMARY")
print("=" * 70)

summary = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression",
              "Logistic Regression (C=1.0)", "Logistic Regression (C=0.01)"],
    "MSE": [mse_lin, mse_ridge, np.nan, np.nan],
    "R2": [r2_lin, r2_ridge, np.nan, np.nan],
    "Accuracy": [np.nan, np.nan, accuracy_base, accuracy_strong],
    "Precision": [np.nan, np.nan, precision_base, precision_strong],
    "Recall": [np.nan, np.nan, recall_base, recall_strong],
    "AUC": [np.nan, np.nan, auc_base, auc_strong],
})
print(summary.to_string(index=False))

summary.to_csv(f"{RESULTS_DIR}/model_summary.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/model_summary.csv")

print("\nDONE. All figures in ./figures/, all tables in ./results/.")