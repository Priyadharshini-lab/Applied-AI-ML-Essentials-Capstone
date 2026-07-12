"""
=====================================================================
 PART 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline
 Input : cleaned_data.csv (produced by Part 1's part1_eda.py)
=====================================================================
Run with:  python3 part3_ensembles.py

Reproduces X_train_scaled / X_test_scaled / y_clf_train / y_clf_test
using the EXACT same feature engineering, split (random_state=42,
stratify=y_clf), and scaling as Part 2's part2_models.py, so this
script can be run standalone.

Produces:
  - console output for every task
  - figures/feature_importance_top5.png
  - figures/learning_curve.png
  - results/decision_tree_comparison.csv
  - results/gini_entropy_comparison.csv
  - results/random_forest_feature_importance.csv
  - results/feature_ablation.csv
  - results/cv_comparison.csv
  - results/gridsearch_results.csv
  - results/learning_curve.csv
  - results/final_model_summary.csv
  - best_model.pkl
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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
# REBUILD X_train_scaled / X_test_scaled / y_clf_train / y_clf_test
# (identical logic to Part 2 -- kept in sync so this script is
# self-contained and reproducible without needing to re-run Part 2)
# =====================================================================
print("=" * 70)
print("REBUILDING PART 2 FEATURES / SPLIT / SCALING")
print("=" * 70)

path = find_file("cleaned_data.csv")
print(f"Loading: {path}")
df = pd.read_csv(path)

day = df[df["IRRADIATION"] > 0].copy()

y_reg = day["AC_POWER"].copy()
clf_threshold = day["AC_POWER"].quantile(0.20)
y_clf = (day["AC_POWER"] < clf_threshold).astype(int)

drop_cols = ["AC_POWER", "DC_POWER", "DAILY_YIELD", "TOTAL_YIELD",
             "DATE_TIME", "PLANT_ID", "SOURCE_KEY_WX"]
X = day.drop(columns=drop_cols)

day_period_order = {"Night": 0, "Morning": 1, "Afternoon": 2, "Evening": 3}
X["DAY_PERIOD_ENC"] = X["DAY_PERIOD"].map(day_period_order)
X = X.drop(columns=["DAY_PERIOD"])

X = pd.get_dummies(X, columns=["SOURCE_KEY_GEN"], drop_first=True)
X = X.astype(float)

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = X.columns.tolist()
print(f"Train shape: {X_train.shape}   Test shape: {X_test.shape}")
print(f"Feature count: {len(feature_names)}")

# =====================================================================
# TASK 1 — DECISION TREE BASELINE (UNCONSTRAINED)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 1: DECISION TREE BASELINE (max_depth=None)")
print("=" * 70)

dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train_scaled, y_clf_train)

dt_base_train_acc = accuracy_score(y_clf_train, dt_base.predict(X_train_scaled))
dt_base_test_acc = accuracy_score(y_clf_test, dt_base.predict(X_test_scaled))
print(f"Unconstrained tree -> Train accuracy: {dt_base_train_acc:.4f}   Test accuracy: {dt_base_test_acc:.4f}")
print(f"Train/Test gap: {dt_base_train_acc - dt_base_test_acc:.4f}")
print(
    "NOTE (for README): a train accuracy near 1.0 with a materially lower test "
    "accuracy is the signature of overfitting. Decision trees are 'high-variance' "
    "learners because at each node the greedy split-selection algorithm locks in "
    "a decision based only on the samples that reached that node -- it never "
    "revisits or corrects earlier splits -- so with no depth limit the tree keeps "
    "partitioning until it can memorize individual training rows (including their noise)."
)

# =====================================================================
# TASK 2 — CONTROLLED DECISION TREE
# =====================================================================
print("\n" + "=" * 70)
print("TASK 2: CONTROLLED DECISION TREE (max_depth=5, min_samples_split=20)")
print("=" * 70)

dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)

dt_ctrl_train_acc = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
dt_ctrl_test_acc = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))
print(f"Controlled tree -> Train accuracy: {dt_ctrl_train_acc:.4f}   Test accuracy: {dt_ctrl_test_acc:.4f}")
print(f"Train/Test gap: {dt_ctrl_train_acc - dt_ctrl_test_acc:.4f}")
print(
    "NOTE (for README): max_depth caps how many sequential splits a sample can pass "
    "through, directly limiting how finely the tree can partition the feature space "
    "(fewer, coarser regions => higher bias, lower variance). min_samples_split=20 "
    "blocks any split that would act on fewer than 20 samples, which stops the tree "
    "from carving out tiny leaves that only fit noise in a handful of rows."
)

dt_comparison = pd.DataFrame({
    "Model": ["Unconstrained Tree", "Controlled Tree (depth=5, min_split=20)"],
    "Train Accuracy": [dt_base_train_acc, dt_ctrl_train_acc],
    "Test Accuracy": [dt_base_test_acc, dt_ctrl_test_acc],
    "Train-Test Gap": [dt_base_train_acc - dt_base_test_acc, dt_ctrl_train_acc - dt_ctrl_test_acc],
})
print("\nComparison table:")
print(dt_comparison.to_string(index=False))
dt_comparison.to_csv(f"{RESULTS_DIR}/decision_tree_comparison.csv", index=False)
print(f"Saved: {RESULTS_DIR}/decision_tree_comparison.csv")

# =====================================================================
# TASK 3 — GINI vs ENTROPY
# =====================================================================
print("\n" + "=" * 70)
print("TASK 3: GINI vs ENTROPY (both max_depth=5)")
print("=" * 70)

dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
gini_test_acc = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
dt_entropy.fit(X_train_scaled, y_clf_train)
entropy_test_acc = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

gini_entropy_table = pd.DataFrame({
    "Criterion": ["gini", "entropy"],
    "Test Accuracy": [gini_test_acc, entropy_test_acc],
})
print(gini_entropy_table.to_string(index=False))
gini_entropy_table.to_csv(f"{RESULTS_DIR}/gini_entropy_comparison.csv", index=False)
print(f"Saved: {RESULTS_DIR}/gini_entropy_comparison.csv")
print(
    "NOTE (for README): Gini impurity = 1 - sum(p_i^2); Entropy = -sum(p_i * log2(p_i)), "
    "where p_i is the proportion of class i's samples at a node. A node with Gini = 0 "
    "is 'pure' -- every sample that reached it belongs to the same class, so no further "
    "split could improve classification purity at that node."
)

# =====================================================================
# TASK 4 — RANDOM FOREST
# =====================================================================
print("\n" + "=" * 70)
print("TASK 4: RANDOM FOREST (n_estimators=100, max_depth=10)")
print("=" * 70)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)

rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_test_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"Random Forest -> Train accuracy: {rf_train_acc:.4f}   Test accuracy: {rf_test_acc:.4f}   "
      f"Test ROC-AUC: {rf_test_auc:.4f}")

importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
top5_importance = importances.head(5)
print("\nTop 5 features by Random Forest importance:")
print(top5_importance.to_string())

importances.rename("importance").reset_index().rename(columns={"index": "feature"}).to_csv(
    f"{RESULTS_DIR}/random_forest_feature_importance.csv", index=False
)
print(f"Saved: {RESULTS_DIR}/random_forest_feature_importance.csv")

plt.figure(figsize=(8, 5))
top5_importance.sort_values().plot.barh(color="teal")
plt.title("Random Forest — Top 5 Feature Importances")
plt.xlabel("Mean decrease in Gini impurity")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_importance_top5.png", dpi=110)
plt.close()
print(f"Saved: {FIG_DIR}/feature_importance_top5.png")

print(
    "NOTE (for README - feature importance): Random Forest importance for a feature is "
    "the average reduction in Gini impurity produced by splits on that feature, averaged "
    "over every split that uses it across all trees in the forest. This differs from a "
    "linear regression coefficient, which measures the size and direction of a feature's "
    "effect on a continuous linear combination assuming a linear relationship -- Random "
    "Forest importance instead reflects how useful a feature was for reducing classification "
    "impurity across many non-linear, tree-structured decision boundaries, with no sign "
    "or linear-effect-size interpretation attached."
)
print(
    "NOTE (for README - bagging paragraph): Random Forest builds many decision trees, each "
    "trained on a bootstrap sample (random sampling with replacement) of the training rows, "
    "so every tree sees a slightly different training set. In addition, at each split only a "
    "random subset of roughly sqrt(number_of_features) candidate features is considered, "
    "which decorrelates the trees from one another (otherwise most trees would keep choosing "
    "the same dominant feature at the top splits). Averaging (for classification, majority-"
    "voting/probability-averaging) the predictions of many such decorrelated, individually "
    "high-variance trees cancels out each tree's idiosyncratic overfitting, producing an "
    "ensemble whose overall variance is much lower than any single deep decision tree while "
    "keeping bias low."
)

# =====================================================================
# TASK 4a — GRADIENT BOOSTING
# =====================================================================
print("\n" + "=" * 70)
print("TASK 4a: GRADIENT BOOSTING (n_estimators=100, lr=0.1, max_depth=3)")
print("=" * 70)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_scaled, y_clf_train)

gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_test_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
print(f"Gradient Boosting -> Train accuracy: {gb_train_acc:.4f}   Test accuracy: {gb_test_acc:.4f}   "
      f"Test ROC-AUC: {gb_test_auc:.4f}")

# =====================================================================
# TASK 4b — FEATURE ABLATION STUDY
# =====================================================================
print("\n" + "=" * 70)
print("TASK 4b: FEATURE ABLATION (drop 5 lowest-importance features)")
print("=" * 70)

lowest5 = importances.sort_values(ascending=True).head(5)
lowest5_features = lowest5.index.tolist()
print(f"5 lowest-importance features: {lowest5_features}")

X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
keep_cols = [c for c in feature_names if c not in lowest5_features]

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_df[keep_cols].values, y_clf_train)
reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_df[keep_cols].values)[:, 1])

ablation_table = pd.DataFrame({
    "Model": ["Full Random Forest (all features)", "Reduced Random Forest (5 lowest-importance features removed)"],
    "Test ROC-AUC": [rf_test_auc, reduced_auc],
})
print(ablation_table.to_string(index=False))
ablation_table.to_csv(f"{RESULTS_DIR}/feature_ablation.csv", index=False)
print(f"Saved: {RESULTS_DIR}/feature_ablation.csv")

auc_delta = rf_test_auc - reduced_auc
were_uninformative = reduced_auc >= rf_test_auc - 0.002
print(f"AUC delta (full - reduced): {auc_delta:.4f}")
if were_uninformative:
    ablation_verdict = (
        "The removed features appear genuinely uninformative -- AUC held steady or "
        "improved without them, suggesting they mostly contributed noise."
    )
else:
    ablation_verdict = (
        "The removed features were contributing real signal -- AUC dropped "
        "meaningfully without them."
    )
print(
    f"NOTE (for README): {ablation_verdict} For production, dropping low-importance "
    "features lowers inference cost, feature-pipeline complexity, and long-term "
    "maintenance burden (fewer upstream data sources to keep alive), but that "
    "simplification is only worth deploying if the resulting AUC degradation stays "
    "below whatever tolerance the business case can accept."
)

# =====================================================================
# TASK 5 — CROSS-VALIDATED COMPARISON
# =====================================================================
print("\n" + "=" * 70)
print("TASK 5: CROSS-VALIDATED COMPARISON (5-fold, ROC-AUC)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

cv_rows = []
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_rows.append((name, scores.mean(), scores.std()))
    print(f"{name:30s} -> mean AUC: {scores.mean():.4f}   std AUC: {scores.std():.4f}")

cv_table = pd.DataFrame(cv_rows, columns=["Model", "Mean CV AUC", "Std CV AUC"])
cv_table.to_csv(f"{RESULTS_DIR}/cv_comparison.csv", index=False)
print(f"Saved: {RESULTS_DIR}/cv_comparison.csv")
print(
    "NOTE (for README): a single train-test split gives one noisy sample of how the model "
    "performs on unseen data -- the estimate depends heavily on which rows happened to land "
    "in the test set. 5-fold cross-validation instead trains and evaluates the model 5 times "
    "on 5 different train/validation partitions and averages the results, giving both a more "
    "stable mean-performance estimate and a standard deviation that quantifies how sensitive "
    "the model is to which rows it is trained/evaluated on."
)

# =====================================================================
# TASK 6 — HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# =====================================================================
print("\n" + "=" * 70)
print("TASK 6: GRIDSEARCHCV — RANDOM FOREST PIPELINE")
print("=" * 70)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42),
)

grid_search = GridSearchCV(
    pipeline, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1,
)
grid_search.fit(X_train, y_clf_train)  # unscaled -- pipeline handles imputation + scaling

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
n_fits = n_configs * 5
print(f"Total distinct hyperparameter configurations: {n_configs}   Total model fits (x5 folds): {n_fits}")
print(
    "NOTE (for README): Grid Search exhaustively evaluates every combination in the grid, "
    "guaranteeing the best combination *within the specified grid* is found, but its cost "
    "grows multiplicatively with the number of hyperparameters and values tried. Randomized "
    "Search instead samples a fixed number of random combinations from the specified "
    "distributions, trading the guarantee of exhaustive coverage for a cost that stays "
    "constant regardless of how large the search space is -- usually finding a near-optimal "
    "combination far faster when the grid is large."
)

best_pipeline = grid_search.best_estimator_

gridsearch_results = pd.DataFrame([{
    "best_params": str(grid_search.best_params_),
    "best_cv_auc": grid_search.best_score_,
    "n_configurations": n_configs,
    "n_total_fits": n_fits,
}])
gridsearch_results.to_csv(f"{RESULTS_DIR}/gridsearch_results.csv", index=False)
print(f"Saved: {RESULTS_DIR}/gridsearch_results.csv")

# =====================================================================
# TASK 7 — MANUAL LEARNING CURVE
# =====================================================================
print("\n" + "=" * 70)
print("TASK 7: MANUAL LEARNING CURVE (best pipeline, 20%-100% of training data)")
print("=" * 70)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_rows = []
for f in fractions:
    n_rows = int(f * len(X_train))
    X_sub = X_train.iloc[:n_rows]
    y_sub = y_clf_train.iloc[:n_rows]

    # Fresh clone of the tuned pipeline, refit on this subset only
    from sklearn.base import clone
    pipe_f = clone(best_pipeline)
    pipe_f.fit(X_sub, y_sub)

    train_auc = roc_auc_score(y_sub, pipe_f.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, pipe_f.predict_proba(X_test)[:, 1])
    learning_rows.append((f, n_rows, train_auc, test_auc))
    print(f"Training fraction: {f:.1f} ({n_rows} rows) -> Training AUC: {train_auc:.4f}   Test AUC: {test_auc:.4f}")

learning_table = pd.DataFrame(learning_rows, columns=["Training fraction", "N rows", "Training AUC", "Test AUC"])
print("\nLearning curve table:")
print(learning_table.to_string(index=False))
learning_table.to_csv(f"{RESULTS_DIR}/learning_curve.csv", index=False)
print(f"Saved: {RESULTS_DIR}/learning_curve.csv")

plt.figure(figsize=(8, 5))
plt.plot(learning_table["Training fraction"], learning_table["Training AUC"], marker="o", label="Training AUC")
plt.plot(learning_table["Training fraction"], learning_table["Test AUC"], marker="o", label="Test AUC")
plt.xlabel("Training fraction")
plt.ylabel("ROC-AUC")
plt.title("Manual Learning Curve — Tuned Random Forest Pipeline")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/learning_curve.png", dpi=110)
plt.close()
print(f"Saved: {FIG_DIR}/learning_curve.png")

train_auc_trend_down = learning_table["Training AUC"].iloc[-1] < learning_table["Training AUC"].iloc[0]
test_auc_trend_up = learning_table["Test AUC"].iloc[-1] > learning_table["Test AUC"].iloc[0]
print(f"Does Training AUC decrease as the training set grows? {train_auc_trend_down}")
print(f"Does Test AUC increase with more training data? {test_auc_trend_up}")
print(
    "NOTE (for README): if Test AUC is still rising noticeably at 100% of the training data, "
    "the model is likely data-limited -- collecting more labeled rows would probably keep "
    "improving it. If Test AUC has flattened out well before 100%, the model has likely hit "
    "its capacity ceiling for the current feature set, and more data alone would not help; a "
    "richer feature set or a different model family would be needed instead."
)

# =====================================================================
# TASK 8 — SERIALIZE THE BEST MODEL
# =====================================================================
print("\n" + "=" * 70)
print("TASK 8: SERIALIZE BEST MODEL")
print("=" * 70)

joblib.dump(best_pipeline, "best_model.pkl")
print("Saved: best_model.pkl")

# --- reload-and-predict sanity check (>=5 lines, runs without errors) ---
loaded_model = joblib.load("best_model.pkl")
hand_crafted_rows = X_test.iloc[:2]
predictions = loaded_model.predict(hand_crafted_rows)
probabilities = loaded_model.predict_proba(hand_crafted_rows)[:, 1]
print(f"Reload-and-predict check -> predictions: {predictions}   probabilities: {probabilities}")

# =====================================================================
# TASK 9 — SUMMARY COMPARISON TABLE
# =====================================================================
print("\n" + "=" * 70)
print("TASK 9: FINAL SUMMARY COMPARISON TABLE")
print("=" * 70)

cv_lookup = cv_table.set_index("Model")

final_summary = pd.DataFrame([
    {"Model": "Logistic Regression",
     "Mean CV AUC": cv_lookup.loc["Logistic Regression", "Mean CV AUC"],
     "Std CV AUC": cv_lookup.loc["Logistic Regression", "Std CV AUC"],
     "Test AUC": np.nan},
    {"Model": "Decision Tree (depth=5)",
     "Mean CV AUC": cv_lookup.loc["Decision Tree (depth=5)", "Mean CV AUC"],
     "Std CV AUC": cv_lookup.loc["Decision Tree (depth=5)", "Std CV AUC"],
     "Test AUC": np.nan},
    {"Model": "Random Forest (untuned)",
     "Mean CV AUC": cv_lookup.loc["Random Forest", "Mean CV AUC"],
     "Std CV AUC": cv_lookup.loc["Random Forest", "Std CV AUC"],
     "Test AUC": rf_test_auc},
    {"Model": "Gradient Boosting",
     "Mean CV AUC": cv_lookup.loc["Gradient Boosting", "Mean CV AUC"],
     "Std CV AUC": cv_lookup.loc["Gradient Boosting", "Std CV AUC"],
     "Test AUC": gb_test_auc},
    {"Model": "Random Forest (GridSearchCV-tuned pipeline)",
     "Mean CV AUC": grid_search.best_score_,
     "Std CV AUC": np.nan,
     "Test AUC": roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])},
])
print(final_summary.to_string(index=False))
final_summary.to_csv(f"{RESULTS_DIR}/final_model_summary.csv", index=False)
print(f"Saved: {RESULTS_DIR}/final_model_summary.csv")

best_row = final_summary.loc[final_summary["Test AUC"].idxmax()]
print(
    f"\nRECOMMENDATION (for README): '{best_row['Model']}' achieved the highest test-set AUC "
    f"({best_row['Test AUC']:.4f}) among the models with a directly comparable test-set score. "
    "It is recommended as the model to deploy because it combines strong discriminative "
    "performance with the variance-reduction benefits of ensembling (many decorrelated trees "
    "averaged together), and -- via the GridSearchCV pipeline -- its hyperparameters were "
    "selected using cross-validated AUC rather than a single train/test split, reducing the "
    "risk that the reported performance is an artifact of one particular data split. It is also "
    "packaged end-to-end (imputation + scaling + model) as a single serialized pipeline, making "
    "it straightforward to reload and serve in production without re-implementing preprocessing."
)

print("\nDONE. All figures in ./figures/, all tables in ./results/, model in ./best_model.pkl")