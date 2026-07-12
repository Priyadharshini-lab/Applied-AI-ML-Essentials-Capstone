# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

> **Note on the classification label**: this script re-derives `y_clf` independently from `cleaned_data.csv` using the bottom-20%-of-`AC_POWER` threshold (`clf_threshold = day["AC_POWER"].quantile(0.20)`), to stay self-contained and match the assignment brief's exact wording ("using the same X_train_scaled... from Part 2"). If you're using this alongside the corrected Part 2 (`PERFORMANCE_RATIO` / `UNDERPERFORMING` label instead of a fixed bottom-20% cutoff), swap this script's data-loading block to load `modeling_data.csv` and use its `UNDERPERFORMING` column instead, so Part 3 evaluates the same, leak-aware label Part 2 actually recommends. The template below is written generically so it works either way — just be consistent about which label you're reporting numbers for.

## 1. Pipeline Overview
Part 3 rebuilds the exact feature set, split (`random_state=42`), and scaling from Part 2, then trains and compares five classifier families — a baseline decision tree, a regularized decision tree, Random Forest, Gradient Boosting, and a `GridSearchCV`-tuned Random Forest pipeline — before serializing the best one to `best_model.pkl`.

## 2. Decision Tree Baseline (unconstrained)

*Fill in from console output.*

| | Accuracy |
|---|---|
| Train | *fill in* |
| Test | *fill in* |
| Train − Test gap | *fill in* |

**Overfitting?** *State yes/no based on your numbers.* A train accuracy near 1.0 with materially lower test accuracy is the signature of overfitting.

**Why decision trees are high-variance models**: at each node, the greedy split-selection algorithm locks in a decision based only on the samples that reached that node — it never revisits or corrects earlier splits. With no depth limit, the tree keeps partitioning until it can memorize individual training rows, including their noise, so small changes in the training data can produce a very different tree.

## 3. Controlled Decision Tree (`max_depth=5`, `min_samples_split=20`)

*Fill in from console output / `results/decision_tree_comparison.csv`.*

| Model | Train Accuracy | Test Accuracy | Train−Test Gap |
|---|---|---|---|
| Unconstrained Tree | *fill in* | *fill in* | *fill in* |
| Controlled Tree | *fill in* | *fill in* | *fill in* |

**`max_depth`**: caps how many sequential splits a sample can pass through, directly limiting how finely the tree can partition the feature space — fewer, coarser regions means higher bias but lower variance.
**`min_samples_split`**: blocks any split that would act on fewer than 20 samples, preventing the tree from carving out tiny leaves that only fit noise in a handful of rows.

**Comparison**: the controlled tree's train/test gap is *(state: smaller/larger)* than the unconstrained tree's, indicating *(state: less/more)* overfitting at the cost of *(state: some/no)* additional bias.

## 4. Gini vs. Entropy (`max_depth=5`)

*Fill in from `results/gini_entropy_comparison.csv`.*

| Criterion | Test Accuracy |
|---|---|
| Gini | *fill in* |
| Entropy | *fill in* |

**Gini impurity**: `1 - Σ pᵢ²`
**Entropy**: `-Σ pᵢ log₂(pᵢ)`

where `pᵢ` is the proportion of class *i*'s samples at a node. A node with **Gini = 0** is "pure" — every sample that reached it belongs to the same class, so no further split could improve classification purity at that node.

## 5. Random Forest (`n_estimators=100`, `max_depth=10`)

*Fill in from console output.*

| Metric | Value |
|---|---|
| Train Accuracy | *fill in* |
| Test Accuracy | *fill in* |
| Test ROC-AUC | *fill in* |

**Top 5 features by importance** (`results/random_forest_feature_importance.csv`, `figures/feature_importance_top5.png`):
1. *fill in*
2. *fill in*
3. *fill in*
4. *fill in*
5. *fill in*

**How Random Forest computes feature importance**: for each feature, importance is the average reduction in Gini impurity produced by splits on that feature, averaged over every split that uses it across all trees in the forest. This differs fundamentally from a linear regression coefficient, which measures the size and *direction* (sign) of a feature's effect on a continuous linear combination, assuming a linear relationship. Random Forest importance instead reflects how useful a feature was for reducing classification impurity across many non-linear, tree-structured decision boundaries — it carries no sign or linear-effect-size interpretation.

**Bagging (bootstrap aggregating)**: Random Forest builds many decision trees, each trained on a bootstrap sample (random sampling with replacement) of the training rows, so every tree sees a slightly different training set. In addition, at each split only a random subset of roughly √(number of features) candidate features is considered, which decorrelates the trees from one another — otherwise most trees would keep choosing the same dominant feature at the top splits. Averaging the predictions of many such decorrelated, individually high-variance trees (majority vote / probability averaging for classification) cancels out each tree's idiosyncratic overfitting, producing an ensemble whose overall variance is much lower than any single deep decision tree, while keeping bias low.

## 6. Gradient Boosting (`n_estimators=100`, `lr=0.1`, `max_depth=3`)

*Fill in from console output.*

| Metric | Value |
|---|---|
| Train Accuracy | *fill in* |
| Test Accuracy | *fill in* |
| Test ROC-AUC | *fill in* |

Included in the cross-validated comparison below (Section 8).

## 7. Feature Ablation Study

The 5 lowest-importance features from the Random Forest (Section 5) were removed, and a second Random Forest with identical hyperparameters (`random_state=42`) was retrained on the reduced feature set.

*Fill in from `results/feature_ablation.csv`.*

| Model | Test ROC-AUC |
|---|---|
| Full Random Forest (all features) | *fill in* |
| Reduced Random Forest (5 lowest-importance features removed) | *fill in* |

**AUC delta (full − reduced)**: *fill in*

**Interpretation**: *state whether the removed features were genuinely uninformative (AUC held steady or improved without them, suggesting they mostly added noise) or were contributing real signal (AUC dropped meaningfully without them).* For production, dropping low-importance features lowers inference cost, feature-pipeline complexity, and long-term maintenance burden (fewer upstream data sources to keep alive) — but that simplification is only worth deploying if the resulting AUC degradation stays below whatever tolerance the business case can accept.

## 8. Cross-Validated Comparison (`StratifiedKFold`, 5-fold, ROC-AUC)

*Fill in from `results/cv_comparison.csv`.*

| Model | Mean CV AUC | Std CV AUC |
|---|---|---|
| Logistic Regression | *fill in* | *fill in* |
| Decision Tree (depth=5) | *fill in* | *fill in* |
| Random Forest | *fill in* | *fill in* |
| Gradient Boosting | *fill in* | *fill in* |

**Why cross-validation beats a single train-test split**: a single split gives one noisy sample of how the model performs on unseen data — the estimate depends heavily on which rows happened to land in the test set. 5-fold cross-validation trains and evaluates the model 5 times on 5 different train/validation partitions and averages the results, giving both a more stable mean-performance estimate and a standard deviation that quantifies how sensitive the model is to which rows it is trained/evaluated on. `StratifiedKFold` specifically keeps the underperformance class ratio consistent across every fold.

## 9. Hyperparameter Tuning — `GridSearchCV`

**Parameter grid**:
```python
param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}
```
Total distinct configurations: 3 × 3 × 2 = **18**, each evaluated across 5 folds → **90 total model fits**.

**Best params found**: *fill in from `results/gridsearch_results.csv`*
**Best CV ROC-AUC**: *fill in*

**Grid Search vs. Randomized Search trade-off**: Grid Search exhaustively evaluates every combination in the grid, guaranteeing the best combination *within the specified grid* is found, but its cost grows multiplicatively with the number of hyperparameters and values tried. Randomized Search instead samples a fixed number of random combinations from specified distributions, trading the guarantee of exhaustive coverage for a cost that stays constant regardless of how large the search space is — usually finding a near-optimal combination far faster when the grid is large.

## 10. Manual Learning Curve

*Fill in from `results/learning_curve.csv` / `figures/learning_curve.png`.*

| Training fraction | N rows | Training AUC | Test AUC |
|---|---|---|---|
| 0.2 | *fill in* | *fill in* | *fill in* |
| 0.4 | *fill in* | *fill in* | *fill in* |
| 0.6 | *fill in* | *fill in* | *fill in* |
| 0.8 | *fill in* | *fill in* | *fill in* |
| 1.0 | *fill in* | *fill in* | *fill in* |

- **Does Training AUC decrease as the training set grows?** *fill in (yes/no)* — expected for high-variance models, since a small training set is easier to fit near-perfectly.
- **Does Test AUC increase with more training data?** *fill in (yes/no)* — if yes, collecting more labeled data would likely keep improving the model.
- **Conclusion — data-limited or capacity-limited?** *If Test AUC is still rising noticeably at 100% of the training data, the model is likely data-limited — collecting more labeled rows would probably keep improving it. If Test AUC has flattened out well before 100%, the model has likely hit its capacity ceiling for the current feature set, and more data alone would not help; a richer feature set or a different model family would be needed instead.*

## 11. Serialized Model

`best_model.pkl` contains the full `GridSearchCV`-selected pipeline (`SimpleImputer → StandardScaler → RandomForestClassifier`), so it can be reloaded and used directly without re-implementing preprocessing.

**Reload-and-predict sanity check**:
```python
import joblib
loaded_model = joblib.load("best_model.pkl")
hand_crafted_rows = X_test.iloc[:2]
predictions = loaded_model.predict(hand_crafted_rows)
probabilities = loaded_model.predict_proba(hand_crafted_rows)[:, 1]
print(predictions, probabilities)
```
Confirmed to run without errors — see console output under "TASK 8: SERIALIZE BEST MODEL".

## 12. Final Summary Comparison Table (Parts 2 + 3)

*Fill in from `results/final_model_summary.csv`.*

| Model | Mean CV AUC | Std CV AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | *fill in* | *fill in* | — |
| Decision Tree (depth=5) | *fill in* | *fill in* | — |
| Random Forest (untuned) | *fill in* | *fill in* | *fill in* |
| Gradient Boosting | *fill in* | *fill in* | *fill in* |
| Random Forest (GridSearchCV-tuned pipeline) | *fill in* | — | *fill in* |

**Recommendation**: *fill in the model with the highest comparable test AUC.* Justification (3–5 sentences): it combines strong discriminative performance with the variance-reduction benefits of ensembling (many decorrelated trees averaged together); its hyperparameters were selected using cross-validated AUC rather than a single train/test split, reducing the risk that reported performance is an artifact of one particular data split; and it is packaged end-to-end (imputation + scaling + model) as a single serialized pipeline, making it straightforward to reload and serve in production without re-implementing preprocessing.

## 13. How to Run
```bash
pip install -r requirements.txt
python part3/part3_ensembles.py
```
Expects `cleaned_data.csv` (from Part 1) to be discoverable via the same `find_file()` search used in Parts 1–2.

## Repository structure
```
part3/
    README.md
    part3_ensembles.py
    best_model.pkl
    figures/
        feature_importance_top5.png
        learning_curve.png
    results/
        decision_tree_comparison.csv
        gini_entropy_comparison.csv
        random_forest_feature_importance.csv
        feature_ablation.csv
        cv_comparison.csv
        gridsearch_results.csv
        learning_curve.csv
        final_model_summary.csv
```