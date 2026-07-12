"""
=====================================================================
 PART 1 — Data Acquisition, Cleaning, and Exploratory Data Analysis
 Dataset : Solar Power Plant Generation + Weather Sensor Data (Plant 1)
=====================================================================
Run with:  python3 part1_eda.py
Produces:
  - console output for every task (redirected to eda_log.txt as well)
  - figures/*.png  (5 required plots + heatmap)
  - cleaned_data.csv
"""


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


FIG_DIR = "figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)


# =====================================================================
# TASK 1 — LOAD DATA, INSPECT SHAPE / DTYPES / HEAD
# =====================================================================
print("=" * 70)
print("TASK 1: LOAD & INSPECT")
print("=" * 70)


gen = pd.read_csv("Plant_1_Generation_Data.csv")
wx  = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")


# Generation timestamps are DD-MM-YYYY, weather timestamps are YYYY-MM-DD -> parse each correctly
gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True)
wx["DATE_TIME"]  = pd.to_datetime(wx["DATE_TIME"])


# One weather sensor covers the whole plant, so merge on DATE_TIME (drop wx's own PLANT_ID to avoid a duplicate column)
df = gen.merge(wx.drop(columns=["PLANT_ID"]), on="DATE_TIME", how="left",
               suffixes=("_GEN", "_WX"))


print("\nFirst five rows:")
print(df.head())


print("\nColumn data types:")
print(df.dtypes)


print(f"\nDataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")


# =====================================================================
# TASK 2 — NULL VALUE ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("TASK 2: NULL VALUE ANALYSIS")
print("=" * 70)


null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / df.shape[0]) * 100
null_table = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct.round(3)})
print(null_table)


high_null_cols = null_table[null_table["null_pct"] > 20].index.tolist()
print(f"\nColumns exceeding 20% null rate: {high_null_cols if high_null_cols else 'NONE'}")


numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    if col not in high_null_cols and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())


print("\nNulls remaining after median fill (numeric cols only):")
print(df.isnull().sum())


# =====================================================================
# TASK 3 — DUPLICATE DETECTION AND REMOVAL
# =====================================================================
print("\n" + "=" * 70)
print("TASK 3: DUPLICATE DETECTION & REMOVAL")
print("=" * 70)


dup_count = df.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")


null_pct_before = (df.isnull().sum() / df.shape[0]) * 100
df = df.drop_duplicates()
null_pct_after = (df.isnull().sum() / df.shape[0]) * 100


print(f"Rows removed: {dup_count}")
print(f"Shape after de-duplication: {df.shape}")
changed = (null_pct_before.round(4) != null_pct_after.round(4)).any()
print(f"Did null percentages change after removing duplicates? {changed}")


# =====================================================================
# TASK 4 — DATA TYPE CORRECTION
# =====================================================================
print("\n" + "=" * 70)
print("TASK 4: DATA TYPE CORRECTION")
print("=" * 70)


mem_before = df.memory_usage(deep=True).sum()
print(f"Memory usage BEFORE conversion: {mem_before:,} bytes")


# PLANT_ID is stored as int64 but is really a categorical identifier (only 1 unique value per plant)
df["PLANT_ID"] = df["PLANT_ID"].astype("category")
# SOURCE_KEY_GEN (inverter id) is a repetitive string column -> category dtype
df["SOURCE_KEY_GEN"] = df["SOURCE_KEY_GEN"].astype("category")
df["SOURCE_KEY_WX"] = df["SOURCE_KEY_WX"].astype("category")


# Demonstrate numeric coercion pattern required by the task (defensive, in case of stray strings)
df["DC_POWER"] = pd.to_numeric(df["DC_POWER"], errors="coerce")


mem_after = df.memory_usage(deep=True).sum()
print(f"Memory usage AFTER conversion:  {mem_after:,} bytes")
print(f"Memory saved: {mem_before - mem_after:,} bytes "
      f"({(1 - mem_after / mem_before) * 100:.2f}% reduction)")


# =====================================================================
# TASK 5 — DESCRIPTIVE STATISTICS AND SKEWNESS
# =====================================================================
print("\n" + "=" * 70)
print("TASK 5: DESCRIPTIVE STATISTICS & SKEWNESS")
print("=" * 70)


numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
print(df[numeric_cols].describe())


skew_vals = df[numeric_cols].skew().sort_values(key=lambda x: x.abs(), ascending=False)
print("\nSkewness per numeric column (sorted by |skew|):")
print(skew_vals)


top_skew_col = skew_vals.index[0]
print(f"\nColumn with highest absolute skewness: {top_skew_col} (skew = {skew_vals.iloc[0]:.3f})")


# =====================================================================
# TASK 6 — OUTLIER DETECTION WITH IQR
# =====================================================================
print("\n" + "=" * 70)
print("TASK 6: OUTLIER DETECTION (IQR METHOD)")
print("=" * 70)


iqr_cols = ["IRRADIATION", "DC_POWER"]
iqr_results = {}
for col in iqr_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    iqr_results[col] = dict(Q1=Q1, Q3=Q3, IQR=IQR, lower=lower, upper=upper, n_outliers=n_outliers)
    print(f"\n{col}:")
    print(f"  Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
    print(f"  Lower bound={lower:.2f}  Upper bound={upper:.2f}")
    print(f"  Rows outside bounds: {n_outliers} ({n_outliers/df.shape[0]*100:.2f}% of rows)")


# =====================================================================
# TASK 7 — VISUALIZATIONS (5 required types)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 7: VISUALIZATIONS")
print("=" * 70)


# 7.1 Line plot - AC_POWER over time for a single inverter, sorted by time
sample_inv = df["SOURCE_KEY_GEN"].cat.categories[0]
line_df = df[df["SOURCE_KEY_GEN"] == sample_inv].sort_values("DATE_TIME")
plt.figure(figsize=(12, 5))
plt.plot(line_df["DATE_TIME"], line_df["AC_POWER"], color="darkorange", linewidth=0.8)
plt.title(f"AC Power Over Time — Inverter {sample_inv}")
plt.xlabel("Date-Time")
plt.ylabel("AC Power (kW)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/1_line_ac_power.png", dpi=110)
plt.close()


# 7.2 Bar chart - mean DC_POWER by inverter (categorical)
plt.figure(figsize=(12, 5))
df.groupby("SOURCE_KEY_GEN", observed=True)["DC_POWER"].mean().plot.bar(color="steelblue")
plt.title("Mean DC Power by Inverter")
plt.xlabel("Inverter (SOURCE_KEY)")
plt.ylabel("Mean DC Power (kW)")
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/2_bar_mean_dc_power.png", dpi=110)
plt.close()


# 7.3 Histogram of most skewed column
plt.figure(figsize=(8, 5))
sns.histplot(df[top_skew_col], bins=20, kde=True, color="seagreen")
plt.title(f"Distribution of {top_skew_col} (skew = {skew_vals.iloc[0]:.2f})")
plt.xlabel(top_skew_col)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/3_hist_{top_skew_col.lower()}.png", dpi=110)
plt.close()


# 7.4 Scatter plot - IRRADIATION vs DC_POWER (expected strong positive relationship)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=42),
                 x="IRRADIATION", y="DC_POWER", alpha=0.4, color="crimson")
plt.title("Irradiation vs DC Power")
plt.xlabel("Irradiation (W/m^2)")
plt.ylabel("DC Power (kW)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/4_scatter_irradiation_dcpower.png", dpi=110)
plt.close()


# 7.5 Box plot - AC_POWER split by a categorical grouping (daytime vs nighttime bucket, derived from hour)
df["HOUR"] = df["DATE_TIME"].dt.hour
df["DAY_PERIOD"] = pd.cut(df["HOUR"], bins=[-1, 5, 11, 17, 23],
                           labels=["Night", "Morning", "Afternoon", "Evening"])
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="DAY_PERIOD", y="AC_POWER", hue="DAY_PERIOD", palette="Set2", legend=False)
plt.title("AC Power Distribution by Day Period")
plt.xlabel("Day Period")
plt.ylabel("AC Power (kW)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/5_box_acpower_dayperiod.png", dpi=110)
plt.close()


print(f"All 5 plots saved to ./{FIG_DIR}/")


# =====================================================================
# TASK 8 — CORRELATION HEAT MAP (PEARSON)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 8: CORRELATION HEAT MAP (PEARSON)")
print("=" * 70)


corr_pearson = df[numeric_cols].corr(method="pearson")
print(corr_pearson.round(3))


plt.figure(figsize=(9, 7))
sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Pearson Correlation Heatmap — Numeric Features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/6_heatmap_pearson.png", dpi=110)
plt.close()


corr_abs = corr_pearson.abs().where(~np.eye(len(numeric_cols), dtype=bool))
max_pair = corr_abs.stack().idxmax()
max_val = corr_abs.stack().max()
print(f"\nHighest absolute correlation pair: {max_pair} = {max_val:.3f}")


# =====================================================================
# TASK 8a — IMPUTATION STRATEGY COMPARISON (mean vs median for top-2 skewed cols)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 8a: IMPUTATION STRATEGY COMPARISON (MEAN vs MEDIAN)")
print("=" * 70)


top2_skew_cols = skew_vals.index[:2].tolist()
for col in top2_skew_cols:
    mean_v = df[col].mean()
    median_v = df[col].median()
    print(f"{col}: mean = {mean_v:.3f} | median = {median_v:.3f} | skew = {skew_vals[col]:.3f}")
    df[col] = df[col].fillna(median_v)   # median chosen for skewed data (see README)


print("\nNulls remaining in these two columns after imputation:")
print(df[top2_skew_cols].isnull().sum())


# =====================================================================
# TASK 8b — SPEARMAN vs PEARSON COMPARISON
# =====================================================================
print("\n" + "=" * 70)
print("TASK 8b: SPEARMAN RANK CORRELATION vs PEARSON")
print("=" * 70)


corr_spearman = df[numeric_cols].corr(method="spearman")
print("\nSpearman matrix:")
print(corr_spearman.round(3))


diff = (corr_spearman - corr_pearson).abs()
diff_pairs = (
    diff.where(~np.eye(len(numeric_cols), dtype=bool))
    .stack()
    .sort_values(ascending=False)
)
# de-duplicate symmetric pairs (A,B) vs (B,A)
seen = set()
top3 = []
for (a, b), v in diff_pairs.items():
    key = frozenset([a, b])
    if key not in seen:
        seen.add(key)
        top3.append((a, b, corr_pearson.loc[a, b], corr_spearman.loc[a, b], v))
    if len(top3) == 3:
        break


diff_table = pd.DataFrame(top3, columns=["Var1", "Var2", "Pearson", "Spearman", "AbsDiff"])
print("\nTop 3 pairs by |Spearman - Pearson|:")
print(diff_table.to_string(index=False))


# =====================================================================
# TASK 8c — GROUPED AGGREGATION
# =====================================================================
print("\n" + "=" * 70)
print("TASK 8c: GROUPED AGGREGATION")
print("=" * 70)


group_col = "DAY_PERIOD"
value_col = "AC_POWER"
grouped = df.groupby(group_col, observed=True)[value_col].agg(["mean", "std", "count"])
print(grouped)


highest_mean_group = grouped["mean"].idxmax()
highest_std_group = grouped["std"].idxmax()
mean_ratio = grouped["mean"].max() / grouped[grouped["mean"] > 0]["mean"].min()


print(f"\nGroup with highest mean: {highest_mean_group}")
print(f"Group with highest std : {highest_std_group}")
print(f"Ratio of highest group mean to lowest (non-zero) group mean: {mean_ratio:.2f}")


# =====================================================================
# SAVE CLEANED DATASET
# =====================================================================
print("\n" + "=" * 70)
print("SAVING CLEANED DATASET")
print("=" * 70)


df.to_csv("cleaned_data.csv", index=False)
print(f"Saved cleaned_data.csv with shape {df.shape}")
print("\nDONE.")

