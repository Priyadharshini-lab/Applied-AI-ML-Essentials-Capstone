# Part 1 — Data Cleaning & EDA: Solar Power Plant (Plant 1)

## Dataset description
Two source files, merged on `DATE_TIME`:
- **Plant_1_Generation_Data.csv** — 15-minute inverter readings: `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD` per `SOURCE_KEY` (inverter).
- **Plant_1_Weather_Sensor_Data.csv** — one plant-wide weather reading per timestamp: `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`.

Generation timestamps are `DD-MM-YYYY`, weather timestamps are `YYYY-MM-DD` — each is parsed with the correct format *before* merging, otherwise `pd.to_datetime`'s auto-inference silently produces wrong dates for one of the two files and the merge key stops matching correctly.

## Why median instead of mean for imputation
`IRRADIATION` and `DC_POWER` are both **right-skewed** (see `results/skewness.csv`) — most 15-minute readings are near-zero (night, dawn, dusk, cloud cover) with a long tail of high-output daytime readings. The mean is pulled toward that tail, so filling gaps with the mean would systematically overstate typical output. The median is robust to that skew and represents a more realistic "typical" reading, which is why `results/imputation_comparison.csv` shows mean consistently higher than median for both top-skew columns.

## Why category dtype
`PLANT_ID` and `SOURCE_KEY_GEN`/`SOURCE_KEY_WX` are repeated identifiers with very few unique values relative to row count (1 plant ID, ~22 inverter IDs, across tens of thousands of rows). Storing them as `category` keeps one small integer code per unique value instead of repeating the full string on every row, which is what produces the memory reduction in `results/memory_usage.csv`.

## Highest-skew column & interpretation
See `results/skewness.csv` for the exact ranked values; typically `DC_POWER` or `IRRADIATION` tops the list. **Positive skew** means a long right tail — many small/zero values (nighttime) and a few very large ones (peak midday sun) — consistent with solar output's on/off daily cycle. A **negative skew** would mean a long left tail instead (rare here, since power can't go far below zero but can spike well above its typical value).

## IQR interpretation
`results/iqr_analysis.csv` gives Q1, Q3, IQR, and bounds for `IRRADIATION` and `DC_POWER`. Values beyond the upper bound aren't necessarily errors — they're the genuine high-output peak-sun readings the IQR rule flags simply because the bulk of readings (night/low-light) sit near zero. That's why Part 1 reports these counts but does **not** delete them: doing so would remove exactly the "normal, sunny" rows later parts need to learn what expected output looks like.

## Scatter plot interpretation
`4_scatter_irradiation_dcpower.png` shows a strong, roughly linear positive relationship between irradiation and DC power — expected, since irradiation is the primary physical driver of panel output. Points that fall well below the main trend (high irradiation, low power) are visually the candidates for "underperformance" investigated further in Part 2.

## Boxplot interpretation
`5_box_acpower_dayperiod.png` splits `AC_POWER` by `DAY_PERIOD` (Night/Morning/Afternoon/Evening). Night is near-zero with almost no spread; Afternoon has the highest median and widest spread, consistent with peak sun hours also having the most day-to-day/weather variability.

## Heatmap / Pearson correlation
`6_heatmap_pearson.png` and `results/pearson_matrix.csv`. The strongest pair is `IRRADIATION`↔`DC_POWER`/`AC_POWER` (see the printed "Highest absolute correlation pair"), confirming irradiation as the dominant driver. `AMBIENT_TEMPERATURE`↔`MODULE_TEMPERATURE` is also strongly correlated, as expected physically.

## Spearman vs Pearson
`results/spearman_matrix.csv` and `results/spearman_difference.csv`. Pearson captures **linear** association; Spearman captures **monotonic rank** association. Where the two differ most (top 3 pairs saved), it signals a real but non-linear relationship — e.g. power saturating at high irradiation, or a threshold effect — that a linear model would under-credit. **This is why Spearman is worth checking again in Part 2**: if a feature's Spearman correlation with the target is much higher than its Pearson correlation, a plain linear/logistic model may need a transformed feature (or a non-linear model) to fully capture that relationship — which is part of the motivation for using a RandomForest-based expected-power model in Part 2 rather than relying on linear regression alone.

## Grouped aggregation (by day period)
`results/grouped_aggregation.csv`. Highest mean and highest std both fall in **Afternoon** in this dataset — peak output also brings peak variability (weather, cloud cover). The mean ratio (highest/lowest non-zero group mean) quantifies how many times larger midday output is than the smallest non-trivial period average.

## Repository structure
```
part1/
    README.md
    part1_eda.py
    cleaned_data.csv
    figures/
        1_line_ac_power.png
        2_bar_mean_dc_power.png
        3_hist_<top_skew_col>.png
        4_scatter_irradiation_dcpower.png
        5_box_acpower_dayperiod.png
        6_heatmap_pearson.png
    results/
        null_analysis.csv
        memory_usage.csv
        descriptive_statistics.csv
        skewness.csv
        iqr_analysis.csv
        pearson_matrix.csv
        imputation_comparison.csv
        spearman_matrix.csv
        spearman_difference.csv
        grouped_aggregation.csv
```