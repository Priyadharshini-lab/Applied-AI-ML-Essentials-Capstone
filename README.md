# Applied AI & ML Essentials Capstone

## Project Overview

This repository contains my submission for the **Applied AI & ML Essentials Capstone Project**, where I designed and implemented a complete end-to-end Artificial Intelligence and Machine Learning solution using a real-world solar power generation dataset.

The project demonstrates the complete AI lifecycle, including data acquisition, preprocessing, exploratory data analysis, supervised machine learning, ensemble learning, model optimization, and Large Language Model (LLM) integration with production-ready guardrails.

---

# Problem Statement

Solar power plants generate massive amounts of sensor data every day. Plant operators need to monitor the performance of multiple solar inverters continuously to identify underperforming units and understand the reasons behind performance degradation.

However,

- manually monitoring thousands of sensor readings is time-consuming,
- identifying abnormal power generation is difficult,
- traditional ML models only provide predictions without explanations,
- operators need human-readable insights to support maintenance decisions.

This project addresses these challenges by building an intelligent AI system that predicts expected solar power generation, detects underperforming operating conditions, compares multiple machine learning models, and generates natural language explanations using a Large Language Model (LLM).

---

# Solution Developed

To solve this problem, I developed a complete machine learning pipeline consisting of four major parts.

### Part 1 — Data Acquisition & Exploratory Data Analysis

- Loaded raw solar plant datasets
- Cleaned missing values
- Removed duplicate records
- Corrected data types
- Performed statistical analysis
- Detected outliers using IQR
- Computed Pearson and Spearman correlations
- Generated multiple visualizations
- Produced a clean dataset for machine learning

---

### Part 2 — Supervised Machine Learning

Built predictive models for solar power generation.

Regression Models

- Linear Regression
- Ridge Regression

Classification Model

- Logistic Regression
- SMOTE for handling class imbalance
- ROC Curve analysis
- Threshold sensitivity analysis
- Bootstrap confidence interval analysis

The models predict expected AC power generation and classify underperforming operating conditions.

---

### Part 3 — Advanced Machine Learning

Implemented multiple ensemble learning algorithms and compared their performance.

Models implemented

- Decision Tree
- Controlled Decision Tree
- Random Forest
- Gradient Boosting

Additional tasks

- Feature Importance Analysis
- Feature Ablation Study
- Cross Validation
- Hyperparameter Tuning using GridSearchCV
- Learning Curve Analysis
- Model Serialization using Joblib

The best-performing model was saved for deployment.

---

### Part 4 — Large Language Model Integration

Integrated an LLM into the machine learning workflow to generate understandable explanations for model predictions.

Features include

- Structured prediction explanations
- Performance ratio interpretation
- Prompt engineering
- Temperature comparison
- PII detection and protection
- Production-ready guardrails

Instead of only predicting values, the system explains *why* an inverter is underperforming in natural language.

---

# Technologies Used

- Python 3.13
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Requests
- Python-dotenv
- Git
- GitHub
- Large Language Model API

---

# Dataset

This project uses the publicly available **Solar Power Generation Dataset**, which contains:

- Plant_1_Generation_Data.csv
- Plant_1_Weather_Sensor_Data.csv

The datasets are available inside the **datasets/** directory.

---

# Repository Structure

```
Applied-AI-ML-Essentials-Capstone/
│
├── datasets/
├── figures/
├── part1/
├── part2/
├── part3/
├── part4/
├── results/
├── cleaned_data.csv
├── expected_power_model.pkl
├── best_model.pkl
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Key Features

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ Regression Modeling

✔ Binary Classification

✔ Ensemble Learning

✔ Hyperparameter Optimization

✔ Cross Validation

✔ Learning Curve Analysis

✔ Model Serialization

✔ LLM-powered Prediction Explanation

✔ Production Guardrails

✔ PII Detection

---

# Results

The developed system successfully

- cleaned and prepared raw solar plant data,
- achieved high regression performance for AC power prediction,
- accurately identified underperforming solar operating conditions,
- compared multiple machine learning algorithms,
- selected the best-performing model,
- generated human-readable explanations using an LLM,
- demonstrated production-ready AI practices including structured outputs and PII protection.

---

# Installation

Install all required packages

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run each part independently.

```bash
python part1/part1_eda.py
```

```bash
python part2/part2_models.py
```

```bash
python part3/part3_ensembles.py
```

```bash
python part4/part4_llm.py
```

---

# License

This project is licensed under the MIT License.

---

# Author

**Priya Dharshini**

Applied AI & ML Essentials Capstone Project
