# 🌞 Applied AI & ML Essentials Capstone

## 📖 Project Overview

This repository contains my submission for the **Applied AI & ML Essentials Capstone Project**. In this project, I developed an **end-to-end Artificial Intelligence and Machine Learning solution** for monitoring the performance of a **solar power plant** using real-world operational data.

The system analyzes sensor readings collected from **solar panels** and **inverters**, predicts the expected AC power generation, detects underperforming operating conditions, compares multiple machine learning models, and integrates a **Large Language Model (LLM)** to generate human-readable explanations for the predictions.

---

# 🚩 Problem Statement

Modern solar power plants consist of **hundreds or even thousands of solar panels** connected to multiple inverters. The amount of electricity generated depends on several environmental and operational factors, including:

- ☀️ Solar irradiation (sunlight intensity)
- 🌡️ Ambient temperature
- 🔥 Module temperature
- ☁️ Weather conditions
- 🍂 Seasonal variations
- 🧹 Dust accumulation on solar panels
- ⏳ Panel aging and degradation
- 🌳 Shading caused by nearby objects
- ⚡ Inverter faults or hardware failures
- 🔌 Electrical connection issues

When any of these factors occur, the solar plant may generate **less electricity than expected**.

Traditionally, technicians must manually inspect hundreds of solar panels and inverters to identify the cause of reduced power generation. This process is:

- Time-consuming
- Labor-intensive
- Expensive
- Difficult to scale for large solar farms

In addition, conventional monitoring systems usually display only sensor readings or predicted values without explaining **why** the power output has decreased.

This project solves these challenges by developing an **AI-powered intelligent monitoring system** that automatically analyzes solar plant data, predicts expected power generation, detects underperforming operating conditions, and generates natural-language explanations to assist maintenance engineers in identifying potential issues quickly.

---

# 💡 Solution Developed

To address the above problem, I developed a complete **AI-powered Solar Monitoring System** consisting of four major stages.

---

## 📊 Part 1 – Data Acquisition & Exploratory Data Analysis (EDA)

In this phase, the raw solar plant data was cleaned and analyzed before model development.

### Tasks Performed

- 📂 Loaded solar generation and weather datasets
- 🧹 Cleaned missing values
- 🗑️ Removed duplicate records
- 🔄 Corrected incorrect data types
- 📈 Performed descriptive statistical analysis
- 📉 Detected outliers using the IQR method
- 🔗 Computed Pearson and Spearman correlations
- 📊 Created multiple visualizations
- 💾 Generated a cleaned dataset for machine learning

---

## 🤖 Part 2 – Supervised Machine Learning

Built predictive models to estimate the **expected AC power generation** and detect underperforming operating conditions.

### Regression Models

- Linear Regression
- Ridge Regression

### Classification Model

- Logistic Regression
- SMOTE for class imbalance handling
- ROC Curve Analysis
- Threshold Sensitivity Analysis
- Bootstrap Confidence Interval Analysis

These models learn the relationship between environmental conditions and solar power generation to estimate expected performance and detect abnormal operating behavior.

---

## 🌲 Part 3 – Advanced Machine Learning

Compared multiple machine learning algorithms to identify the most accurate model.

### Models Implemented

- Decision Tree
- Controlled Decision Tree
- Random Forest
- Gradient Boosting

### Additional Analysis

- Feature Importance Analysis
- Feature Ablation Study
- Cross Validation
- Hyperparameter Tuning using GridSearchCV
- Learning Curve Analysis
- Model Serialization using Joblib

The best-performing model was selected and saved for deployment.

---

## 🤖 Part 4 – Large Language Model (LLM) Integration

Integrated a **Large Language Model (LLM)** into the machine learning workflow to generate understandable explanations for predictions.

Instead of only displaying predicted values, the LLM explains possible reasons for reduced power generation, including:

- ☁️ Low solar irradiation due to cloudy weather
- 🔥 High module temperature reducing panel efficiency
- 🧹 Dust accumulation on solar panels
- 🍂 Seasonal changes
- ⚡ Possible inverter malfunction
- 🌦️ Environmental conditions affecting generation

### Additional Features

- Structured Prediction Explanations
- Prompt Engineering
- Temperature Comparison
- Personally Identifiable Information (PII) Protection
- Production-ready Guardrails

This enables maintenance engineers to quickly understand the likely causes of underperformance without manually analyzing thousands of sensor readings.

---

# 🛠️ Technologies Used

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
- Large Language Model (LLM) API

---

# 📂 Dataset

This project uses the publicly available **Solar Power Generation Dataset**, consisting of:

- Plant_1_Generation_Data.csv
- Plant_1_Weather_Sensor_Data.csv

The datasets are available inside the **datasets/** directory.

---

# 📁 Repository Structure

```text
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

# ⭐ Key Features

- ✅ Data Cleaning
- ✅ Exploratory Data Analysis
- ✅ Feature Engineering
- ✅ Regression Modeling
- ✅ Binary Classification
- ✅ Ensemble Learning
- ✅ Hyperparameter Optimization
- ✅ Cross Validation
- ✅ Learning Curve Analysis
- ✅ Model Serialization
- ✅ LLM-powered Prediction Explanation
- ✅ Production Guardrails
- ✅ PII Detection

---

# 📈 Project Outcome

The developed AI system successfully:

- ✅ Cleans and prepares raw solar plant data for analysis.
- ✅ Predicts expected AC power generation using machine learning.
- ✅ Detects underperforming solar inverters and operating conditions.
- ✅ Compares multiple machine learning models to identify the best-performing solution.
- ✅ Generates human-readable explanations using a Large Language Model (LLM).
- ✅ Reduces the need for manual monitoring by helping maintenance engineers identify potential issues such as dust accumulation, weather effects, seasonal variations, overheating, and inverter faults.

---

# ⚙️ Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run each part independently.

### Part 1

```bash
python part1/part1_eda.py
```

### Part 2

```bash
python part2/part2_models.py
```

### Part 3

```bash
python part3/part3_ensembles.py
```

### Part 4

```bash
python part4/part4_llm.py
```

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👩‍💻 Author

**Priya Dharshini**

Applied AI & ML Essentials Capstone Project






























