Applied AI & ML Essentials Capstone
Project Overview

This repository contains my submission for the Applied AI & ML Essentials Capstone Project. In this project, I developed an end-to-end Artificial Intelligence and Machine Learning solution for monitoring the performance of a solar power plant using real-world operational data.

The system analyzes sensor readings from solar panels and inverters, predicts the expected AC power generation, identifies underperforming operating conditions, compares multiple machine learning models, and uses a Large Language Model (LLM) to generate human-readable explanations for the predictions.

Problem Statement

Solar power plants consist of hundreds or even thousands of solar panels connected to multiple inverters. The amount of electricity generated depends on several factors, including:

Solar irradiation (sunlight intensity)
Ambient temperature
Module temperature
Weather conditions
Seasonal changes
Dust accumulation on solar panels
Panel aging and degradation
Shading caused by nearby objects
Inverter faults or hardware failures
Electrical connection issues

When any of these factors occur, the solar plant may generate less electricity than expected. Detecting these problems manually is difficult because technicians must inspect large numbers of solar panels and inverters spread across a wide area. Manual inspection is time-consuming, labor-intensive, and expensive, making it challenging to identify the root cause of reduced power generation quickly.

Traditional monitoring systems typically display only sensor readings or predicted values. They do not explain why an inverter or panel is underperforming, making troubleshooting more difficult for maintenance engineers.

This project addresses these challenges by developing an intelligent AI-based monitoring system that automatically analyzes solar plant data, predicts expected power generation, detects underperforming conditions, and provides clear natural-language explanations to help operators understand possible causes and take appropriate maintenance actions.

Solution Developed

To address the above problem, I developed a complete AI-powered solar monitoring system consisting of four major stages.

Part 1 – Data Acquisition and Exploratory Data Analysis
Loaded raw solar generation and weather sensor datasets
Cleaned missing and duplicate values
Corrected data types
Performed descriptive statistical analysis
Detected outliers using the IQR method
Calculated Pearson and Spearman correlations
Created multiple visualizations to understand plant behavior
Generated a cleaned dataset for machine learning
Part 2 – Supervised Machine Learning

Developed predictive models to estimate the expected AC power output of solar inverters and identify underperforming operating conditions.

Implemented:

Linear Regression
Ridge Regression
Logistic Regression
SMOTE for handling class imbalance
ROC Curve analysis
Threshold sensitivity analysis
Bootstrap confidence interval analysis

These models learn the relationship between weather conditions and power generation to estimate expected performance and detect abnormal behavior.

Part 3 – Advanced Machine Learning

Compared multiple ensemble learning algorithms to identify the most accurate model.

Implemented:

Decision Tree
Controlled Decision Tree
Random Forest
Gradient Boosting

Also performed:

Feature Importance Analysis
Feature Ablation Study
Cross Validation
Hyperparameter Tuning using GridSearchCV
Learning Curve Analysis
Model Serialization using Joblib

The best-performing model was selected and saved for deployment.

Part 4 – Large Language Model Integration

Integrated a Large Language Model (LLM) to generate understandable explanations for model predictions.

Instead of simply displaying prediction results, the LLM explains possible reasons for low power generation, such as:

Low solar irradiation due to cloudy weather
High module temperature reducing panel efficiency
Dust accumulation on the solar panels
Seasonal variations in sunlight
Possible inverter malfunction
Reduced performance caused by environmental conditions

The system also includes:

Structured prediction explanations
Prompt engineering
Temperature comparison
Personally Identifiable Information (PII) protection
Production-ready guardrails

This enables maintenance engineers to understand the likely causes of underperformance quickly without manually analyzing large amounts of sensor data.

Project Outcome

The developed AI system successfully:

Cleans and prepares raw solar plant data for analysis.
Predicts expected AC power generation using machine learning.
Detects underperforming solar inverters and operating conditions.
Compares multiple machine learning models to identify the best-performing solution.
Generates human-readable explanations using a Large Language Model (LLM).
Reduces the need for manual monitoring by helping maintenance engineers identify potential issues such as dust accumulation, weather effects, seasonal variations, overheating, and inverter faults more efficiently.









































































































































