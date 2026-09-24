# Diabetes Prediction Model

A machine learning project that predicts whether a patient has diabetes from medical measurements, using the **Pima Indians Diabetes Database**. Five classification models are trained, compared with standard metrics and ROC curves, and the best one is saved with `joblib`.

Model Link: https://colab.research.google.com/drive/15hjXa-6lp6O5k70DFDpiYbC9J5SGOccq?usp=sharing

## Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Limitations & Future Work](#-limitations--future-work)
- [Tech Stack](#-tech-stack)

---

## Overview

**Goal:** Build a binary classifier that predicts diabetes (`Outcome = 1`) or no diabetes (`Outcome = 0`) from eight medical attributes.

**What this project covers:**
- Data loading and exploratory data analysis (EDA)
- Cleaning hidden missing values (impossible zeros)
- Leak-free preprocessing with scikit-learn pipelines
- Training and comparing 5 classifiers
- Evaluation with Accuracy, Precision, Recall, F1-score and ROC-AUC
- ROC curve comparison and confusion matrices
- Saving and reloading the best model with `joblib`

---

## Dataset

**Source:** [Pima Indians Diabetes Database](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)

- **768 patients**, all female, at least 21 years old, of Pima Indian heritage
- **8 features + 1 target**

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes family-history score |
| `Age` | Age in years |
| `Outcome` | **Target:** 1 = diabetes, 0 = no diabetes |

**Class balance:** 500 non-diabetic (65.1%) and 268 diabetic (34.9%).

>  **Data quality note:** The dataset has no `NaN` values, but `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` contain zeros that are medically impossible. These are placeholders for missing data (for example, 374 of 768 `Insulin` values are zero). This project treats them as missing and imputes them.

---

##  Project Structure

```
├── Diabetes Prediction Model.ipynb   # Main notebook (EDA, training, evaluation)
└── README.md
```

---

## Methodology

### 1. Data exploration
Head, shape, `info()`, missing values, `describe()`, class distribution, feature histograms and a correlation heatmap. `Glucose` shows the strongest correlation with the outcome, followed by `BMI`, `Age` and `Pregnancies`.

### 2. Preprocessing
- Replace impossible zeros with `NaN` in the five affected columns
- Separate features (`X`) and target (`y`)
- **80/20 train/test split**, stratified to preserve the class ratio (`random_state=42`)
- Median imputation and standard scaling inside a scikit-learn `Pipeline`, so both are **fit on training data only** (no data leakage)

### 3. Models trained
| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Random Forest | `n_estimators=300` |
| Support Vector Machine | RBF kernel, `probability=True` |
| Decision Tree | `max_depth=5` |
| K-Nearest Neighbors | `n_neighbors=7` |

### 4. Evaluation
Accuracy, Precision, Recall, F1 and ROC-AUC on the held-out test set (154 samples), plus a combined ROC curve plot, metric comparison charts and confusion matrices.

### 5. Model selection
The best model is chosen by **ROC-AUC**, which measures how well a model ranks patients across all thresholds and is robust to class imbalance. Recall is reported alongside it because missing a diabetic patient is costly in a medical setting.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest**  | 0.740 | 0.659 | 0.537 | 0.592 | **0.816** |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.546 | 0.813 |
| SVM | 0.740 | 0.652 | 0.556 | 0.600 | 0.796 |
| KNN | 0.727 | 0.625 | 0.556 | 0.588 | 0.790 |
| Decision Tree | 0.760 | 0.639 | 0.722 | 0.678 | 0.762 |

**Takeaways**
- **Random Forest** achieved the best ROC-AUC (0.816) and was saved as the final model.
- **Logistic Regression** is close behind (0.813) and is simpler and more interpretable.
- The **Decision Tree** has the highest accuracy and recall but the lowest AUC, so it ranks patients less reliably across thresholds.

> Exact numbers can vary slightly with library versions. Results here use `random_state=42`.

---

## Getting Started

Click the **link** at the top, then choose **Runtime → Run all**. No setup is needed.

## Using the Saved Model

The saved file is a complete pipeline (imputer, scaler and classifier), so you can pass it raw patient values directly.

```python
import joblib
import pandas as pd

model = joblib.load("diabetes_model.pkl")

patient = pd.DataFrame([{
    "Pregnancies": 2, "Glucose": 150, "BloodPressure": 72, "SkinThickness": 35,
    "Insulin": 150, "BMI": 33.6, "DiabetesPedigreeFunction": 0.63, "Age": 45,
}])

prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0, 1]

print("Diabetic" if prediction == 1 else "Not diabetic", f"({probability:.1%})")
```

> Use the same column names and order as the training data. Missing values can be passed as `NaN` (or `0` for the five affected columns, but `NaN` is preferred).

---

## Limitations & Future Work

- **Small dataset** (768 rows), so metrics can shift noticeably with a different split. Cross-validation would give more stable estimates.
- **Narrow population:** all patients are Pima Indian women aged 21+, so the model may not generalize to other groups.
- **Moderate recall** (about 54% for the best model): many diabetic patients are missed at the default 0.5 threshold.
- **Not a medical tool.** This is an educational project and must not be used for real diagnosis.

**Ideas for improvement**
- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Stratified k-fold cross-validation
- Class-imbalance handling (`class_weight="balanced"`, SMOTE)
- Threshold tuning to prioritize recall
- Gradient boosting models (XGBoost, LightGBM)
- Model explainability with SHAP or feature importances
- A simple web app (Streamlit or Flask) for interactive predictions

---

## Tech Stack

Python · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Joblib · Google Colab

---
