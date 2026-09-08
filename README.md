# 🚢 Titanic Survival Prediction & Exploratory Data Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Model Accuracy](https://img.shields.io/badge/Test%20Accuracy-82.12%25-brightgreen.svg)]()
[![5-Fold CV](https://img.shields.io/badge/5--Fold%20CV-83.05%25-green.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8530-blueviolet.svg)]()

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) project analyzing passenger survival factors from the 1912 Titanic disaster. This project automates data hygiene, conducts deep demographic comparisons, and trains a tuned **Random Forest Classifier** to answer the fundamental question:

> **"What sorts of people were likely to survive the Titanic disaster?"**

---

## 📊 Key Highlights & Results

| Metric | Score / Value | Description |
| :--- | :---: | :--- |
| **Test Accuracy** | **82.12%** | Evaluated on an independent 20% holdout test set (179 passengers) |
| **5-Fold Cross-Validation** | **83.05% (± 1.62%)** | Stratified across the full dataset to verify generalization |
| **ROC-AUC Score** | **0.8530** | Measures discrimination power between survivors and casualties |
| **Non-Survivor Recall** | **89.1%** | 98 of 110 non-survivors correctly identified |
| **Survivor Precision** | **80.3%** | High confidence when predicting passenger survival |
| **Production Artifact** | `titanic_model.joblib` | Serialized model bundle ready for live inference |

---

## 🔍 The Core Finding: Who Was Likely to Survive?

Through feature importance analysis and demographic modeling, survival likelihood was strongly dictated by social rank, age, and maritime evacuation protocol (*"women and children first"*):

1. **Gender & Social Title (`Title_Mr`, `Sex`)**:
   - **Female Passengers (`Mrs`, `Miss`)**: **~74.2% survived**.
   - **Adult Male Passengers (`Mr`)**: Only **~18.9% survived**.
   - The title `Mr` was the single strongest negative predictor of survival.

2. **Socio-Economic Class & Wealth (`Pclass`, `Fare`, `Has_Cabin`)**:
   - **1st Class Females**: **96.8%** survival rate.
   - **3rd Class Males**: **13.5%** survival rate.
   - Higher fare values and possessing an upper-deck cabin substantially improved evacuation access.

3. **Age (`Age`, `Title_Master`)**:
   - Children under 10 (especially young boys with title `Master`) were prioritized during lifeboat boarding.
   - The highest casualty rate was concentrated in working-age men (ages 18–35).

4. **Family Dynamics (`FamilySize`, `IsAlone`)**:
   - Passengers travelling in **small families (2–4 members)** achieved the highest survival rates (**55%–72%**).
   - Solo travelers and large families (5+ members) experienced higher casualty rates (<30%).

---

## 📈 Visual Exploratory Dashboard

The pipeline generates comprehensive visual dashboards saved under [`plots/`](plots/):

![Titanic EDA Dashboard](plots/titanic_comparative_dashboard.png)

Included analyses:
- **Survival by Class & Sex** (Socio-economic impact)
- **Age KDE Distribution** (Demographic survival curves)
- **Fare by Class & Survival** (Outlier-free boxplots)
- **Survival by Family Size** (Companionship dynamics)
- **Survival by Port & Cabin Allocation** (Boarding and deck factors)
- **Feature Correlation Heatmap** (Multivariate relationships)

---

## 🛠️ Data Pipeline & Feature Engineering

1. **Intelligent Imputation**:
   - `Embarked`: Mode-imputed (`'S'`, Southampton).
   - `Age`: Imputed using the **median age grouped by Passenger Class (`Pclass`) and Gender (`Sex`)**, preserving social-economic demographic distributions.
   - `Cabin`: Transformed into a binary indicator `Has_Cabin` (`1` if recorded, `0` otherwise) to retain upper-deck signals without loss from sparsity.
2. **Feature Extraction**:
   - Extracted social titles (`Mr`, `Mrs`, `Miss`, `Master`, `Other`) from passenger names.
   - Calculated `FamilySize = SibSp + Parch + 1`.
   - Generated binary `IsAlone` travel indicator.
3. **Encoding & Hygiene**:
   - One-hot encoded categorical features (`Embarked`, `Title`).
   - Mapped `Sex` to binary (`0` = male, `1` = female).
   - Output dataset saved to [`Titanic_Dataset_cleaned.csv`](Titanic_Dataset_cleaned.csv) with **0 missing values**.

---

## ⚙️ Model Architecture & Evaluation Strategy

- **Validation Split**: Stratified **80/20 train-test split** (712 training passengers, 179 testing passengers) with random seed `42` to guarantee class proportion fidelity.
- **Cross-Validation**: Standard **5-fold cross-validation** across the entire dataset.
- **Algorithm**: `RandomForestClassifier` (`n_estimators=100`, `max_depth=6`, `random_state=42`) for robust handling of non-linear interaction terms.

### Test Set Confusion Matrix (20% Holdout, N = 179)

```text
               Predicted Died    Predicted Survived
Actual Died          98                  12          (89.1% Recall)
Actual Survived      20                  49          (71.0% Recall)
```

---

## 📂 Project Structure

```text
Titanic_Survival_model/
│
├── README.md                      # Project documentation and quickstart guide
├── PROJECT_REPORT.md              # In-depth technical development report
│
├── titanic.py                     # Primary pipeline: ETL, EDA, 80/20 training, CV, live inference
├── plot_analysis.py               # Standalone generator for publication-ready visual figures
│
├── Titanic_Dataset.csv            # Original raw Titanic dataset (891 rows, 12 columns)
├── Titanic_Dataset_cleaned.csv    # Cleaned, engineered dataset (891 rows, 16 features)
├── titanic_model.joblib           # Serialized Random Forest model artifact
│
└── plots/
    ├── titanic_comparative_dashboard.png  # 6-panel exploratory dashboard
    ├── class_gender_survival.png          # Class & gender breakdown
    ├── age_class_survival_violin.png      # Age distributions across tiers
    └── full_correlation_heatmap.png       # Complete correlation matrix
```

---

## 🚀 Quickstart & Usage

### 1. Prerequisites

Ensure Python 3.8+ is installed with the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Run the Full Model Pipeline

Execute [`titanic.py`](titanic.py) to clean data, train the model, output metrics, and run interactive predictions:

```bash
# Run with interactive graphical plot dashboard:
python titanic.py

# Run in headless/console mode (no GUI window popup):
python titanic.py --no-show
```

### 3. Generate Extra Analytical Plots

To export individual high-resolution figures to [`plots/`](plots/):

```bash
python plot_analysis.py
```

### 4. Live Prediction Demo

The trained model bundle in [`titanic_model.joblib`](titanic_model.joblib) can be loaded to evaluate new passenger profiles:

```python
import joblib
import pandas as pd

# Load serialized model bundle
bundle = joblib.load('titanic_model.joblib')
model = bundle['model']
features = bundle['features']

# Example: 1st class adult female traveling with family
sample = {f: 0 for f in features}
sample.update({
    'Pclass': 1, 'Sex': 1, 'Age': 28.0, 'Fare': 75.0,
    'FamilySize': 2, 'IsAlone': 0, 'Has_Cabin': 1, 'Title_Mrs': 1
})

input_df = pd.DataFrame([sample])
prob = model.predict_proba(input_df)[0, 1]
print(f"Survival Probability: {prob:.1%}")
# Output: ~96.6% (Survived)
```

---

## 📜 Detailed Report

For the complete technical breakdown, methodology analysis, and research findings, see [PROJECT_REPORT.md](PROJECT_REPORT.md).
