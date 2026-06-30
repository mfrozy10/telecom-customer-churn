# Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Table of Contents
1. [Background & Problem Statement](#1-background--problem-statement)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Methodology](#4-methodology)
5. [Model Performance](#5-model-performance)
6. [Feature Importance & Business Insights](#6-feature-importance--business-insights)
7. [Interactive Simulation App](#7-interactive-simulation-app)
8. [Tech Stack](#8-tech-stack)
9. [How to Run](#9-how-to-run)

---

## 1. Background & Problem Statement

Customer churn — the phenomenon where customers stop doing business with a company — is one of the most critical challenges in the telecommunications industry. Acquiring a new customer costs **5 to 7 times more** than retaining an existing one, making churn prediction a high-value problem for any subscription-based business.

This project builds a **binary classification model** that predicts whether a telecom customer will churn, using demographic data, service subscription details, and billing information. The goal is not only to achieve strong predictive accuracy, but to extract **actionable business insights** that a retention team can act on immediately.

**Business Question:**
> *Which customers are most likely to leave, and what factors drive their decision?*

---

## 2. Dataset

| Property | Detail |
|---|---|
| Source | IBM Telco Customer Churn Dataset |
| Records | 7,043 customers |
| Features | 38 columns (raw) → 46 features (after encoding) |
| Target | `Customer Status` → binary: **Churned (1)** vs **Not Churned (0)** |
| Churn Rate | **26.5%** (imbalanced — handled via `scale_pos_weight`) |

**Files:**

```
dataset/
├── telecom_customer_churn.csv        # Raw dataset
├── telecom_data_dictionary.csv       # Column descriptions
├── telecom_zipcode_population.csv    # Supplementary geo data
└── telecom_churn_processed.csv       # Cleaned & encoded (generated)
```

---

## 3. Project Structure

```
Customer Churn/
│
├── dataset/
│   ├── telecom_customer_churn.csv
│   ├── telecom_data_dictionary.csv
│   ├── telecom_zipcode_population.csv
│   └── telecom_churn_processed.csv       # Cleaned & encoded (generated)
│
├── notebooks/
│   └── Churn_Prediction.ipynb            # EDA + Preprocessing + Modeling
│
├── models/
│   └── best_model.pkl                    # Saved XGBoost tuned model (232 KB)
│
├── app.py                                # Streamlit simulation app
├── requirements.txt                      # Python dependencies
├── CLAUDE.md                             # Project coding conventions
└── README.md
```

> All analysis, preprocessing, and modeling are contained in a single narrative notebook (`Churn_Prediction.ipynb`) following a storytelling approach — every code cell is preceded by a Markdown explanation of the analytical decision being made.

---

## 4. Methodology

### Section 1–9 — Exploratory Data Analysis

The EDA phase was designed to surface patterns that directly inform preprocessing and modeling decisions, not just describe the data.

**Key steps:**
- Loaded and profiled 7,043 records across 38 columns
- Identified and categorised **missing values by root cause** rather than treating them uniformly:
  - Internet-related columns (21.7% missing) → customers without internet service → filled with `"No"` / `0`
  - Phone-related columns (9.7% missing) → customers without phone service → filled with `"No"` / median
  - `Offer` column (55% missing) → no active promotion → filled with `"No Offer"`
  - `Churn Reason` / `Churn Category` → **post-churn leakage**, dropped entirely
- Confirmed **zero duplicate rows** and zero duplicate Customer IDs
- Visualised target distribution (26.5% churn rate → imbalanced)
- Computed Pearson correlation heatmap; identified `Tenure in Months` (r = −0.35) and `Number of Referrals` (r = −0.29) as the strongest numerical predictors

**Encoding strategy:**
- Label Encoding for 6 binary columns (Yes/No)
- One-Hot Encoding for 12 multi-class columns (Contract, Payment Method, Internet Type, etc.)
- Final feature matrix: **7,043 rows × 46 features**

---

### Section 10–15 — Modeling & Hyperparameter Tuning

**Train / Test Split**

80/20 split with **stratification** on the target to preserve the 26.5% churn ratio in both subsets.

| Subset | Rows | Churn Rate |
|---|---|---|
| Training | 5,634 | 26.5% |
| Testing | 1,409 | 26.5% |

**Baseline Comparison**

Two algorithms were trained with imbalance-aware configurations:

| Model | Configuration |
|---|---|
| Random Forest | `class_weight='balanced'`, 200 trees |
| XGBoost | `scale_pos_weight=2.77` (neg/pos ratio), 200 trees |

**5-Fold Stratified Cross Validation**

Used `StratifiedKFold` to ensure stable, unbiased performance estimates:

| Model | AUC Mean | AUC Std | F1 Mean |
|---|---|---|---|
| Random Forest | 0.8912 | ±0.0056 | 0.8274 |
| **XGBoost** | **0.8926** | ±0.0082 | 0.8341 |

XGBoost achieved the highest CV AUC and was selected for hyperparameter tuning.

**RandomizedSearchCV Tuning**

50 iterations × 5-fold CV (250 total fits) over the following search space:

| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | [100, 200, 300, 500] |
| `max_depth` | [3, 4, 5, 6, 8] |
| `learning_rate` | [0.01, 0.05, 0.1, 0.2] |
| `subsample` | [0.6, 0.8, 1.0] |
| `colsample_bytree` | [0.6, 0.8, 1.0] |
| `gamma` | [0, 0.1, 0.5, 1] |
| `reg_alpha` | [0, 0.1, 1] |
| `reg_lambda` | [1, 2, 5] |

**Best parameters found:**

```python
{
    "learning_rate":    0.05,
    "max_depth":        3,
    "n_estimators":     200,
    "subsample":        1.0,
    "colsample_bytree": 0.6,
    "gamma":            0.1,
    "reg_alpha":        1,
    "reg_lambda":       1
}
```

---

### Section 16 — Feature Importance

Feature importance was computed using three XGBoost metrics (weight, gain, cover) and visualised as horizontal bar charts. **Gain** was used as the primary metric, as it reflects the actual predictive contribution of each feature.

---

## 5. Model Performance

### Final Comparison (Testing Set)

| Model | ROC-AUC | F1 Weighted | Accuracy | Churn Recall |
|---|---|---|---|---|
| Random Forest (baseline) | 0.8862 | 0.8169 | 82% | 56% |
| XGBoost (baseline) | 0.8821 | 0.8308 | 83% | 70% |
| **XGBoost (tuned) ✅** | **0.9027** | **0.8120** | **80%** | **85%** |

### Classification Report — Best Model (XGBoost Tuned)

```
              precision    recall  f1-score   support

 Not Churned       0.94      0.79      0.85      1035
     Churned       0.59      0.85      0.70       374

    accuracy                           0.80      1409
   macro avg       0.76      0.82      0.77      1409
weighted avg       0.84      0.80      0.81      1409

ROC-AUC : 0.9027
```

### Why ROC-AUC > Accuracy for This Problem

In a churn prediction context, **missing a churner (false negative) is more costly than a false alarm (false positive)**. The tuned model deliberately trades some precision on the "Churned" class to achieve **85% recall** — meaning it correctly flags 85 out of every 100 customers who would have left. This is the right trade-off for a retention campaign.

---

## 6. Feature Importance & Business Insights

### Top 10 Features by Gain (XGBoost Tuned)

| Rank | Feature | Gain | Direction |
|---|---|---|---|
| 1 | `Contract_Two Year` | 21.8% | ↓ churn |
| 2 | `Premium Tech Support_No` | 15.6% | ↑ churn |
| 3 | `Online Security_No` | 13.4% | ↑ churn |
| 4 | `Contract_One Year` | 11.9% | ↓ churn |
| 5 | `Internet Type_Fiber Optic` | 10.3% | ↑ churn |
| 6 | `Number of Referrals` | 6.7% | ↓ churn |
| 7 | `Payment Method_Credit Card` | 5.5% | varies |
| 8 | `Premium Tech Support_Yes` | 5.5% | ↓ churn |
| 9 | `Streaming Movies_Yes` | 4.8% | ↓ churn |
| 10 | `Number of Dependents` | 4.6% | ↓ churn |

### Actionable Business Recommendations

**1. Prioritise Long-Term Contracts**
The single most powerful predictor of retention is a two-year contract (21.8% gain). Customers on month-to-month contracts are substantially more likely to churn. A targeted campaign offering discounted upgrades to annual or two-year plans could significantly reduce churn risk.

**2. Bundle Add-On Services as Retention Tools**
`Premium Tech Support` and `Online Security` rank #2 and #3. Customers without these add-ons churn at a much higher rate — not because the features themselves prevent churn, but because they create product stickiness. Offering these as free trials or bundled discounts for at-risk customers is a low-cost, high-impact intervention.

**3. Investigate Fiber Optic Satisfaction**
Fiber Optic customers (10.3% gain, positive churn direction) appear to have higher churn rates — likely because they have higher service expectations and more competitive alternatives. A dedicated NPS survey or proactive outreach for this segment is warranted.

**4. Leverage the Referral Programme**
`Number of Referrals` is the strongest continuous predictor of loyalty (6.7% gain). Customers who have referred others are significantly less likely to churn. Expanding referral incentives could simultaneously increase acquisition and improve retention.

**5. Target the High-Risk Profile**
Customers matching this profile are the highest churn risk:
- Month-to-month contract
- No Premium Tech Support
- No Online Security
- Fiber Optic internet
- Zero referrals made

---

## 7. Interactive Simulation App

A **Streamlit web app** (`app.py`) is included for real-time churn prediction via a form-based interface — no code required to use it.

### Features

- **3-column form layout** covering all customer attributes:
  - Personal Info & Account (gender, age, tenure, contract, offer, payment method)
  - Services (phone, internet type, add-on subscriptions)
  - Financials (monthly charge, total charges, revenue)
- **Conditional logic** — internet-dependent fields (security, streaming, etc.) are automatically handled when Internet Service is set to No
- **Live prediction** on form submit — loads the saved `best_model.pkl` and returns:
  - Binary verdict: **CHURN RISK DETECTED** or **NOT CHURN**
  - Exact churn probability (0–100%)
  - Colour-coded probability gauge (green / yellow / red)
- **Feature importance explainer** — expandable panel listing the top 10 global drivers
- **Debug panel** — shows the raw 46-feature vector sent to the model

### How to Launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Example Predictions

| Profile | Contract | Tech Support | Internet | Tenure | Churn Probability |
|---|---|---|---|---|---|
| High-risk | Month-to-Month | No | Fiber Optic | 6 months | ~75% |
| Low-risk | Two Year | Yes | Cable | 60 months | ~2% |

---

## 8. Tech Stack

| Category | Library / Tool |
|---|---|
| Language | Python 3.13 |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn 1.8 |
| Gradient Boosting | XGBoost 3.2 |
| Simulation App | Streamlit |
| Notebook Execution | nbconvert, nbclient |
| Model Persistence | pickle |

---

## 9. How to Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/telecom-churn-prediction.git
cd telecom-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**
```bash
# Option A — open in Jupyter
jupyter notebook notebooks/Churn_Prediction.ipynb

# Option B — execute headlessly and save outputs
python -m nbconvert --to notebook --execute --inplace notebooks/Churn_Prediction.ipynb
```

**4. Launch the simulation app**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**5. Load the saved model programmatically**
```python
import pickle

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# model is ready to predict on a 46-column encoded DataFrame
predictions   = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

---

*Built as part of a Machine Learning portfolio project. Dataset sourced from IBM Sample Data.*
