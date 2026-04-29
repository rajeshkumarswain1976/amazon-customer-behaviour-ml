# Amazon Customer Behaviour — ML Pipeline

A clean, interpretable, end-to-end machine learning pipeline built on real Amazon customer purchase data. The focus of this project is on **honesty of labels, correctness of methodology, and simplicity of implementation** — every model trains on ground-truth data from the dataset, with no synthetic overrides, no inflated metrics, and no unnecessary complexity.

The pipeline covers the full data science workflow: raw data cleaning, customer segmentation, lifetime value prediction, and churn classification — with all trained models exported as `.pkl` files ready for integration into any application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Model Results](#model-results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Key Design Decisions](#key-design-decisions)

---

## Project Overview

Three production-ready models trained on real Amazon customer data:

| Model | Algorithm | Task |
|-------|-----------|------|
| CLV Prediction | Linear Regression | Predict Customer Lifetime Value |
| Churn Prediction | Random Forest + SMOTE | Predict whether a customer will churn |
| Customer Segmentation | Elbow Method + KMeans | Group customers into 3 behaviour-based segments |

---

## Dataset

**File:** `amazon.xlsx`

| Property | Value |
|----------|-------|
| Rows | 1,591 |
| Columns | 19 |
| Churn distribution | 797 churned / 794 not churned (near-perfectly balanced) |
| CLV range | 56 – 7,204 |
| Loyalty Score range | 1 – 100 |

### Columns

```
Customer_ID, Customer_Name, Age, Gender, Location, Product_Category,
Product_ID, Purchase_Date, Purchase_Amount, Payment_Method, Rating,
Feedback_Comments, Customer_Lifetime_Value, Loyalty_Score,
Discount_Applied, Return_Status, Customer_Segment,
Preferred_Shopping_Channel, Churn
```

---

## Project Structure

```
amazon-customer-behaviour-ml/
├── AMAZON_3_1_updated.ipynb     # Main notebook (33 cells)
├── amazon.xlsx                  # Raw input dataset (never overwritten)
├── Segmented_Customers.xlsx     # Output with cluster labels added
├── clv_model.pkl                # Trained LinearRegression model
├── clv_scaler.pkl               # StandardScaler for CLV features
├── churn_model.pkl              # Trained RandomForestClassifier
├── churn_scaler.pkl             # StandardScaler for churn features
├── kmeans_model.pkl             # Trained KMeans model
├── kmeans_scaler.pkl            # StandardScaler for segmentation features
├── requirements.txt
└── README.md
```

---

## Pipeline Walkthrough

### 1. Data Loading & Cleaning

- Loads `amazon.xlsx` and strips all column name whitespace
- Fills missing numeric values with **median**
- Fills missing categorical values with **mode**
- Removes duplicate records on `Customer_ID` + `Purchase_Date`
- Label-encodes: `Gender`, `Payment_Method`, `Preferred_Shopping_Channel`
- Converts `Return_Status` to binary (Yes → 1, No → 0)
- Removes outliers using **Z-score < 3** on `Purchase_Amount`

### 2. Customer Segmentation — Elbow + KMeans

- Features: `Customer_Lifetime_Value`, `Loyalty_Score`, `Purchase_Amount`
- Elbow method plotted across k=2 to k=9 to justify `k=3`
- All features scaled with `StandardScaler` before clustering
- Silhouette score: **0.348** — confirms well-separated clusters
- Segment labels merged back into the main DataFrame and exported

### 3. CLV Prediction — Linear Regression

- Target: `Customer_Lifetime_Value` (real column from dataset, range 56–7,204)
- 80/20 train-test split with `random_state=42`
- Features scaled with `StandardScaler`

**Features used (5):**
```
Age, Purchase_Amount, Discount_Applied, Payment_Method, Loyalty_Score
```

### 4. Churn Prediction — Random Forest + SMOTE

- Target: real `Churn` column from dataset (binary 0/1)
- Stratified 80/20 split to preserve class distribution
- SMOTE applied on training set only — strictly after the split
- RandomForestClassifier with 100 estimators

**Features used (5):**
```
Age, Purchase_Amount, Discount_Applied, Payment_Method, Loyalty_Score
```

---

## Model Results

### CLV Prediction

| Metric | Value |
|--------|-------|
| R² Score | **0.7622** |
| MSE | 652,889 |
| MAE | 598.1 |
| MAPE | **28.2%** |
| Target column | Customer_Lifetime_Value (real, range 56–7,204) |

### Churn Prediction

| Metric | Value |
|--------|-------|
| Accuracy | **84.3%** |
| Precision (Churn class) | 0.84 |
| Recall (Churn class) | 0.84 |
| F1-Score (Churn class) | **0.83** |
| Balancing | SMOTE on training set only |
| Label source | Real Churn column from dataset |

**Classification report:**
```
              precision    recall  f1-score   support

    No Churn       0.85      0.84      0.84       159
       Churn       0.84      0.84      0.84       160

    accuracy                           0.84       319
   macro avg       0.84      0.84      0.84       319
weighted avg       0.84      0.84      0.84       319
```

### Customer Segmentation

| Metric | Value |
|--------|-------|
| Clusters (k) | 3 |
| Silhouette score | **0.348** |
| k selection | Elbow method (plotted) |
| Features | Customer_Lifetime_Value, Loyalty_Score, Purchase_Amount |

---

## Technologies Used

```
Python 3.x
pandas
numpy
scikit-learn
imbalanced-learn (SMOTE)
matplotlib
seaborn
scipy
joblib
openpyxl
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
scipy
joblib
openpyxl
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/amazon-customer-behaviour-ml.git
cd amazon-customer-behaviour-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Make sure `amazon.xlsx` is in the root folder alongside the notebook.

### 4. Run the notebook

```bash
jupyter notebook AMAZON_3_1_updated.ipynb
```

Run all cells top to bottom. The notebook will clean the data, train all three models, save 6 `.pkl` files, and export `Segmented_Customers.xlsx`.

### 5. Use the saved models in your own script

```python
import joblib
import numpy as np

# ── CLV Prediction ──────────────────────────────────────────
clv_model  = joblib.load('clv_model.pkl')
clv_scaler = joblib.load('clv_scaler.pkl')

# Features: Age, Purchase_Amount, Discount_Applied,
#           Payment_Method (encoded), Loyalty_Score
sample_clv = np.array([[35, 250.0, 0, 2, 65]])
predicted_clv = clv_model.predict(clv_scaler.transform(sample_clv))
print(f"Predicted CLV: ${predicted_clv[0]:.2f}")

# ── Churn Prediction ────────────────────────────────────────
churn_model  = joblib.load('churn_model.pkl')
churn_scaler = joblib.load('churn_scaler.pkl')

# Features: Age, Purchase_Amount, Discount_Applied,
#           Payment_Method (encoded), Loyalty_Score
sample_churn = np.array([[35, 250.0, 0, 2, 65]])
churn_pred = churn_model.predict(churn_scaler.transform(sample_churn))
print(f"Churn: {'Will Churn' if churn_pred[0] == 1 else 'No Churn'}")

# ── Customer Segment ────────────────────────────────────────
kmeans_model  = joblib.load('kmeans_model.pkl')
kmeans_scaler = joblib.load('kmeans_scaler.pkl')

# Features: Customer_Lifetime_Value, Loyalty_Score, Purchase_Amount
sample_seg = np.array([[1500.0, 65, 250.0]])
segment = kmeans_model.predict(kmeans_scaler.transform(sample_seg))
print(f"Customer Segment: {int(segment[0])}")
```

---

## Output Files

| File | Description |
|------|-------------|
| `Segmented_Customers.xlsx` | Full DataFrame with `Segment` column added (1,591 rows) |
| `clv_model.pkl` | LinearRegression trained on 5 core features |
| `clv_scaler.pkl` | StandardScaler fitted on CLV training data |
| `churn_model.pkl` | RandomForestClassifier trained on 5 core features |
| `churn_scaler.pkl` | StandardScaler fitted on churn training data |
| `kmeans_model.pkl` | KMeans (k=3) fitted on 3 segmentation features |
| `kmeans_scaler.pkl` | StandardScaler fitted on segmentation data |

---

## Key Design Decisions

**Real churn labels, not synthetic ones.**
The dataset contains a ground-truth `Churn` column with a near-perfect 50/50 class balance (797 churned vs 794 not). This project trains exclusively on those real labels. The reported 84.3% accuracy reflects genuine predictive power — the model learned actual customer behaviour, not an invented rule derived from CLV thresholds or loyalty scores.

**SMOTE applied on training data only.**
SMOTE is applied strictly after the train-test split, on the training set alone. The test set always contains the original real distribution. This is the correct methodology — applying SMOTE before splitting leaks synthetic samples into the test set and produces artificially inflated accuracy figures.

**Elbow method to select k=3.**
Rather than assuming a cluster count, the notebook computes and plots inertia for k=2 through k=9. The elbow appears clearly at k=3, and the resulting silhouette score of 0.348 confirms the clusters are meaningfully separated in feature space.

**`Customer_Lifetime_Value` as the CLV target.**
The dataset provides a real CLV column ranging from 56 to 7,204. Using it as the regression target gives a MAPE of 28.2%. Replacing it with an engineered proxy (Avg_Purchase × Order_Count) compresses the range to 10–1,217 and inflates MAPE to approximately 73% — a common mistake avoided here.

**Minimal, interpretable feature set.**
Both the CLV and churn models use 5 core features drawn directly from the raw dataset — no feature engineering dependencies required. This makes the pipeline straightforward to audit, easy to explain to a non-technical audience, and simple to deploy in any environment.

---

## Acknowledgements

Dataset used for academic and learning purposes.
Analysis conducted as part of a customer behaviour study using Amazon purchase data.


> **Note:** An extended version of this project with expanded feature engineering,
> 11-feature churn model (85.58% accuracy), CLV model with 9 features (R² 0.7648),
> and a full Streamlit app is available at
> [customer_purchase_behaviour_prediction]([(https://github.com/rajeshkumarswain1976/customer_purchase_behaviour_prediction)]).
