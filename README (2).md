# 🔧 Predictive Maintenance — Machine Failure Classification

## Overview

This project tackles a real-world industrial problem: **predicting machine failure before it happens** using sensor data from the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).

The dataset contains **10,000 records** of machine sensor readings — air temperature, process temperature, rotational speed, torque, and tool wear — along with a binary label indicating whether the machine failed.

---

## The Challenge — Class Imbalance

Only **3.4% of machines** in the dataset actually failed:

| Class | Count | Percentage |
|---|---|---|
| No Failure (0) | 9,661 | 96.6% |
| Failure (1) | 339 | 3.4% |

A naive model that always predicts "no failure" would achieve **96% accuracy** — but would be completely useless in practice. This made choosing the right evaluation metric critical.

> **In a real factory, missing a machine failure is far more costly than a false alarm. Recall on the failure class was the most important metric.**

---

## Project Structure

```
predictive-maintenance/
├── data/
│   └── ai4i2020.csv
├── 01_eda.ipynb        # EDA, preprocessing, and modeling
└── README.md
```

> The dataset is publicly available. Download it from the link above and place it in a `data/` folder in the project root.

---

## Workflow

### 1. Data Cleaning
- Checked for missing values → **none found**
- Checked for duplicates → **none found**
- Dropped `UDI` and `Product ID` — identifier columns with no predictive value
- Dropped `TWF`, `HDF`, `PWF`, `OSF`, `RNF` — failure sub-type columns that cause **data leakage**, as they directly reveal why a machine failed

### 2. Exploratory Data Analysis

Key findings from comparing failed vs. non-failed machines:

- **Torque** — failed machines had noticeably higher torque values
- **Tool wear** — higher tool wear was strongly associated with failure
- **Rotational speed** — right-skewed distribution; lower speeds linked to failure
- **Temperatures** — slight differences between classes but less decisive

**Correlation heatmap findings:**
- Air temperature and process temperature are highly correlated (**0.88**) — they carry almost the same information
- Rotational speed and torque have a strong negative correlation (**-0.88**) — physically expected: higher speed means lower torque
- Torque showed the strongest correlation with machine failure (**0.19**), confirming the boxplot findings

### 3. Preprocessing
- Encoded the categorical `Type` column using `pd.get_dummies()`
- Split data using `stratify=y` to preserve the 96.6% / 3.4% class ratio in both train and test sets
- Scaled features using `StandardScaler` — fitted only on training data to prevent data leakage

### 4. Modeling

Four approaches were tested to handle the class imbalance:

| Model | Precision (Failure) | Recall (Failure) | F1 (Failure) | Accuracy |
|---|---|---|---|---|
| Logistic Regression + class_weight | 0.14 | 0.82 | 0.24 | 0.82 |
| Random Forest (baseline) | 0.92 | 0.50 | 0.65 | 0.98 |
| Random Forest + class_weight | 0.91 | 0.43 | 0.58 | 0.97 |
| Random Forest + SMOTE | 0.46 | 0.76 | 0.58 | 0.96 |
| Random Forest + SMOTE + threshold 0.6 | 0.55 | 0.71 | 0.62 | 0.97 |
| **XGBoost + scale_pos_weight** | **0.71** | **0.76** | **0.74** | **0.98** |

### 5. Final Model — XGBoost

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=42
)
```

`scale_pos_weight` tells XGBoost to give more importance to the minority class (failures) during training — without needing to generate synthetic data.

---

## Results

```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1932
           1       0.71      0.76      0.74        68

    accuracy                           0.98      2000
   macro avg       0.85      0.88      0.86      2000
weighted avg       0.98      0.98      0.98      2000
```

The final XGBoost model:
- Catches **76% of actual machine failures** before they happen
- Maintains **98% overall accuracy**
- Achieves the best **F1-score of 0.74** on the failure class across all models tested

---

## Key Learnings

- **Accuracy is misleading** for imbalanced datasets — always evaluate with precision, recall, and F1
- **Data leakage** must be identified and removed early — failure sub-type columns were dropped for this reason
- **Logistic Regression** had high recall but very low precision — too many false alarms
- **SMOTE** helps recall but can hurt precision — threshold tuning partially compensates
- **XGBoost with scale_pos_weight** handled class imbalance more effectively than all other approaches

---

## Libraries Used

- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, modeling, evaluation
- `imbalanced-learn` — SMOTE oversampling
- `xgboost` — final model

---

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) — UCI Machine Learning Repository
