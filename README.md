# Fraud Detection for E-commerce & Banking

This project develops a machine learning system to detect fraudulent
transactions in e-commerce and banking data. The system utilizes geolocation
mapping, advanced feature engineering, and class imbalance handling to identify
sophisticated fraud patterns.

## 🚀 Project Structure

```text
fraud-detection/
├── data/
│   ├── raw/             # Original datasets (Fraud_Data.csv, IpAddress_to_Country.csv)
│   └── processed/       # Cleaned data with engineered features
├── models/              # Serialized model files (.joblib)
├── notebooks/           # Jupyter notebooks for EDA and Feature Engineering
├── scripts/             # Python scripts for data processing and pipeline
├── src/                 # Reusable source code (Data loaders, transformers)
└── tests/               # Unit tests for project validation
```

## Create and activate a virtual environment and install modules
```bash
python -m venv .venv
source .venv/bin/activate  # For bash/zsh
# source .venv/bin/activate.fish  # For fish
pip install -r requirements.txt
```
--- 
##  Task 1: Data Analysis & Preprocessing
### Geolocation Integration

Transactions were mapped to countries by performing a range-based lookup
between IP addresses and global IP ranges. This allowed for the identification
of high-risk geographical regions.

### Feature Engineering

Key features created to improve model performance:
    1. time_since_signup: Duration between account creation and transaction.
    2. transaction_velocity: Frequency of device and IP address appearances.
    3. hour_of_day & day_of_week: Captured temporal fraud patterns.

### Handling Class Imbalance

Fraudulent cases accounted for only ~9% of the dataset. We utilized SMOTE
(Synthetic Minority Over-sampling Technique) to balance the training set,
ensuring the model learns the characteristics of the minority class without
losing information from the majority class.

## 🤖 Task 2: Modeling & Evaluation

We evaluated multiple classification algorithms to determine the most effective approach for detecting fraud:

    1. Logistic Regression: Used as a baseline to measure linear separability.
    2. Random Forest: Leveraged for its ability to capture non-linear interactions between features.

### Evaluation Metrics

Since accuracy is misleading in imbalanced datasets, we prioritize:

     1.Recall: The ability to catch as much fraud as possible.
     2.F1-Score: The harmonic mean of precision and recall.
     3.ROC-AUC: The model's ability to distinguish between fraud and legitimate transactions.
