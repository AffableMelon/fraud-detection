#!/usr/bin/env python3
"""
Task 2: Model Building & Training
Trains fraud detection models for e-commerce and credit card datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, auc, f1_score,
    average_precision_score, roc_auc_score
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a trained model and return metrics"""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Performance")
    print(f"{'='*60}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    return {
        'model_name': model_name,
        'f1_score': f1,
        'auc_pr': auc_pr,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def train_ecommerce_models():
    """Train models for e-commerce fraud detection"""
    print("\n" + "="*80)
    print("PART 1: E-COMMERCE FRAUD DETECTION")
    print("="*80)
    
    # Load data
    print("\nLoading e-commerce fraud data...")
    ecom_df = pd.read_csv('../data/processed/fraud_data_processed.csv')
    print(f"Dataset shape: {ecom_df.shape}")
    print(f"Fraud ratio: {ecom_df['class'].mean():.4f}")
    
    # Prepare features and target
    non_numeric_cols = ['user_id', 'signup_time', 'purchase_time', 'device_id', 
                        'source', 'browser', 'sex', 'country']
    cols_to_drop = [col for col in non_numeric_cols if col in ecom_df.columns]
    cols_to_drop.append('class')
    
    X = ecom_df.drop(columns=cols_to_drop, errors='ignore')
    y = ecom_df['class']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    print(f"Features: {list(X.columns)}")
    print(f"Missing values after imputation: {X.isnull().sum().sum()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    
    # Apply SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced training set: {X_train_balanced.shape[0]}")
    
    results = []
    models = {}
    
    # 1. Logistic Regression
    print("\n" + "-"*60)
    print("Training Logistic Regression...")
    print("-"*60)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_balanced, y_train_balanced)
    lr_results = evaluate_model(lr, X_train_balanced, X_test, y_train_balanced, y_test, 
                                "E-commerce - Logistic Regression")
    results.append(lr_results)
    models['lr'] = lr
    
    # 2. Random Forest
    print("\n" + "-"*60)
    print("Training Random Forest...")
    print("-"*60)
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                                 min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_balanced, y_train_balanced)
    rf_results = evaluate_model(rf, X_train_balanced, X_test, y_train_balanced, y_test,
                                "E-commerce - Random Forest")
    results.append(rf_results)
    models['rf'] = rf
    
    # 3. XGBoost
    print("\n" + "-"*60)
    print("Training XGBoost...")
    print("-"*60)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                        eval_metric='logloss')
    xgb.fit(X_train_balanced, y_train_balanced)
    xgb_results = evaluate_model(xgb, X_train_balanced, X_test, y_train_balanced, y_test,
                                 "E-commerce - XGBoost")
    results.append(xgb_results)
    models['xgb'] = xgb
    
    # 4. LightGBM
    print("\n" + "-"*60)
    print("Training LightGBM...")
    print("-"*60)
    lgbm = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                          verbose=-1)
    lgbm.fit(X_train_balanced, y_train_balanced)
    lgbm_results = evaluate_model(lgbm, X_train_balanced, X_test, y_train_balanced, y_test,
                                  "E-commerce - LightGBM")
    results.append(lgbm_results)
    models['lgbm'] = lgbm
    
    # Model comparison
    print("\n" + "="*80)
    print("E-COMMERCE MODEL COMPARISON")
    print("="*80)
    results_df = pd.DataFrame([{
        'Model': r['model_name'],
        'F1-Score': r['f1_score'],
        'AUC-PR': r['auc_pr'],
        'ROC-AUC': r['roc_auc']
    } for r in results])
    print(results_df.to_string(index=False))
    
    # Select best model (highest AUC-PR)
    best_idx = results_df['AUC-PR'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model_key = list(models.keys())[best_idx]
    best_model = models[best_model_key]
    
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"  AUC-PR: {results_df.loc[best_idx, 'AUC-PR']:.4f}")
    print(f"  F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': list(X.columns),
        'results': results_df
    }


def train_creditcard_models():
    """Train models for credit card fraud detection"""
    print("\n" + "="*80)
    print("PART 2: CREDIT CARD FRAUD DETECTION")
    print("="*80)
    
    # Load data
    print("\nLoading credit card fraud data...")
    cc_df = pd.read_csv('../data/raw/creditcard.csv')
    print(f"Dataset shape: {cc_df.shape}")
    print(f"Fraud ratio: {cc_df['Class'].mean():.4f}")
    
    # Prepare features and target
    X = cc_df.drop('Class', axis=1)
    y = cc_df['Class']
    
    print(f"Number of features: {X.shape[1]}")
    print(f"Missing values: {X.isnull().sum().sum()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    
    # Apply SMOTE
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced training set: {X_train_balanced.shape[0]}")
    
    results = []
    models = {}
    
    # 1. Logistic Regression
    print("\n" + "-"*60)
    print("Training Logistic Regression...")
    print("-"*60)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_balanced, y_train_balanced)
    lr_results = evaluate_model(lr, X_train_balanced, X_test, y_train_balanced, y_test,
                                "Credit Card - Logistic Regression")
    results.append(lr_results)
    models['lr'] = lr
    
    # 2. Random Forest
    print("\n" + "-"*60)
    print("Training Random Forest...")
    print("-"*60)
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                                 min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_balanced, y_train_balanced)
    rf_results = evaluate_model(rf, X_train_balanced, X_test, y_train_balanced, y_test,
                                "Credit Card - Random Forest")
    results.append(rf_results)
    models['rf'] = rf
    
    # 3. XGBoost
    print("\n" + "-"*60)
    print("Training XGBoost...")
    print("-"*60)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                        eval_metric='logloss')
    xgb.fit(X_train_balanced, y_train_balanced)
    xgb_results = evaluate_model(xgb, X_train_balanced, X_test, y_train_balanced, y_test,
                                 "Credit Card - XGBoost")
    results.append(xgb_results)
    models['xgb'] = xgb
    
    # 4. LightGBM
    print("\n" + "-"*60)
    print("Training LightGBM...")
    print("-"*60)
    lgbm = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                          verbose=-1)
    lgbm.fit(X_train_balanced, y_train_balanced)
    lgbm_results = evaluate_model(lgbm, X_train_balanced, X_test, y_train_balanced, y_test,
                                  "Credit Card - LightGBM")
    results.append(lgbm_results)
    models['lgbm'] = lgbm
    
    # Model comparison
    print("\n" + "="*80)
    print("CREDIT CARD MODEL COMPARISON")
    print("="*80)
    results_df = pd.DataFrame([{
        'Model': r['model_name'],
        'F1-Score': r['f1_score'],
        'AUC-PR': r['auc_pr'],
        'ROC-AUC': r['roc_auc']
    } for r in results])
    print(results_df.to_string(index=False))
    
    # Select best model (highest AUC-PR)
    best_idx = results_df['AUC-PR'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model_key = list(models.keys())[best_idx]
    best_model = models[best_model_key]
    
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"  AUC-PR: {results_df.loc[best_idx, 'AUC-PR']:.4f}")
    print(f"  F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': list(X.columns),
        'results': results_df
    }


def main():
    """Main training pipeline"""
    import os
    os.makedirs('../models', exist_ok=True)
    
    # Train e-commerce models
    ecom_results = train_ecommerce_models()
    
    # Train credit card models
    cc_results = train_creditcard_models()
    
    # Save models and artifacts
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    # Save e-commerce artifacts
    joblib.dump(ecom_results['best_model'], '../models/best_ecom_fraud_model.pkl')
    print(f"✓ Saved: ../models/best_ecom_fraud_model.pkl")
    
    joblib.dump(ecom_results['feature_names'], '../models/ecom_feature_names.pkl')
    print(f"✓ Saved: ../models/ecom_feature_names.pkl")
    
    ecom_test_data = {
        'X_test': ecom_results['X_test'],
        'y_test': ecom_results['y_test']
    }
    joblib.dump(ecom_test_data, '../models/ecom_test_data.pkl')
    print(f"✓ Saved: ../models/ecom_test_data.pkl")
    
    # Save credit card artifacts
    joblib.dump(cc_results['best_model'], '../models/best_creditcard_fraud_model.pkl')
    print(f"✓ Saved: ../models/best_creditcard_fraud_model.pkl")
    
    joblib.dump(cc_results['feature_names'], '../models/cc_feature_names.pkl')
    print(f"✓ Saved: ../models/cc_feature_names.pkl")
    
    cc_test_data = {
        'X_test': cc_results['X_test'],
        'y_test': cc_results['y_test']
    }
    joblib.dump(cc_test_data, '../models/cc_test_data.pkl')
    print(f"✓ Saved: ../models/cc_test_data.pkl")
    
    # Save summary
    print("\n" + "="*80)
    print("TASK 2 SUMMARY")
    print("="*80)
    print(f"\nE-commerce Best Model: {ecom_results['best_model_name']}")
    print(ecom_results['results'].to_string(index=False))
    print(f"\nCredit Card Best Model: {cc_results['best_model_name']}")
    print(cc_results['results'].to_string(index=False))
    

if __name__ == "__main__":
    main()
