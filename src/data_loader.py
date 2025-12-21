import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FraudDataProcessor:
    def __init__(self, fraud_path, ip_path, credit_path):
        self.fraud_path = fraud_path
        self.ip_path = ip_path
        self.credit_path = credit_path
        self.df_fraud = None
        self.df_ip = None
        self.df_credit = None

    def load_data(self):
        """Loads datasets with validation and error handling."""
        paths = {
            "Fraud Data": self.fraud_path,
            "IP Mapping": self.ip_path,
            "Credit Card Data": self.credit_path
        }
        
        for name, path in paths.items():
            if not os.path.exists(path):
                logging.error(f"File not found: {path}")
                raise FileNotFoundError(f"Missing essential file: {path}")
            
            try:
                if name == "Fraud Data":
                    self.df_fraud = pd.read_csv(path)
                elif name == "IP Mapping":
                    self.df_ip = pd.read_csv(path)
                else:
                    self.df_credit = pd.read_csv(path)
                logging.info(f"Successfully loaded {name} with {len(self.df_credit if name=='Credit Card Data' else self.df_fraud)} rows.")
            except Exception as e:
                logging.error(f"Error loading {name}: {e}")
                raise
    def clean_and_process(self):
            """Handles type conversion, merging, and initial feature engineering."""
            try:
                # 1. Process E-commerce Fraud Data
                self.df_fraud['signup_time'] = pd.to_datetime(self.df_fraud['signup_time'])
                self.df_fraud['purchase_time'] = pd.to_datetime(self.df_fraud['purchase_time'])
                self.df_fraud['ip_address'] = self.df_fraud['ip_address'].astype(np.int64)
                
                # --- FEATURE ENGINEERING: Time Since Signup ---
                self.df_fraud['time_since_signup'] = (
                    self.df_fraud['purchase_time'] - self.df_fraud['signup_time']
                ).dt.total_seconds()
                
                # 2. IP to Country Merge
                self.df_ip['lower_bound_ip_address'] = self.df_ip['lower_bound_ip_address'].astype(np.int64)
                self.df_ip['upper_bound_ip_address'] = self.df_ip['upper_bound_ip_address'].astype(np.int64)
                
                self.df_fraud = self.df_fraud.sort_values('ip_address')
                self.df_ip = self.df_ip.sort_values('lower_bound_ip_address')
                
                merged = pd.merge_asof(
                    self.df_fraud, self.df_ip, 
                    left_on='ip_address', 
                    right_on='lower_bound_ip_address'
                )
                
                # Validation: Check upper bound
                merged['country'] = np.where(
                    merged['ip_address'] <= merged['upper_bound_ip_address'], 
                    merged['country'], 
                    'Unknown'
                )
                self.df_fraud = merged.fillna({'country': 'Unknown'})
                logging.info("IP-to-Country mapping and initial feature engineering completed.")
    
            except KeyError as e:
                logging.error(f"Missing expected column during processing: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error during cleaning/processing: {e}")
                raise