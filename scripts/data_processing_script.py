import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import FraudDataLoader

def main():
    loader = FraudDataLoader()
    
    print("Loading raw data...")
    fraud_path = 'data/raw/Fraud_Data.csv'
    ip_path = 'data/raw/IpAddress_to_Country.csv'
    
    fraud_df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)
    
    # 1. Cleaning
    print("Cleaning data...")
    fraud_df = fraud_df.drop_duplicates()
    fraud_df = loader.convert_ip_to_int(fraud_df)
    
    # 2. Merging
    print("Merging with country data (Range Lookup)...")
    merged_df = loader.merge_ip_to_country(fraud_df, ip_df)
    
    # 3. Engineering
    print("Engineering features...")
    final_df = loader.engineer_features(merged_df)
    
    # 4. Save
    output_path = 'data/processed/fraud_data_processed.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Success! Processed data saved to {output_path}")

if __name__ == "__main__":
    main()