import sys
import os
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import FraudDataProcessor 

def main():
    fraud_path = 'data/raw/Fraud_Data.csv'
    ip_path = 'data/raw/IpAddress_to_Country.csv'
    credit_path = 'data/raw/creditcard.csv'
    output_path = 'data/processed/fraud_data_processed.csv'

    processor = FraudDataProcessor(
        fraud_path=fraud_path,
        ip_path=ip_path,
        credit_path=credit_path
    )
    
    try:
        # 1. Load Data (Includes the new error handling and validation)
        processor.load_data()
        
        # 2. Clean and Process (Includes IP-to-Country mapping and type conversion)
        processor.clean_and_process()
        
        
        # 3. Save the processed e-commerce data
        if processor.df_fraud is not None:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            processor.df_fraud.to_csv(output_path, index=False)
            logging.info(f"Success! Processed data saved to {output_path}")
            
    except Exception as e:
        logging.error(f"The data pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()