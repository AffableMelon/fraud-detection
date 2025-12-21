import pandas as pd
import numpy as np

class FraudDataLoader:
    def __init__(self):
        pass

    def convert_ip_to_int(self, df, col='ip_address'):
        # Force conversion to int64 for consistency
        df[col] = df[col].astype(float).astype(np.int64)
        return df

    def merge_ip_to_country(self, fraud_df, ip_df):
        # 1. Force identical types
        fraud_df['ip_address'] = fraud_df['ip_address'].astype(np.int64)
        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(np.int64)
        ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(np.int64)

        # 2. Sort
        fraud_df = fraud_df.sort_values('ip_address')
        ip_df = ip_df.sort_values('lower_bound_ip_address')
        
        # 3. Perform the asof merge
        merged = pd.merge_asof(
            fraud_df, ip_df, 
            left_on='ip_address', 
            right_on='lower_bound_ip_address'
        )
        
        # 4. HANDLE NaNs IMMEDIATELY AFTER MERGE
        # If an IP has no lower bound match, upper_bound will be NaN. 
        # We fill it with 0 so the np.where comparison doesn't break.
        merged['upper_bound_ip_address'] = merged['upper_bound_ip_address'].fillna(0)
        
        # 5. Check upper bound constraint
        merged['country'] = np.where(
            merged['ip_address'] <= merged['upper_bound_ip_address'], 
            merged['country'], 
            'Unknown'
        )
        
        # Final cleanup of any remaining NaNs in country
        merged['country'] = merged['country'].fillna('Unknown')
        
        return merged

    def engineer_features(self, df):
        # (Keep the rest of your engineering code here...)
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['device_id_count'] = df.groupby('device_id')['device_id'].transform('count')
        df['ip_count'] = df.groupby('ip_address')['ip_address'].transform('count')
        return df