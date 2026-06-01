import pandas as pd
import numpy as np

print("Loading data...")
try:
    df = pd.read_parquet("data/gold/model_input.parquet")
    print(df["Is_Censored_Flag"].value_counts(normalize=True))
    
    censored = df[df["Is_Censored_Flag"] == 1]
    uncensored = df[df["Is_Censored_Flag"] == 0]
    
    print("\npoi_driver_catchment")
    print(f"Censored Mean: {censored['poi_driver_catchment'].mean():.2f} | Uncensored Mean: {uncensored['poi_driver_catchment'].mean():.2f}")
    
    print("\npoi_cannibal_risk")
    print(f"Censored Mean: {censored['poi_cannibal_risk'].mean():.2f} | Uncensored Mean: {uncensored['poi_cannibal_risk'].mean():.2f}")
    
except Exception as e:
    print(f"Error: {e}")
