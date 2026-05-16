"""Gold — Build final model input by joining all feature groups."""

import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.io import write_parquet, read_parquet

# Import the new finalized architecture feature modules
from src.gold.feature_historical import build_historical_features
from src.gold.feature_capacity import build_capacity_features
from src.gold.feature_poi import build_poi_features
from src.gold.feature_seasonality import build_seasonality_features
from src.gold.feature_censoring import build_censoring_features
from src.modeling.censor_probability import calculate_censor_probability

logger = get_logger("gold.build_model_input")

def build_model_input(config: dict) -> None:
    logger.info("Building final gold model input (Stage 2 Features)...")
    
    # 1. Historical Features
    df_hist = build_historical_features(config)
    
    # 2. Capacity Features (Requires historical for efficiency)
    df_cap = build_capacity_features(config, df_hist)
    
    # 3. Spatial Features
    df_poi = build_poi_features(config)
    
    # 4. Temporal/Seasonality Features
    df_seas = build_seasonality_features(config)
    
    # Get Distributor mapping
    txn_raw = read_parquet(config["paths"]["silver"]["transactions"])
    outlet_dist = txn_raw.groupby("Outlet_ID")["Distributor_ID"].first().reset_index()
    
    # Join everything onto the core Capacity/Outlet master
    df = df_cap.merge(df_hist, on="Outlet_ID", how="left")
    df = df.merge(df_poi, on="Outlet_ID", how="left")
    df = df.merge(outlet_dist, on="Outlet_ID", how="left")
    df = df.merge(df_seas, on="Distributor_ID", how="left")
    
    # Fill NAs for outlets that might be missing from some joins
    df = df.fillna(0)
    
    # 5. Censor Features (Calculated on the merged DataFrame)
    df_cens = build_censoring_features(df, config)
    df = df.merge(df_cens, on="Outlet_ID", how="left")
    
    # 6. Stage 3 Censor Probability Model
    df["censor_probability"] = calculate_censor_probability(df)
    
    # Save output
    out_path = config["paths"]["gold"]["model_input"]
    write_parquet(df, out_path)
    logger.info(f"Gold model input successfully built and saved to {out_path}. Shape: {df.shape}.")
    
    # Optional cleanup of deprecated files to avoid confusion
    for old_file in ["src/gold/feature_transaction.py", "src/gold/feature_outlet_profile.py"]:
        if os.path.exists(old_file):
            logger.info(f"Removing deprecated file: {old_file}")
            os.remove(old_file)
