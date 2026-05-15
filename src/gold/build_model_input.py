"""Gold — Build final model input by joining all features."""

import pandas as pd
from src.utils.logger import get_logger
from src.utils.io import write_parquet, read_parquet
from src.gold.feature_transaction import build_transaction_features
from src.gold.feature_outlet_profile import build_outlet_profile_features
from src.gold.feature_poi import build_poi_features
from src.gold.feature_seasonality import build_seasonality_features
from src.gold.feature_censoring import build_censoring_features

logger = get_logger("gold.build_model_input")

def build_model_input(config: dict) -> None:
    logger.info("Building final gold model input...")
    
    df_txn = build_transaction_features(config)
    df_out = build_outlet_profile_features(config)
    df_poi = build_poi_features(config)
    df_seas = build_seasonality_features(config)
    
    txn_raw = read_parquet(config["paths"]["silver"]["transactions"])
    outlet_dist = txn_raw.groupby("Outlet_ID")["Distributor_ID"].first().reset_index()
    
    # Join everything
    df = df_out.merge(df_txn, on="Outlet_ID", how="left")
    df = df.merge(df_poi, on="Outlet_ID", how="left")
    df = df.merge(outlet_dist, on="Outlet_ID", how="left")
    df = df.merge(df_seas, on="Distributor_ID", how="left")
    
    # Fill NAs
    df = df.fillna(0)
    
    # Calculate censoring features using the fully merged df
    df_cens = build_censoring_features(df, config)
    df = df.merge(df_cens, on="Outlet_ID", how="left")
    
    out_path = config["paths"]["gold"]["model_input"]
    write_parquet(df, out_path)
    logger.info(f"Gold model input saved to {out_path} with shape {df.shape}.")
