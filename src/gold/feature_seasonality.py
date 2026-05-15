"""Gold — Seasonal pattern features."""

import pandas as pd
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger("gold.feature_seasonality")

def build_seasonality_features(config: dict) -> pd.DataFrame:
    logger.info("Building seasonality features...")
    dist = read_parquet(config["paths"]["silver"]["distributor_seasonality"])
    
    target_month = config["project"]["target_month"]
    
    # Average historical seasonality for the target month
    jan_dist = dist[dist["Month"] == target_month].copy()
    
    seas_map = {"Un-Favorable": -1, "Moderate": 0, "Favorable": 1}
    jan_dist["seas_score"] = jan_dist["Seasonality_Index"].map(seas_map)
    
    dist_features = jan_dist.groupby("Distributor_ID")["seas_score"].mean().reset_index()
    dist_features.rename(columns={"seas_score": "Jan_Seasonality_Score"}, inplace=True)
    
    logger.info("Finished seasonality features.")
    return dist_features
