"""Gold — Temporal and Seasonality Features."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger("gold.feature_seasonality")

def build_seasonality_features(config: dict) -> pd.DataFrame:
    """Build Temporal features reflecting January target conditions and historical behavior."""
    logger.info("Building Temporal/Seasonality features...")
    
    target_month = config["project"]["target_month"]  # 1
    target_year = config["project"]["target_year"]    # 2026
    
    # 1. Distributor Seasonal Score
    dist = read_parquet(config["paths"]["silver"]["distributor_seasonality"])
    jan_dist = dist[dist["Month"] == target_month].copy()
    
    seas_map = {"Un-Favorable": -1, "Moderate": 0, "Favorable": 1}
    jan_dist["distributor_seasonal_score"] = jan_dist["Seasonality_Index"].map(seas_map)
    dist_features = jan_dist.groupby("Distributor_ID")["distributor_seasonal_score"].mean().reset_index()
    
    # 2. Holiday Counts for Target Month
    try:
        holidays = read_parquet(config["paths"]["silver"]["holiday_list"])
        holidays["Date"] = pd.to_datetime(holidays["Date"])
        jan_hols = holidays[(holidays["Date"].dt.year == target_year) & (holidays["Date"].dt.month == target_month)]
        
        # We attach these constants to the dist_features just to carry them forward into the single row-per-outlet join
        dist_features["target_month_holidays_total"] = len(jan_hols)
        
        # Holiday type counts
        type_counts = jan_hols["Holiday_Type"].value_counts()
        dist_features["target_month_public_hols"] = type_counts.get("Public", 0)
        dist_features["target_month_bank_hols"] = type_counts.get("Bank", 0)
        dist_features["target_month_poya_hols"] = type_counts.get("Poya Day", 0)
        dist_features["target_month_mercantile_hols"] = type_counts.get("Mercantile", 0)
    except Exception as e:
        logger.warning(f"Could not load/process holidays: {e}")
        dist_features["target_month_holidays_total"] = 0
        
    logger.info("Finished building Temporal/Seasonality features.")
    return dist_features
