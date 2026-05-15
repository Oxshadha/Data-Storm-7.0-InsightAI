"""Gold — Transaction behavioral features."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.io import read_parquet, write_parquet

logger = get_logger("gold.feature_transaction")

def build_transaction_features(config: dict) -> pd.DataFrame:
    """Build transaction features at the Outlet level."""
    logger.info("Building transaction features...")
    
    df = read_parquet(config["paths"]["silver"]["transactions"])
    
    # 1. Monthly Aggregations
    monthly = df.groupby(["Outlet_ID", "Year", "Month"]).agg(
        monthly_volume=("Volume_Liters", "sum"),
        monthly_bill=("Total_Bill_Value", "sum")
    ).reset_index()
    
    # Sort for recency
    monthly = monthly.sort_values(["Outlet_ID", "Year", "Month"])
    
    # Outlet level metrics
    outlets = df.groupby("Outlet_ID").agg(
        total_volume=("Volume_Liters", "sum"),
        total_bill=("Total_Bill_Value", "sum"),
        sku_diversity=("SKU_ID", "nunique"),
        return_ratio=("Is_Return", "mean"),
        zero_ratio=("Is_Zero", "mean"),
        total_transactions=("Volume_Liters", "count")
    ).reset_index()
    
    outlets["avg_price_per_liter"] = outlets["total_bill"] / outlets["total_volume"].replace(0, np.nan)
    
    # Active months & Recency
    active_months = monthly.groupby("Outlet_ID").size().reset_index(name="active_months")
    
    # Recent 3 months (Oct, Nov, Dec 2025)
    recent_3m = monthly[(monthly["Year"] == 2025) & (monthly["Month"] >= 10)]
    recent_3m_avg = recent_3m.groupby("Outlet_ID")["monthly_volume"].mean().reset_index(name="recent_3m_avg")
    
    # Peak and Percentiles
    monthly_stats = monthly.groupby("Outlet_ID").agg(
        peak_volume=("monthly_volume", "max"),
        p90_volume=("monthly_volume", lambda x: np.percentile(x, 90)),
        cv_volume=("monthly_volume", lambda x: np.std(x) / (np.mean(x) + 1e-6))
    ).reset_index()
    
    # Merge
    res = outlets.merge(active_months, on="Outlet_ID", how="left")
    res = res.merge(recent_3m_avg, on="Outlet_ID", how="left").fillna({"recent_3m_avg": 0})
    res = res.merge(monthly_stats, on="Outlet_ID", how="left")
    
    logger.info(f"Built transaction features for {len(res):,} outlets.")
    return res
