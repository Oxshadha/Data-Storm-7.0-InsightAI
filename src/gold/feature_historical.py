"""Gold — Historical Demand Features."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger("gold.feature_historical")

def build_historical_features(config: dict) -> pd.DataFrame:
    """Build historical demand features at the Outlet level."""
    logger.info("Building historical demand features...")
    
    df = read_parquet(config["paths"]["silver"]["transactions"])
    
    # Ensure sorted order for time-based metrics
    df = df.sort_values(["Outlet_ID", "Year", "Month"])
    
    # 1. Monthly Aggregations
    monthly = df.groupby(["Outlet_ID", "Year", "Month"]).agg(
        monthly_volume=("Volume_Liters", "sum"),
        monthly_bill=("Total_Bill_Value", "sum")
    ).reset_index()
    
    # Add absolute month index for recency and trend calculations
    monthly["month_idx"] = (monthly["Year"] - monthly["Year"].min()) * 12 + monthly["Month"]
    max_month_idx = monthly["month_idx"].max()
    
    # Outlet level global metrics
    # We gracefully handle Is_Return/Is_Zero depending on whether silver layer created them
    outlets = df.groupby("Outlet_ID").agg(
        total_volume=("Volume_Liters", "sum"),
        total_bill=("Total_Bill_Value", "sum"),
        sku_diversity=("SKU_ID", "nunique")
    ).reset_index()
    
    if "Is_Return" in df.columns:
        outlets["return_ratio"] = df.groupby("Outlet_ID")["Is_Return"].mean().values
    else:
        outlets["return_ratio"] = df.groupby("Outlet_ID").apply(lambda x: (x["Volume_Liters"] < 0).mean()).values
        
    if "Is_Zero" in df.columns:
        outlets["zero_volume_ratio"] = df.groupby("Outlet_ID")["Is_Zero"].mean().values
    else:
        outlets["zero_volume_ratio"] = df.groupby("Outlet_ID").apply(lambda x: (x["Volume_Liters"] == 0).mean()).values
    
    outlets["avg_bill_per_liter"] = outlets["total_bill"] / outlets["total_volume"].replace(0, np.nan)
    
    # Active months & Recency
    active_months = monthly.groupby("Outlet_ID").agg(
        active_month_count=("month_idx", "count"),
        last_active_month=("month_idx", "max")
    ).reset_index()
    active_months["recency"] = max_month_idx - active_months["last_active_month"]
    
    # Recent N-month averages
    def get_recent_avg(n_months):
        recent_df = monthly[monthly["month_idx"] > (max_month_idx - n_months)]
        return recent_df.groupby("Outlet_ID")["monthly_volume"].mean().reset_index(name=f"recent_{n_months}m_avg")

    recent_3m = get_recent_avg(3)
    recent_6m = get_recent_avg(6)
    recent_12m = get_recent_avg(12)
    
    # EMA trend (Exponential Moving Average of volume)
    def calc_ema(series, alpha=0.3):
        if len(series) == 0: return 0
        return series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        
    ema_trend = monthly.groupby("Outlet_ID")["monthly_volume"].apply(calc_ema).reset_index(name="ema_trend")
    
    # Target Month (January) Historical Behavior Ratio
    target_month = config.get("project", {}).get("target_month", 1)
    jan_vols = monthly[monthly["Month"] == target_month].groupby("Outlet_ID")["monthly_volume"].mean().reset_index(name="jan_avg_vol")
    all_vols = monthly.groupby("Outlet_ID")["monthly_volume"].mean().reset_index(name="overall_avg_vol")
    jan_ratio = jan_vols.merge(all_vols, on="Outlet_ID")
    jan_ratio["jan_historical_ratio"] = jan_ratio["jan_avg_vol"] / jan_ratio["overall_avg_vol"].replace(0, np.nan)
    
    # Peak and Percentiles
    monthly_stats = monthly.groupby("Outlet_ID").agg(
        historical_peak=("monthly_volume", "max"),
        historical_p90=("monthly_volume", lambda x: np.percentile(x, 90)),
        historical_p95=("monthly_volume", lambda x: np.percentile(x, 95)),
        volatility_cv=("monthly_volume", lambda x: np.std(x) / (np.mean(x) + 1e-6))
    ).reset_index()
    
    # Merge all
    res = outlets.merge(active_months[["Outlet_ID", "active_month_count", "recency"]], on="Outlet_ID", how="left")
    res = res.merge(recent_3m, on="Outlet_ID", how="left")
    res = res.merge(recent_6m, on="Outlet_ID", how="left")
    res = res.merge(recent_12m, on="Outlet_ID", how="left")
    res = res.merge(ema_trend, on="Outlet_ID", how="left")
    res = res.merge(jan_ratio[["Outlet_ID", "jan_historical_ratio"]], on="Outlet_ID", how="left").fillna({"jan_historical_ratio": 1.0})
    res = res.merge(monthly_stats, on="Outlet_ID", how="left")
    
    res = res.fillna(0) # Fill NaNs for recent averages where outlet wasn't active
    
    logger.info(f"Built historical demand features for {len(res):,} outlets.")
    return res
