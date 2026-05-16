"""Gold — Constraint and Censor Proxy Features."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("gold.feature_censoring")

def build_censoring_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build analytical indicators that an outlet might be operating under constraints."""
    logger.info("Building Constraint/Censor features...")
    
    cens = df[["Outlet_ID"]].copy()
    
    # 1. Plateau Score & Low Variance Flag
    # High plateau score implies variance is suspiciously low for an FMCG retail environment
    cv = df["volatility_cv"].fillna(1.0)
    cens["plateau_score"] = np.exp(-cv * 5)  # transforms low CV to high score in [0,1]
    cens["low_variance_flag"] = (cv < 0.15).astype(int)
    
    # 2. Operational Anomalies
    cens["high_return_anomaly"] = (df["return_ratio"].fillna(0) > 0.05).astype(int)
    cens["stockout_proxy_score"] = df["zero_volume_ratio"].fillna(0)
    
    # 3. High POI / Low Sales Mismatch
    # Identifies outlets in top 20% of POI gravity but bottom 50% of volume (chronically underperforming their location)
    if "poi_gravity_score" in df.columns:
        poi_p80 = df["poi_gravity_score"].quantile(0.80)
        vol_p50 = df["total_volume"].quantile(0.50)
        cens["high_poi_low_sales_mismatch"] = ((df["poi_gravity_score"] >= poi_p80) & (df["total_volume"] <= vol_p50)).astype(int)
    else:
        cens["high_poi_low_sales_mismatch"] = 0
        
    # 4. Low Cooler / High Peer Mismatch
    # Identifies outlets that are massively over-indexing on their cooler capacity compared to peers
    if "cooler_efficiency_percentile" in df.columns:
        cens["low_cooler_high_peer_mismatch"] = (df["cooler_efficiency_percentile"] > 0.90).astype(int)
    else:
        cens["low_cooler_high_peer_mismatch"] = 0
        
    # 5. Sudden Sales Drop
    # Detects recent severe constraint (e.g., credit lock or delivery stop)
    if "recent_3m_avg" in df.columns and "recent_12m_avg" in df.columns:
        recent_3 = df["recent_3m_avg"].fillna(0)
        recent_12 = df["recent_12m_avg"].fillna(0)
        drop_ratio = (recent_12 - recent_3) / (recent_12 + 1e-6)
        cens["sudden_sales_drop"] = (drop_ratio > 0.30).astype(int)
    else:
        cens["sudden_sales_drop"] = 0
        
    logger.info("Finished Constraint/Censor features.")
    return cens