"""Gold — Censoring indicators."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("gold.feature_censoring")

def build_censoring_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Identifies outlets that are likely constrained based on historical variance and volume drops.
    Expects df to contain 'cv_volume', 'active_months', 'peak_volume', 'Outlet_Type', 'Outlet_Size', 'Distributor_ID'.
    """
    logger.info("Building censoring features...")
    cens_df = df[["Outlet_ID"]].copy()
    
    cv_thresh = config.get("modeling", {}).get("censoring", {}).get("cv_threshold", 0.15)
    
    # Flag 1: Plateau Flag (Very low variance implies a strict delivery cap)
    cens_df["Is_Plateaued"] = np.where((df["cv_volume"] < cv_thresh) & (df["active_months"] >= 3), 1, 0)
    
    # Peer Group Definition
    peer_group_90th = df.groupby(["Outlet_Type", "Outlet_Size", "Distributor_ID"])["peak_volume"].transform(lambda x: x.quantile(0.90))
    df["Peer_Group_90th_Vol"] = peer_group_90th
    
    # Flag 2: Underperforming vs Peers
    cens_df["Is_Underperforming_Peer"] = np.where(df["peak_volume"] < (0.5 * df["Peer_Group_90th_Vol"]), 1, 0)
    
    # Flag 3: High return ratio
    if "return_ratio" in df.columns:
        cens_df["is_high_return_censored"] = (df["return_ratio"] > 0.05).astype(int)
        
    # Flag 4: Stockout proxy
    if "zero_ratio" in df.columns:
        cens_df["is_stockout_censored"] = (df["zero_ratio"] > 0.05).astype(int)
        
    logger.info("Finished censoring features.")
    return cens_df