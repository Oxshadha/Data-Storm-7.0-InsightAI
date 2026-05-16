"""Modeling — Core demand estimation."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("modeling.demand_estimation")

def estimate_demand(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Estimate Latent Potential for January 2026.
    Since true potential is unknown (unsupervised), we use a rule-based 
    expert system combining peer group performance, peak volume, and censoring flags.
    """
    logger.info("Estimating latent potential demand...")
    
    res = df[["Outlet_ID"]].copy()
    
    # Base estimate is recent average adjusted by seasonality
    seas_adj = 1.0 + (df["Jan_Seasonality_Score"].fillna(0) * 0.1)
    base_est = df["recent_3m_avg"] * seas_adj
    
    # Model 1: Peer Group Uplift (90th percentile of similar outlets)
    peer_est = df["Peer_Group_90th_Vol"].fillna(base_est)
    
    # Model 2: Historical Peak (outlet's own best month)
    peak_est = df["peak_volume"].fillna(base_est)
    
    # Is the outlet constrained/censored?
    is_censored = (df["Is_Plateaued"] == 1) | (df["is_stockout_censored"] == 1) | (df["is_high_return_censored"] == 1)
    
    # Estimate calculation
    # If censored, true potential is much higher than base estimate. We take the max of its peak or its peer's 90th percentile.
    # If not censored, true potential is close to its base estimate or peak.
    estimated_potential = np.where(
        is_censored,
        np.maximum.reduce([base_est * 1.1, peer_est, peak_est * 1.05]),
        np.maximum.reduce([base_est, peak_est * 0.9])
    )
    
    # Add a growth factor for 2026 based on market assumption
    market_growth = 1.05
    estimated_potential = estimated_potential * market_growth
    
    res["Latent_Potential"] = np.round(estimated_potential, 2)
    
    logger.info("Finished demand estimation.")
    return res
