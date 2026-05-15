"""Gold — Outlet profile features."""

import pandas as pd
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger("gold.feature_outlet_profile")

def build_outlet_profile_features(config: dict) -> pd.DataFrame:
    logger.info("Building outlet profile features...")
    df = read_parquet(config["paths"]["silver"]["outlet_master"])
    
    # Ordered size
    size_map = {"Unknown": 0, "Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4}
    df["Outlet_Size_Score"] = df["Outlet_Size"].map(size_map).fillna(0)
    
    # One hot encoding for Outlet_Type
    dummies = pd.get_dummies(df["Outlet_Type"], prefix="Type", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    
    logger.info("Finished outlet profile features.")
    return df
