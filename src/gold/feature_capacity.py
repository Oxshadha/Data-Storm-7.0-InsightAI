"""Gold — Outlet Capacity Features."""

import pandas as pd
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger("gold.feature_capacity")

def build_capacity_features(config: dict, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build outlet capacity features based on the Final Architecture.
    Requires historical_df to calculate liters_per_cooler and cooler_efficiency_percentile.
    """
    logger.info("Building outlet capacity features...")
    df = read_parquet(config["paths"]["silver"]["outlet_master"])
    
    # 1. Outlet Size numeric encoding
    size_map = {"Unknown": 0, "Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4}
    df["Outlet_Size_Score"] = df["Outlet_Size"].map(size_map).fillna(0)
    
    # 2. Outlet Type encoding (One-Hot)
    dummies = pd.get_dummies(df["Outlet_Type"], prefix="Type", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    
    # 3. Capacity utilization metrics
    # We use historical peak to estimate their maximum demonstrated capacity
    if "historical_peak" in historical_df.columns:
        df = df.merge(historical_df[["Outlet_ID", "historical_peak"]], on="Outlet_ID", how="left")
        df["historical_peak"] = df["historical_peak"].fillna(0)
        
        # Liters per cooler (add 1 to avoid division by zero for outlets with 0 coolers)
        df["liters_per_cooler"] = df["historical_peak"] / (df["Cooler_Count"] + 1.0)
        
        # Cooler efficiency percentile compared to their peer Outlet Type
        df["cooler_efficiency_percentile"] = df.groupby("Outlet_Type")["liters_per_cooler"].rank(pct=True)
        
        # Drop the joined historical feature to avoid duplication in the final merge
        df = df.drop(columns=["historical_peak"])
    else:
        logger.warning("'historical_peak' not found. Skipping efficiency percentile calculation.")
        df["liters_per_cooler"] = 0
        df["cooler_efficiency_percentile"] = 0
        
    logger.info(f"Finished capacity features for {len(df):,} outlets.")
    return df
