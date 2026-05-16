"""Modeling — Model C: Feature-Space Peer Benchmark (Business Reality Anchor)."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("modeling.model_peer_benchmark")

def calculate_peer_benchmark(df_train: pd.DataFrame, df_test: pd.DataFrame) -> np.ndarray:
    """Calculate the 95th percentile benchmark of feature-space peers."""
    logger.info("Calculating Feature-Space Peer Benchmark (Stage 4 - Model C)...")
    
    # We combine for grouping, but only calculate quantiles using historical training data
    df = pd.concat([df_train, df_test], axis=0).drop_duplicates(subset=["Outlet_ID"]).copy()
    
    # 1. Define POI Density Tier (using urban_density_score)
    # We use qcut to create quartiles: Low, Medium, High, VeryHigh
    if "poi_density_tier" not in df.columns:
        # rank method="first" handles duplicate values safely
        df["poi_density_tier"] = pd.qcut(df["urban_density_score"].rank(method="first"), q=4, labels=["Low", "Med", "High", "VeryHigh"])
        
    # 2. Define Cooler Band
    if "cooler_band" not in df.columns:
        df["cooler_band"] = pd.cut(df["Cooler_Count"].fillna(0), bins=[-1, 0, 1, 3, 10], labels=["0", "1", "2-3", "4+"])
        
    # Ensure df_train has these new bands
    train_mapped = df_train.merge(df[["Outlet_ID", "poi_density_tier", "cooler_band"]], on="Outlet_ID", how="left")
    test_mapped = df_test.merge(df[["Outlet_ID", "poi_density_tier", "cooler_band"]], on="Outlet_ID", how="left")
    
    # 3. Define the precise similarity dimensions
    # 'spatial_cluster' captures the macro-region
    peer_cols = ["Outlet_Type", "Outlet_Size_Score", "cooler_band", "Distributor_ID", "poi_density_tier", "spatial_cluster"]
    
    # Calculate the 95th percentile volume per exact peer group (ONLY using historical data)
    peer_p95 = train_mapped.groupby(peer_cols)["total_volume"].quantile(0.95).reset_index(name="peer_p95_benchmark")
    
    # Map back to test set
    test_merged = test_mapped.merge(peer_p95, on=peer_cols, how="left")
    
    # 4. Fallback 1: Broad Category + Size + Region
    fallback_1_cols = ["Outlet_Type", "Outlet_Size_Score", "spatial_cluster"]
    fallback_1 = train_mapped.groupby(fallback_1_cols)["total_volume"].quantile(0.95).reset_index(name="fb1_benchmark")
    test_merged = test_merged.merge(fallback_1, on=fallback_1_cols, how="left")
    test_merged["peer_p95_benchmark"] = test_merged["peer_p95_benchmark"].fillna(test_merged["fb1_benchmark"])
    
    # 5. Fallback 2: Broad Category Only (Global baseline for rare combos)
    fallback_2 = train_mapped.groupby("Outlet_Type")["total_volume"].quantile(0.95).reset_index(name="fb2_benchmark")
    test_merged = test_merged.merge(fallback_2, on="Outlet_Type", how="left")
    test_merged["peer_p95_benchmark"] = test_merged["peer_p95_benchmark"].fillna(test_merged["fb2_benchmark"])
    
    # Ultimate global fallback (should theoretically never hit)
    global_p95 = train_mapped["total_volume"].quantile(0.95)
    test_merged["peer_p95_benchmark"] = test_merged["peer_p95_benchmark"].fillna(global_p95)
    
    logger.info("Finished Peer Benchmark calculations.")
    return test_merged["peer_p95_benchmark"].values
