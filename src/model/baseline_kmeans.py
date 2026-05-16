"""
Predictive Modeling — Route 1: Spatial Heuristic (Baseline)

Implements the K-Means + Percentile Mapping heuristic to estimate
latent unconstrained demand for censored outlets.

Steps:
1. Standardize numerical and spatial features.
2. Run K-Means clustering to create behavioral peer groups.
3. Calculate the 90th percentile volume of uncensored shops in each cluster.
4. Project this ceiling onto censored shops.

Usage:
    python -m src.model.baseline_kmeans
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils.config import load_config, resolve_path
from src.utils.io import read_parquet, write_parquet
from src.utils.logger import get_logger

logger = get_logger("model.baseline_kmeans")


def run_baseline_model(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Modeling: Running K-Means Baseline Heuristic")

    # 1. Load ABT
    abt_path = config["paths"]["gold"]["root"] + "/model_input.parquet"
    logger.info("Loading Analytical Base Table...")
    abt = read_parquet(abt_path)

    # 2. Select Features for Clustering
    # We want to group by structural, spatial, and temporal behaviors
    # We DO NOT cluster on Target (Total_Volume) or Is_Censored.
    feature_cols = [
        "poi_total_catchment",
        "poi_driver_catchment",
        "poi_cannibal_risk",
        "Tuition_Weekend_Surge",
        "Tourist_Peak_Multiplier",
        "Sports_Big_Match_Spike",
        "Health_Catchment_Spike",
        "Number_of_Weekends",
        "Holiday_Count",
        "Has_High_Footfall_Catchment",
        "Has_Cannibalization_Risk"
    ]
    
    # Also add encoded categoricals if helpful, e.g., Outlet_Type or Dynamic_Tier
    # We'll one-hot encode them simply
    logger.info("Encoding categorical features...")
    cat_cols = ["Outlet_Type", "Dynamic_Tier"]
    
    # Fill any lingering NaNs to prevent KMeans failure
    abt[feature_cols] = abt[feature_cols].fillna(0)
    abt[cat_cols] = abt[cat_cols].fillna("Unknown")
    
    abt_encoded = pd.get_dummies(abt[feature_cols + cat_cols], columns=cat_cols, drop_first=True)
    
    # 3. Standardize Features
    logger.info(f"Standardizing {len(abt_encoded.columns)} features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(abt_encoded)

    # 4. K-Means Clustering
    logger.info("Running K-Means with 30 clusters (this may take a moment)...")
    kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    abt["Cluster_ID"] = kmeans.fit_predict(X_scaled)
    
    # 5. Calculate 90th Percentile per Cluster on UNCENSORED shops
    logger.info("Calculating empirical ceilings (90th percentile) per cluster...")
    uncensored_df = abt[abt["Is_Censored"] == 0]
    
    # If a cluster has no uncensored shops, we fall back to the global 90th percentile
    global_p90 = uncensored_df["Total_Volume"].quantile(0.90)
    
    cluster_ceilings = (
        uncensored_df.groupby("Cluster_ID")["Total_Volume"]
        .quantile(0.90)
        .reset_index()
        .rename(columns={"Total_Volume": "Cluster_P90_Volume"})
    )
    
    abt = abt.merge(cluster_ceilings, on="Cluster_ID", how="left")
    abt["Cluster_P90_Volume"] = abt["Cluster_P90_Volume"].fillna(global_p90)
    
    # 6. Decensoring Projection
    logger.info("Projecting latent demand for censored outlets...")
    
    # Initialize predicted volume with actual volume
    abt["Predicted_Volume"] = abt["Total_Volume"]
    
    # For censored shops, project the ceiling (ensure it never drops below reality)
    censored_mask = abt["Is_Censored"] == 1
    abt.loc[censored_mask, "Predicted_Volume"] = np.maximum(
        abt.loc[censored_mask, "Total_Volume"],
        abt.loc[censored_mask, "Cluster_P90_Volume"]
    )
    
    # Calculate the Growth Gap
    abt["Growth_Gap_Liters"] = abt["Predicted_Volume"] - abt["Total_Volume"]
    
    total_growth = abt["Growth_Gap_Liters"].sum()
    censored_count = censored_mask.sum()
    
    logger.info(f"Baseline Complete!")
    logger.info(f"Identified {censored_count:,} monthly capacity constraints.")
    logger.info(f"Projected Unlocked Latent Demand: {total_growth:,.2f} Liters")
    
    # 7. Save outputs
    # Save the full enhanced ABT
    out_path = config["paths"]["gold"]["root"] + "/model_baseline_output.parquet"
    write_parquet(abt, out_path)
    
    # Generate the requested deliverable format: insightai_predictions.csv
    # The requirement usually asks for Outlet_ID and Predicted_Volume
    # Since our grain is Outlet-Month, we might need to aggregate to Outlet_ID
    # But usually predictions are either monthly or aggregated. Let's provide both.
    
    pred_path = resolve_path(config["paths"]["output"]["predictions"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Let's group by Outlet_ID to show total potential over the period
    outlet_preds = (
        abt.groupby("Outlet_ID")
        .agg(
            Actual_Volume=("Total_Volume", "sum"),
            Predicted_Volume=("Predicted_Volume", "sum"),
            Growth_Gap=("Growth_Gap_Liters", "sum")
        )
        .reset_index()
    )
    
    outlet_preds.to_csv(pred_path, index=False)
    logger.info(f"Saved submission file to {pred_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_baseline_model()
