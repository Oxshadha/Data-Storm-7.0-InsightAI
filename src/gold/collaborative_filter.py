"""
Gold Layer — Collaborative Filtering (Portfolio Imbalance Correction)

Implements the "Portfolio Imbalance" correction logic.
If a shop (e.g., an Eatery) only sells Cola, but its peer group proves
that Eateries in that region also need Ginger Beer, this script uses 
collaborative filtering to "fill in" the missing portfolio.

Steps:
1. Define Peer Groups (Distributor_ID + Outlet_Type + Dynamic_Tier).
2. Identify "Core SKUs" for each Peer Group (SKUs sold by >50% of the peers).
3. Calculate the median monthly volume for each Core SKU in that Peer Group.
4. For every outlet, identify if it is missing a Core SKU.
5. Generate synthetic transaction rows with the imputed median volume.
6. Export `transactions_cf_enhanced.parquet` to be consumed by the ABT builder.

Usage:
    python -m src.gold.collaborative_filter
"""

import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.logger import get_logger

logger = get_logger("gold.collaborative_filter")

def run_collaborative_filtering(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Gold: Running Collaborative Filtering for Portfolio Imbalance")

    paths = config["paths"]

    # 1. Load Data
    logger.info("Loading Transactions and Master Data...")
    trans_df = read_parquet(paths["silver"]["transactions"])
    master_df = read_parquet(paths["silver"]["outlet_master"])

    # 2. Define Peer Groups
    logger.info("Defining Peer Groups...")
    # Get Distributor_ID from transactions
    outlet_dist = trans_df[["Outlet_ID", "Distributor_ID"]].drop_duplicates()
    # Assume 1 outlet -> 1 distributor. If not, just take the first.
    outlet_dist = outlet_dist.groupby("Outlet_ID").first().reset_index()
    
    master_df = master_df.merge(outlet_dist, on="Outlet_ID", how="left")
    master_df["Distributor_ID"] = master_df["Distributor_ID"].astype(str).fillna("Unknown")
    
    # Peer Group is defined by Region (Distributor), Business Model (Outlet_Type), and Size (Dynamic_Tier)
    master_df["Peer_Group"] = master_df["Distributor_ID"] + "_" + master_df["Outlet_Type"].astype(str) + "_" + master_df["Dynamic_Tier"].astype(str)
    
    # Merge Peer Group into transactions
    # Note: We group by Outlet and SKU to see what a shop historically sells
    shop_portfolio = trans_df.groupby(["Outlet_ID", "SKU_ID"]).size().reset_index(name="Tx_Count")
    shop_portfolio = shop_portfolio.merge(master_df[["Outlet_ID", "Peer_Group"]], on="Outlet_ID", how="left")
    
    # 3. Identify Core SKUs per Peer Group
    logger.info("Calculating Core Portfolios via Peer Behavior...")
    # Count total unique shops per peer group
    peer_group_sizes = master_df.groupby("Peer_Group").size().reset_index(name="Total_Shops_In_Group")
    
    # Count how many shops sell each SKU within the peer group
    sku_penetration = shop_portfolio.groupby(["Peer_Group", "SKU_ID"])["Outlet_ID"].nunique().reset_index(name="Shops_Selling_SKU")
    sku_penetration = sku_penetration.merge(peer_group_sizes, on="Peer_Group")
    
    # Penetration %
    sku_penetration["Penetration_Pct"] = sku_penetration["Shops_Selling_SKU"] / sku_penetration["Total_Shops_In_Group"]
    
    # A Core SKU is one that >50% of your exact peers sell
    core_skus = sku_penetration[sku_penetration["Penetration_Pct"] > 0.50].copy()
    
    # 4. Calculate Median Monthly Volume for Imputation
    logger.info("Calculating Imputation Volumes...")
    # Get monthly volume per shop for each SKU
    monthly_sku_vol = trans_df.groupby(["Outlet_ID", "Year", "Month", "SKU_ID"])["Volume_Liters"].sum().reset_index()
    monthly_sku_vol = monthly_sku_vol.merge(master_df[["Outlet_ID", "Peer_Group"]], on="Outlet_ID", how="left")
    
    # Find the median volume of that SKU inside that Peer Group (excluding zeros)
    median_vols = monthly_sku_vol.groupby(["Peer_Group", "SKU_ID"])["Volume_Liters"].median().reset_index(name="Imputed_Volume")
    core_skus = core_skus.merge(median_vols, on=["Peer_Group", "SKU_ID"], how="left")
    
    # 5. Impute Missing Portfolios
    logger.info("Finding Missing SKUs and Imputing Latent Demand...")
    synthetic_rows = []
    
    # For every outlet, get its peer group, look at the core SKUs, and see if it's missing them
    for outlet_id, group in shop_portfolio.groupby("Outlet_ID"):
        peer_group = group["Peer_Group"].iloc[0]
        if pd.isna(peer_group):
            continue
            
        shop_skus = set(group["SKU_ID"])
        group_core = core_skus[core_skus["Peer_Group"] == peer_group]
        
        for _, row in group_core.iterrows():
            core_sku = row["SKU_ID"]
            if core_sku not in shop_skus:
                # Portfolio Imbalance detected!
                imputed_vol = row["Imputed_Volume"]
                if pd.notna(imputed_vol) and imputed_vol > 0:
                    # Create a synthetic row for every active month the shop existed
                    # To keep it simple, we add it to their most recent active month 
                    # OR spread it across all months they had transactions.
                    # We will spread it across all months they were active to maintain time-series integrity.
                    shop_months = trans_df[trans_df["Outlet_ID"] == outlet_id][["Year", "Month", "Distributor_ID"]].drop_duplicates()
                    
                    for _, m_row in shop_months.iterrows():
                        synthetic_rows.append({
                            "Outlet_ID": outlet_id,
                            "Distributor_ID": m_row["Distributor_ID"],
                            "Year": m_row["Year"],
                            "Month": m_row["Month"],
                            "SKU_ID": core_sku,
                            "Volume_Liters": imputed_vol,
                            "Total_Bill_Value": 0, # Imputed, no revenue
                            "Is_Return": 0,
                            "Lazy_Rep_Flag": 0,
                            "Is_CF_Imputed": 1  # Flag for tracking
                        })

    if synthetic_rows:
        cf_df = pd.DataFrame(synthetic_rows)
        logger.info(f"Generated {len(cf_df):,} synthetic transaction rows via Collaborative Filtering.")
        
        # Add tracking flag to original data
        trans_df["Is_CF_Imputed"] = 0
        
        # Combine
        enhanced_trans_df = pd.concat([trans_df, cf_df], ignore_index=True)
    else:
        logger.info("No missing core SKUs found. Network portfolio is perfectly balanced.")
        trans_df["Is_CF_Imputed"] = 0
        enhanced_trans_df = trans_df

    # 6. Save Output
    out_path = paths["gold"]["root"] + "/transactions_cf_enhanced.parquet"
    write_parquet(enhanced_trans_df, out_path)
    
    logger.info(f"Saved Enhanced Transactions: {out_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_collaborative_filtering()
