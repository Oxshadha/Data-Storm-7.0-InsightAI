"""
Silver Layer — Outlet Master Cleaning.

Handles:
  1. Typo corrections (config-driven mapping)
  2. Case normalization + whitespace trimming
  3. DQ Checks (nulls, formats, valid values)
  4. Master Data Decay Detection (Dynamic_Tier clustering based on volume)
  
Usage:
    python -m src.silver.clean_outlet_master
"""

import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.logger import get_logger
from src.silver.dq_checks import (
    add_rejection_column,
    check_nulls,
    check_format,
    check_valid_values,
    split_by_rejection
)
from src.silver.quarantine import write_quarantine_outputs

logger = get_logger("silver.clean_outlet_master")


def compute_dynamic_tier(master_df: pd.DataFrame, trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 'Dynamic_Tier' by clustering outlets based on their average monthly volume.
    This resolves the "Master Data Decay" trap.
    """
    logger.info("Computing Dynamic_Tier based on actual transaction behavior...")
    
    # 1. Calculate avg monthly volume per outlet
    # Group by Outlet_ID and Month (using Year-Month combination implicitly or just unique months)
    # To keep it simple, sum volume per outlet, then divide by number of active months.
    
    monthly_vols = (
        trans_df.groupby(["Outlet_ID", "Year", "Month"], observed=True)["Volume_Liters"]
        .sum()
        .reset_index()
    )
    
    avg_vols = (
        monthly_vols.groupby("Outlet_ID", observed=True)["Volume_Liters"]
        .mean()
        .reset_index()
        .rename(columns={"Volume_Liters": "Avg_Monthly_Volume"})
    )
    
    # 2. Merge with master_df
    master_df = master_df.merge(avg_vols, on="Outlet_ID", how="left")
    master_df["Avg_Monthly_Volume"] = master_df["Avg_Monthly_Volume"].fillna(0)
    
    # 3. Cluster into Dynamic Tiers using quantiles
    # Let's map to 4 tiers matching the spirit of Small/Medium/Large/Extra Large
    quantiles = master_df["Avg_Monthly_Volume"].quantile([0.25, 0.5, 0.75]).to_dict()
    
    def assign_tier(vol):
        if vol <= quantiles[0.25]: return "Tier_4_Small"
        elif vol <= quantiles[0.5]: return "Tier_3_Medium"
        elif vol <= quantiles[0.75]: return "Tier_2_Large"
        else: return "Tier_1_ExtraLarge"
        
    master_df["Dynamic_Tier"] = master_df["Avg_Monthly_Volume"].apply(assign_tier)
    
    # 4. Flag decaying data for visibility
    # Simple check: If original is 'Small' but tier is 'Tier_1_ExtraLarge' etc.
    # We just trust Dynamic_Tier, but we log the discrepancy.
    discrepancy = master_df[
        (master_df["Outlet_Size"] == "Small") & (master_df["Dynamic_Tier"] == "Tier_1_ExtraLarge")
    ]
    logger.info(f"Master Data Decay: Found {len(discrepancy)} 'Small' outlets acting like 'Extra Large'.")
    
    return master_df


def clean_outlet_master(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    master_path = config["paths"]["bronze"]["outlet_master"]
    trans_path = config["paths"]["bronze"]["transactions"]
    clean_path = config["paths"]["silver"]["outlet_master"]
    
    logger.info("=" * 60)
    logger.info("Silver: Cleaning Outlet Master")
    
    df = read_parquet(master_path)
    
    # ── 1. Text Standardization (Before DQ checks) ──────────────────
    
    # Trim whitespace
    df["Outlet_Type"] = df["Outlet_Type"].str.strip()
    df["Outlet_Size"] = df["Outlet_Size"].astype(str).str.strip().replace("nan", np.nan)
    
    # Typo corrections
    type_map = config["dq_checks"]["outlet_master"]["outlet_type_corrections"]
    df["Outlet_Type"] = df["Outlet_Type"].replace(type_map)
    
    size_map = config["dq_checks"]["outlet_master"]["outlet_size_corrections"]
    df["Outlet_Size"] = df["Outlet_Size"].replace(size_map)
    
    # ── 2. DQ Checks (Quarantine Path) ─────────────────────────────
    
    df = add_rejection_column(df)
    
    mandatory = config["dq_checks"]["outlet_master"]["mandatory_columns"]
    df = check_nulls(df, mandatory)
    
    # Format checks for IDs
    df = check_format(df, "Outlet_ID", r"^OUT_\d{5}$")
    
    # Valid values
    valid_types = config["dq_checks"]["outlet_master"]["valid_outlet_types"]
    valid_sizes = config["dq_checks"]["outlet_master"]["valid_outlet_sizes"]
    
    df = check_valid_values(df, "Outlet_Type", valid_types)
    df = check_valid_values(df, "Outlet_Size", valid_sizes)
    
    # Note: If Outlet_Size is null, it isn't in mandatory_columns so it won't be quarantined by check_nulls.
    # However, check_valid_values handles notna(), so it ignores nulls.
    # We want to quarantine null sizes or keep them? Let's quarantine null sizes because 
    # the instructions said "Null Outlet_Size handling".
    df = check_nulls(df, ["Outlet_Size"])

    # ── 3. Master Data Decay Detection ─────────────────────────────
    # Load transactions for volume calculation
    trans_df = read_parquet(trans_path)
    df = compute_dynamic_tier(df, trans_df)

    # ── 4. Split & Quarantine ──────────────────────────────────────
    write_quarantine_outputs(df, "outlet_master", clean_path, config)
    logger.info("=" * 60)


if __name__ == "__main__":
    clean_outlet_master()
