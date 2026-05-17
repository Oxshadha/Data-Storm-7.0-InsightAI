"""
Silver Layer — Transaction Forensics & Cleaning.

Implements the forensic approach to the transactions dataset:
  1. Tags System Ghosts (zeros, duplicate retries) for quarantine.
  2. Tags Negative Volumes as returns (kept in clean data for aggregation).
  3. Tags Lazy Reps for Gold layer handling (kept in clean data).
  4. Applies standard DQ checks (nulls, formats, ref integrity).
  5. Aggregates the final clean dataset to net out returns and compress.

Usage:
    python -m src.silver.clean_transactions
"""

import pandas as pd
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.logger import get_logger
from src.silver.dq_checks import (
    add_rejection_column,
    check_nulls,
    check_referential_integrity,
    check_format,
    check_value_range,
    check_zero_volumes,
    check_duplicate_retries,
    tag_negative_volumes,
    check_lazy_rep,
)
from src.silver.quarantine import write_quarantine_outputs

logger = get_logger("silver.clean_transactions")


def clean_transactions(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    # ── Load Bronze Data ───────────────────────────────────────
    trans_path = config["paths"]["bronze"]["transactions"]
    master_path = config["paths"]["bronze"]["outlet_master"]
    
    logger.info("=" * 60)
    logger.info("Silver: Cleaning Transactions")
    
    df = read_parquet(trans_path)
    ref_df = read_parquet(master_path)
    
    # ── 1. DQ Checks (Quarantine Path) ─────────────────────────
    df = add_rejection_column(df)
    
    # Check nulls in mandatory columns
    mandatory = config["dq_checks"]["transactions"]["mandatory_columns"]
    df = check_nulls(df, mandatory)
    
    # Format checks for IDs
    df = check_format(df, "Outlet_ID", r"^OUT_\d{5}$")
    df = check_format(df, "Distributor_ID", r"^DIST_(W|C|NW|S)_\d{2}$")
    df = check_format(df, "SKU_ID", r"^SKU_\d{2}$")
    
    # Value range check (allow negatives for returns, catch absurd anomalies)
    v_min = config["dq_checks"]["transactions"]["volume_min"]
    v_max = config["dq_checks"]["transactions"]["volume_max"]
    df = check_value_range(df, "Volume_Liters", v_min, v_max)
    
    # Referential integrity (Outlet must exist in master)
    df = check_referential_integrity(
        df=df,
        ref_df=ref_df,
        fk_column="Outlet_ID",
        pk_column="Outlet_ID",
        ref_name="outlet_master"
    )
    
    # The System Ghosts: Zero Volumes (Quarantine)
    df = check_zero_volumes(df, "Volume_Liters")
    
    # The System Ghosts: Duplicate Retries (Quarantine)
    # Identical transactions (same outlet, date, sku, volume, bill) = retry
    dupe_keys = config["dq_checks"]["transactions"]["duplicate_key"]
    df = check_duplicate_retries(df, dupe_keys)

    # ── 2. Business Logic Tagging (Clean Path) ─────────────────
    
    # The System Ghosts: Negative Returns (Tag, DO NOT Quarantine)
    df = tag_negative_volumes(df, "Volume_Liters", "Is_Return")
    
    # The System Ghosts: Lazy Rep Trap (Tag, DO NOT Quarantine)
    df = check_lazy_rep(
        df,
        outlet_col="Outlet_ID",
        sku_col="SKU_ID",
        volume_col="Volume_Liters",
        max_sku_count=2,
        flag_col="Lazy_Rep_Flag"
    )
    
    # ── 3. Split & Quarantine ──────────────────────────────────
    clean_path = config["paths"]["silver"]["transactions"]
    
    # Split the dataframe using our quarantine manager
    # write_quarantine_outputs will save the rejected rows to the rejected folder
    # and return the counts. But we still need the clean_df to aggregate it.
    
    from src.silver.dq_checks import split_by_rejection
    clean_df, quarantined_df = split_by_rejection(df)
    
    # Write quarantine outputs (we use the full df for write_quarantine_outputs)
    write_quarantine_outputs(df, "transactions", clean_path, config)
    
    # ── 4. Aggregate Clean Transactions ────────────────────────
    # Aggregate clean data to net out returns and reduce granularity
    # Group by Outlet, Year, Month, Distributor, SKU
    logger.info("Aggregating clean transactions...")
    
    # Lazy_Rep_Flag is at the outlet level, so it's safe to take the max/first
    agg_funcs = {
        "Volume_Liters": "sum",
        "Total_Bill_Value": "sum",
        "Is_Return": "max",
        "Lazy_Rep_Flag": "max"
    }
    
    group_cols = ["Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID"]
    
    clean_agg = (
        clean_df.groupby(group_cols, observed=True)
        .agg(agg_funcs)
        .reset_index()
    )
    
    logger.info(
        f"Aggregated clean transactions from {len(clean_df):,} "
        f"to {len(clean_agg):,} rows."
    )
    
    # Write aggregated clean data to silver
    write_parquet(clean_agg, clean_path)
    logger.info("=" * 60)

if __name__ == "__main__":
    clean_transactions()
