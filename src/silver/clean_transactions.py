"""
Silver Layer — Transaction forensics.
Handles System Ghosts: negative returns, zero volumes, duplicates, outliers.
"""

import pandas as pd

from src.utils.logger import get_logger
from src.utils.io import read_parquet, write_parquet
from src.silver.quarantine import QuarantineManager
from src.silver.dq_checks import (
    check_nulls,
    check_duplicates,
    check_negative_volumes,
    check_zero_volumes,
    check_value_range,
)

logger = get_logger("silver.clean_transactions")

def clean_transactions(config: dict) -> None:
    """Apply forensic cleaning to transactions data."""
    logger.info("Starting transaction cleaning.")
    
    bronze_path = config["paths"]["bronze"]["transactions"]
    df = read_parquet(bronze_path)
    
    qm = QuarantineManager(config)
    
    # 1. Run DQ checks
    mandatory_cols = config["dq_checks"]["transactions"]["mandatory_columns"]
    null_rejections = check_nulls(df, mandatory_cols, "transactions")
    qm.add_rejections(null_rejections)
    
    dup_keys = config["dq_checks"]["transactions"]["duplicate_key"]
    dup_rejections = check_duplicates(df, dup_keys, "transactions")
    qm.add_rejections(dup_rejections)
    
    v_min = config["dq_checks"]["transactions"]["volume_min"]
    v_max = config["dq_checks"]["transactions"]["volume_max"]
    range_rejections = check_value_range(df, "Volume_Liters", v_min, v_max, "transactions")
    qm.add_rejections(range_rejections)
    
    # 2. Tag negative and zero volumes
    neg_rejections = check_negative_volumes(df)
    qm.add_rejections(neg_rejections)
    
    zero_rejections = check_zero_volumes(df)
    qm.add_rejections(zero_rejections)
    
    # 3. Clean dataframe
    # Drop rows with nulls in mandatory columns
    df = df.dropna(subset=mandatory_cols)
    
    # Drop duplicates
    df = df.drop_duplicates(subset=dup_keys, keep="first")
    
    # Add flags
    df["Is_Return"] = (df["Volume_Liters"] < 0).astype(int)
    df["Is_Zero"] = (df["Volume_Liters"] == 0).astype(int)
    
    qm.flush()
    
    # Write to silver
    silver_path = config["paths"]["silver"]["transactions"]
    write_parquet(df, silver_path)
    logger.info("Finished transaction cleaning.")
