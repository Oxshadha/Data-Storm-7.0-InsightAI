"""Silver Layer — Seasonality data validation."""

import pandas as pd
from src.utils.logger import get_logger
from src.utils.io import read_parquet, write_parquet
from src.silver.quarantine import QuarantineManager
from src.silver.dq_checks import check_nulls, check_duplicates

logger = get_logger("silver.clean_seasonality")

def clean_seasonality(config: dict) -> None:
    """Validate seasonality index values and completeness."""
    logger.info("Starting seasonality cleaning.")
    bronze_path = config["paths"]["bronze"]["distributor_seasonality"]
    df = read_parquet(bronze_path)
    
    qm = QuarantineManager(config)
    
    null_rejections = check_nulls(df, ["Distributor_ID", "Year", "Month", "Seasonality_Index"], "distributor_seasonality", id_column="Distributor_ID")
    qm.add_rejections(null_rejections)
    
    dup_rejections = check_duplicates(df, ["Distributor_ID", "Year", "Month"], "distributor_seasonality", id_column="Distributor_ID")
    qm.add_rejections(dup_rejections)
    
    # Strip whitespace
    df["Seasonality_Index"] = df["Seasonality_Index"].astype(str).str.strip()
    
    valid_vals = config["dq_checks"]["distributor_seasonality"]["valid_seasonality_values"]
    invalid_mask = ~df["Seasonality_Index"].isin(valid_vals)
    if invalid_mask.any():
        logger.warning(f"Found {invalid_mask.sum()} invalid Seasonality_Index records.")
        
    df = df.drop_duplicates(subset=["Distributor_ID", "Year", "Month"], keep="first")
    df = df.dropna()
    
    qm.flush()
    
    silver_path = config["paths"]["silver"]["distributor_seasonality"]
    write_parquet(df, silver_path)
    logger.info("Finished seasonality cleaning.")
