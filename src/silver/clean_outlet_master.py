"""
Silver Layer — Outlet master cleaning.
Handles typos, case inconsistencies, null values.
"""

from src.utils.logger import get_logger
from src.utils.io import read_parquet, write_parquet
from src.silver.quarantine import QuarantineManager
from src.silver.dq_checks import check_nulls, check_duplicates

logger = get_logger("silver.clean_outlet_master")


def clean_outlet_master(config: dict) -> None:
    """Apply cleaning to outlet_master data."""
    logger.info("Starting outlet master cleaning.")
    
    # 1. Load from Bronze
    bronze_path = config["paths"]["bronze"]["outlet_master"]
    df = read_parquet(bronze_path)
    
    qm = QuarantineManager(config)
    
    # 2. Run Pre-Cleaning DQ Checks
    mandatory_cols = config["dq_checks"]["outlet_master"]["mandatory_columns"]
    null_rejections = check_nulls(df, mandatory_cols, "outlet_master")
    qm.add_rejections(null_rejections)
    
    dup_rejections = check_duplicates(df, ["Outlet_ID"], "outlet_master")
    qm.add_rejections(dup_rejections)
    
    # 3. Clean Outlet_Type (trim and replace typos)
    if "Outlet_Type" in df.columns:
        df["Outlet_Type"] = df["Outlet_Type"].str.strip()
        typo_map = config["dq_checks"]["outlet_master"]["outlet_type_corrections"]
        df["Outlet_Type"] = df["Outlet_Type"].replace(typo_map)
        
    # 4. Clean Outlet_Size (trim and replace typos)
    if "Outlet_Size" in df.columns:
        df["Outlet_Size"] = df["Outlet_Size"].str.strip()
        size_map = config["dq_checks"]["outlet_master"]["outlet_size_corrections"]
        df["Outlet_Size"] = df["Outlet_Size"].replace(size_map)
        
        # 5. Handle null Outlet_Size records
        df["Outlet_Size"] = df["Outlet_Size"].fillna("Unknown")
        df.loc[df["Outlet_Size"] == "", "Outlet_Size"] = "Unknown"

    # Drop duplicates keeping first
    df = df.drop_duplicates(subset=["Outlet_ID"], keep="first")
    
    # Flush quarantine records
    qm.flush()
    
    # 6. Write clean data to Silver
    silver_path = config["paths"]["silver"]["outlet_master"]
    write_parquet(df, silver_path)
    logger.info("Finished outlet master cleaning.")
