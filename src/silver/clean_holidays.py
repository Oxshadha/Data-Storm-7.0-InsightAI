"""Silver Layer — Holiday list cleaning."""

import pandas as pd
from src.utils.logger import get_logger
from src.utils.io import read_parquet, write_parquet
from src.silver.quarantine import QuarantineManager
from src.silver.dq_checks import check_nulls, check_duplicates

logger = get_logger("silver.clean_holidays")

def clean_holidays(config: dict) -> None:
    """Parse dates, handle duplicates, standardize holiday types."""
    logger.info("Starting holiday list cleaning.")
    bronze_path = config["paths"]["bronze"]["holiday_list"]
    df = read_parquet(bronze_path)
    
    qm = QuarantineManager(config)
    
    null_rejections = check_nulls(df, ["Date", "Holiday_Name", "Holiday_Type"], "holiday_list", id_column="Holiday_Name")
    qm.add_rejections(null_rejections)
    
    dup_rejections = check_duplicates(df, ["Date", "Holiday_Name", "Holiday_Type"], "holiday_list", id_column="Holiday_Name")
    qm.add_rejections(dup_rejections)
    
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    df = df.drop_duplicates(subset=["Date", "Holiday_Name", "Holiday_Type"], keep="first")
    df = df.dropna()
    
    qm.flush()
    
    silver_path = config["paths"]["silver"]["holiday_list"]
    write_parquet(df, silver_path)
    logger.info("Finished holiday list cleaning.")
