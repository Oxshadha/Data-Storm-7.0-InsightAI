"""
Silver Layer — Holiday List Cleaning.

Handles:
  1. Parsing ISO dates
  2. Deduplication (same Date/Holiday_Name pairs)
  3. Format and null checks

Usage:
    python -m src.silver.clean_holidays
"""

import pandas as pd
from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.logger import get_logger
from src.silver.dq_checks import (
    add_rejection_column,
    check_nulls,
    check_duplicates
)
from src.silver.quarantine import write_quarantine_outputs

logger = get_logger("silver.clean_holidays")

def clean_holidays(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    bronze_path = config["paths"]["bronze"]["holiday_list"]
    clean_path = config["paths"]["silver"]["holiday_list"]
    
    logger.info("=" * 60)
    logger.info("Silver: Cleaning Holiday List")
    
    df = read_parquet(bronze_path)
    
    # ── 1. Data Parsing ────────────────────────────────────────
    # Standardize date format
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    # ── 2. DQ Checks (Quarantine Path) ─────────────────────────
    df = add_rejection_column(df)
    
    df = check_nulls(df, ["Date", "Holiday_Name", "Holiday_Type"])
    
    # Deduplicate on Date and Holiday_Name
    df = check_duplicates(df, ["Date", "Holiday_Name"])
    
    # ── 3. Split & Quarantine ──────────────────────────────────
    write_quarantine_outputs(df, "holiday_list", clean_path, config)
    logger.info("=" * 60)


if __name__ == "__main__":
    clean_holidays()
