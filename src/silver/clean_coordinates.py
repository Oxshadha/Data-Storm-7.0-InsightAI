"""
Silver Layer — Outlet Coordinates Cleaning.

Handles:
  1. Geo-bounds validation for Sri Lanka (flag coordinates outside)
  2. Co-location detection (multiple outlets at same exact coordinates)

Usage:
    python -m src.silver.clean_coordinates
"""

import pandas as pd
from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.logger import get_logger
from src.silver.dq_checks import (
    add_rejection_column,
    check_nulls,
    check_format,
    check_value_range,
    check_duplicates
)
from src.silver.quarantine import write_quarantine_outputs

logger = get_logger("silver.clean_coordinates")

def check_colocated(df: pd.DataFrame, reason_col="Rejection_Reason") -> pd.DataFrame:
    """
    Tag outlets that share the exact same coordinates.
    These are not necessarily errors (could be a mall), but we tag them for feature engineering.
    We don't quarantine them, we just tag. Wait, actually we can quarantine if they are exactly 
    the same outlet ID, but that's handled by duplicate check.
    For co-location, we just add a flag column.
    """
    dupe_coords = df.duplicated(subset=["Latitude", "Longitude"], keep=False)
    df["Is_Colocated"] = dupe_coords
    count = dupe_coords.sum()
    if count:
        logger.info(f"  CO-LOCATION: {count:,} outlets share coordinates with another outlet.")
    return df

def clean_coordinates(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    bronze_path = config["paths"]["bronze"]["outlet_coordinates"]
    clean_path = config["paths"]["silver"]["outlet_coordinates"]
    
    logger.info("=" * 60)
    logger.info("Silver: Cleaning Outlet Coordinates")
    
    df = read_parquet(bronze_path)
    
    # ── 1. DQ Checks (Quarantine Path) ─────────────────────────
    df = add_rejection_column(df)
    
    df = check_nulls(df, ["Outlet_ID", "Latitude", "Longitude"])
    df = check_format(df, "Outlet_ID", r"^OUT_\d{5}$")
    
    # Exact duplicates on Outlet_ID
    df = check_duplicates(df, ["Outlet_ID"])
    
    # Geo-bounds for Sri Lanka
    lat_min = config["dq_checks"]["outlet_coordinates"]["latitude_min"]
    lat_max = config["dq_checks"]["outlet_coordinates"]["latitude_max"]
    lon_min = config["dq_checks"]["outlet_coordinates"]["longitude_min"]
    lon_max = config["dq_checks"]["outlet_coordinates"]["longitude_max"]
    
    df = check_value_range(df, "Latitude", lat_min, lat_max)
    df = check_value_range(df, "Longitude", lon_min, lon_max)
    
    # ── 2. Business Logic Tagging (Clean Path) ─────────────────
    df = check_colocated(df)
    
    # ── 3. Split & Quarantine ──────────────────────────────────
    write_quarantine_outputs(df, "outlet_coordinates", clean_path, config)
    logger.info("=" * 60)


if __name__ == "__main__":
    clean_coordinates()
