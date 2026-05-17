"""
Silver Layer — POI Standardization (Overture Maps Data).

Handles:
  1. Deduplicating Overture Maps places.
  2. Null checks and formatting.

Usage:
    python -m src.silver.clean_poi
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
import os

logger = get_logger("silver.clean_poi")

def clean_poi(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    bronze_dir = config["paths"]["bronze"]["root"]
    bronze_path = f"{bronze_dir}/poi_raw.parquet"
    clean_path = config["paths"]["silver"]["poi"]
    
    logger.info("=" * 60)
    logger.info("Silver: Cleaning Overture Maps POIs")
    
    if not os.path.exists(bronze_path):
        logger.warning(f"Bronze POI file {bronze_path} not found. Skipping clean_poi.")
        return
        
    df = read_parquet(bronze_path)
    
    # ── 1. DQ Checks (Quarantine Path) ─────────────────────────
    df = add_rejection_column(df)
    
    df = check_nulls(df, ["osm_id", "poi_category", "longitude", "latitude"])
    
    # Deduplicate on osm_id
    df = check_duplicates(df, ["osm_id"])
    
    # ── 2. Split & Quarantine ──────────────────────────────────
    write_quarantine_outputs(df, "poi", clean_path, config)
    logger.info("=" * 60)


if __name__ == "__main__":
    clean_poi()
