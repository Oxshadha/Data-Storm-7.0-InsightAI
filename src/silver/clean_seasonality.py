"""
Silver Layer — Seasonality Cleaning.

Handles:
  1. Validates Seasonality_Index values
  2. Null checks
  3. Format checks

Usage:
    python -m src.silver.clean_seasonality
"""

from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.logger import get_logger
from src.silver.dq_checks import (
    add_rejection_column,
    check_nulls,
    check_format,
    check_value_range,
    check_valid_values,
    check_duplicates
)
from src.silver.quarantine import write_quarantine_outputs

logger = get_logger("silver.clean_seasonality")

def clean_seasonality(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    bronze_path = config["paths"]["bronze"]["distributor_seasonality"]
    clean_path = config["paths"]["silver"]["distributor_seasonality"]
    
    logger.info("=" * 60)
    logger.info("Silver: Cleaning Seasonality")
    
    df = read_parquet(bronze_path)
    
    # ── 1. DQ Checks (Quarantine Path) ─────────────────────────
    df = add_rejection_column(df)
    
    df = check_nulls(df, ["Distributor_ID", "Year", "Month", "Seasonality_Index"])
    df = check_format(df, "Distributor_ID", r"^DIST_(W|C|NW|S)_\d{2}$")
    
    # Check duplicates for (Distributor_ID, Year, Month)
    df = check_duplicates(df, ["Distributor_ID", "Year", "Month"])
    
    # Year/Month bounds
    df = check_value_range(df, "Year", 2023, 2026)
    df = check_value_range(df, "Month", 1, 12)
    
    # Valid values
    valid_vals = config["dq_checks"]["distributor_seasonality"]["valid_seasonality_values"]
    df = check_valid_values(df, "Seasonality_Index", valid_vals)
    
    # ── 2. Split & Quarantine ──────────────────────────────────
    write_quarantine_outputs(df, "distributor_seasonality", clean_path, config)
    logger.info("=" * 60)


if __name__ == "__main__":
    clean_seasonality()
