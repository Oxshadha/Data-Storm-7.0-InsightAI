"""
Bronze Layer — Ingest raw CSV files into Parquet with schema enforcement.

Reads CSVs from data/bronze/, converts to Parquet with:
  - Category dtypes for ID/categorical columns (memory optimization)
  - Schema locked down early to prevent type-mismatch crashes in Silver

Usage:
    python -m src.bronze.ingest_internal
    python run_pipeline.py --stage bronze
"""

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("bronze.ingest_internal")

# ── Source CSV → Parquet mapping + dtype casts ────────────────

SOURCE_FILES = {
    "transactions": {
        "csv": "transactions_history_final.csv",
        "parquet": "transactions_history.parquet",
        "category_cols": ["Outlet_ID", "Distributor_ID", "SKU_ID"],
        "dtype_overrides": {"Year": "int16", "Month": "int8"},
    },
    "outlet_master": {
        "csv": "outlet_master.csv",
        "parquet": "outlet_master.parquet",
        "category_cols": ["Outlet_ID", "Outlet_Size", "Outlet_Type"],
        "dtype_overrides": {"Cooler_Count": "int8"},
    },
    "outlet_coordinates": {
        "csv": "outlet_coordinates.csv",
        "parquet": "outlet_coordinates.parquet",
        "category_cols": ["Outlet_ID"],
        "dtype_overrides": {},
    },
    "distributor_seasonality": {
        "csv": "distributor_seasonality_details.csv",
        "parquet": "distributor_seasonality.parquet",
        "category_cols": ["Distributor_ID", "Seasonality_Index"],
        "dtype_overrides": {"Year": "int16", "Month": "int8"},
    },
    "holiday_list": {
        "csv": "holiday_list.csv",
        "parquet": "holiday_list.parquet",
        "category_cols": ["Holiday_Type"],
        "dtype_overrides": {},
    },
}


def ingest_all(config: dict | None = None) -> dict:
    """
    Convert all raw CSVs in data/bronze/ to Parquet with schema enforcement.

    - Casts ID and categorical columns to `category` dtype for memory savings
    - Enforces numeric dtypes to catch type mismatches early
    - Logs row counts and memory usage

    Returns dict of {dataset_name: parquet_path}.
    """
    if config is None:
        config = load_config()

    bronze_dir = resolve_path(config["paths"]["bronze"]["root"])
    results = {}

    for name, spec in SOURCE_FILES.items():
        csv_path = bronze_dir / spec["csv"]
        parquet_path = bronze_dir / spec["parquet"]

        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}. Skipping {name}.")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"Ingesting: {spec['csv']} → {spec['parquet']}")

        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"  Loaded: {len(df):,} rows × {len(df.columns)} cols")
        mem_before = df.memory_usage(deep=True).sum() / 1e6

        # Cast category columns
        for col in spec["category_cols"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Cast numeric overrides
        for col, dtype in spec["dtype_overrides"].items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        mem_after = df.memory_usage(deep=True).sum() / 1e6
        logger.info(
            f"  Memory: {mem_before:.1f} MB → {mem_after:.1f} MB "
            f"({(1 - mem_after/mem_before)*100:.0f}% reduction)"
        )

        # Write Parquet
        df.to_parquet(parquet_path, index=False)
        file_size = parquet_path.stat().st_size / 1e6
        logger.info(f"  Wrote: {parquet_path.name} ({file_size:.1f} MB on disk)")

        results[name] = str(parquet_path)

    logger.info(f"{'='*60}")
    logger.info(f"Bronze ingestion complete: {len(results)} datasets.")
    return results


if __name__ == "__main__":
    ingest_all()
