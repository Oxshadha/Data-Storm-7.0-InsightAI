"""
Bronze Layer — Ingest internal CSV files as-is into Parquet.
Zero transformations. Preserves original data exactly.
"""

from src.utils.config import load_config, resolve_path
from src.utils.io import read_csv_raw, write_parquet
from src.utils.logger import get_logger

logger = get_logger("bronze.ingest_internal")

# Mapping: config key → raw CSV filename
SOURCE_FILES = {
    "transactions": "transactions_history_final.csv",
    "outlet_master": "outlet_master.csv",
    "outlet_coordinates": "outlet_coordinates.csv",
    "distributor_seasonality": "distributor_seasonality_details.csv",
    "holiday_list": "holiday_list.csv",
}


def ingest_all(config: dict | None = None) -> dict:
    """Ingest all internal CSV files into Bronze parquet. Returns paths dict."""
    if config is None:
        config = load_config()

    raw_dir = config["paths"]["raw_source"]
    bronze_paths = config["paths"]["bronze"]
    results = {}

    for key, csv_name in SOURCE_FILES.items():
        logger.info(f"{'='*60}")
        logger.info(f"Ingesting: {csv_name} → bronze/{key}")

        csv_path = f"{raw_dir}/{csv_name}"
        df = read_csv_raw(csv_path)

        parquet_path = bronze_paths[key]
        write_parquet(df, parquet_path)
        results[key] = parquet_path

    logger.info(f"{'='*60}")
    logger.info(f"Bronze ingestion complete: {len(results)} datasets ingested.")
    return results


if __name__ == "__main__":
    ingest_all()
