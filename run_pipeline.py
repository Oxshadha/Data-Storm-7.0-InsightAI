"""
Master Pipeline Orchestrator — Bronze → Silver → Gold → Predict

Usage:
    python run_pipeline.py                # Run full pipeline
    python run_pipeline.py --stage bronze # Run only bronze stage
"""

import argparse
import sys

from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger, setup_logging

logger = get_logger("pipeline")


def run_bronze(config: dict) -> None:
    """Stage 1: Raw ingestion — CSV → Parquet, zero transforms."""
    logger.info("=" * 60)
    logger.info("STAGE 1: BRONZE — Raw Ingestion")
    logger.info("=" * 60)
    from src.bronze.ingest_internal import ingest_all
    ingest_all(config)
    
    from src.bronze.ingest_poi import scrape_pois
    logger.info("Running scrape_pois...")
    scrape_pois(config)


def run_silver(config: dict) -> None:
    """Stage 2: Forensic cleaning — DQ checks, quarantine."""
    logger.info("=" * 60)
    logger.info("STAGE 2: SILVER — Forensic Cleaning & DQ Checks")
    logger.info("=" * 60)
    
    from src.silver.clean_outlet_master import clean_outlet_master
    from src.silver.clean_transactions import clean_transactions
    from src.silver.clean_coordinates import clean_coordinates
    from src.silver.clean_seasonality import clean_seasonality
    from src.silver.clean_holidays import clean_holidays
    
    logger.info("Running clean_outlet_master...")
    clean_outlet_master(config)
    
    logger.info("Running clean_transactions...")
    clean_transactions(config)
    
    logger.info("Running clean_coordinates...")
    clean_coordinates(config)
    
    logger.info("Running clean_seasonality...")
    clean_seasonality(config)
    
    logger.info("Running clean_holidays...")
    clean_holidays(config)
    
    from src.silver.clean_poi import clean_poi
    logger.info("Running clean_poi...")
    clean_poi(config)


def run_gold(config: dict) -> None:
    """Stage 3: Feature engineering."""
    logger.info("=" * 60)
    logger.info("STAGE 3: GOLD — Feature Engineering")
    logger.info("=" * 60)
    
    from src.gold.build_model_input import build_model_input
    logger.info("Running build_model_input...")
    build_model_input(config)


def run_predict(config: dict) -> None:
    """Stage 4: Demand estimation & prediction."""
    logger.info("=" * 60)
    logger.info("STAGE 4: PREDICT — Demand Estimation")
    logger.info("=" * 60)
    from src.modeling.predict import generate_predictions
    logger.info("Running generate_predictions...")
    generate_predictions(config)


STAGES = {
    "bronze": run_bronze,
    "silver": run_silver,
    "gold": run_gold,
    "predict": run_predict,
}


def main():
    parser = argparse.ArgumentParser(description="InsightAI Pipeline Runner")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default=None,
        help="Run a specific stage. If omitted, runs all stages.",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    ensure_dirs(config)

    logger.info("🚀 InsightAI Pipeline — Data Storm 7.0")
    logger.info(f"Team: {config['project']['team_name']}")

    if args.stage:
        STAGES[args.stage](config)
    else:
        for name, fn in STAGES.items():
            fn(config)

    logger.info("✅ Pipeline complete.")


if __name__ == "__main__":
    main()
