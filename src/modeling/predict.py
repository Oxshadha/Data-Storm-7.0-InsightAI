"""Modeling — Generate final predictions CSV."""

import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.io import read_parquet
from src.modeling.demand_estimation import estimate_demand

logger = get_logger("modeling.predict")

def generate_predictions(config: dict) -> None:
    logger.info("Generating final predictions...")
    
    in_path = config["paths"]["gold"]["model_input"]
    df = read_parquet(in_path)
    
    predictions = estimate_demand(df, config)
    
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "insightai_predictions.csv"
    
    predictions.to_csv(out_path, index=False)
    logger.info(f"Predictions saved to {out_path} with shape {predictions.shape}.")
