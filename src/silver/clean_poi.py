"""Silver Layer — POI data standardization."""

from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger
from src.utils.io import read_parquet, write_parquet
from src.silver.quarantine import QuarantineManager
from src.silver.dq_checks import check_nulls, check_duplicates, check_value_range

logger = get_logger("silver.clean_poi")

def clean_poi(config: dict) -> None:
    """Standardize and validate scraped POI data."""
    logger.info("Starting POI cleaning.")
    
    raw_path = Path(config["paths"]["bronze"]["poi_raw"]) / "all_pois_raw.parquet"
    if not raw_path.exists():
        logger.warning(f"Raw POI file not found at {raw_path}. Skip cleaning.")
        return
        
    df = read_parquet(str(raw_path))
    qm = QuarantineManager(config)
    
    null_rej = check_nulls(df, ["osm_id", "category", "lat", "lon"], "poi", id_column="osm_id")
    qm.add_rejections(null_rej)
    
    dup_rej = check_duplicates(df, ["osm_id"], "poi", id_column="osm_id")
    qm.add_rejections(dup_rej)
    
    lat_min = config["dq_checks"]["outlet_coordinates"]["latitude_min"]
    lat_max = config["dq_checks"]["outlet_coordinates"]["latitude_max"]
    lat_rej = check_value_range(df, "lat", lat_min, lat_max, "poi")
    qm.add_rejections(lat_rej)
    
    lon_min = config["dq_checks"]["outlet_coordinates"]["longitude_min"]
    lon_max = config["dq_checks"]["outlet_coordinates"]["longitude_max"]
    lon_rej = check_value_range(df, "lon", lon_min, lon_max, "poi")
    qm.add_rejections(lon_rej)
    
    df = df.dropna(subset=["osm_id", "category", "lat", "lon"])
    df = df.drop_duplicates(subset=["osm_id"], keep="first")
    
    qm.flush()
    
    silver_path = config["paths"]["silver"]["poi"]
    write_parquet(df, silver_path)
    logger.info("Finished POI cleaning.")
