"""Silver Layer — Coordinate geo-validation."""

from src.utils.logger import get_logger
from src.utils.io import read_parquet, write_parquet
from src.silver.quarantine import QuarantineManager
from src.silver.dq_checks import check_nulls, check_duplicates, check_value_range

logger = get_logger("silver.clean_coordinates")

def clean_coordinates(config: dict) -> None:
    """Validate coordinates within Sri Lanka bounds."""
    logger.info("Starting coordinate cleaning.")
    bronze_path = config["paths"]["bronze"]["outlet_coordinates"]
    df = read_parquet(bronze_path)
    
    qm = QuarantineManager(config)
    
    null_rejections = check_nulls(df, ["Outlet_ID", "Latitude", "Longitude"], "outlet_coordinates")
    qm.add_rejections(null_rejections)
    
    dup_rejections = check_duplicates(df, ["Outlet_ID"], "outlet_coordinates")
    qm.add_rejections(dup_rejections)
    
    lat_min = config["dq_checks"]["outlet_coordinates"]["latitude_min"]
    lat_max = config["dq_checks"]["outlet_coordinates"]["latitude_max"]
    lat_rej = check_value_range(df, "Latitude", lat_min, lat_max, "outlet_coordinates")
    qm.add_rejections(lat_rej)
    
    lon_min = config["dq_checks"]["outlet_coordinates"]["longitude_min"]
    lon_max = config["dq_checks"]["outlet_coordinates"]["longitude_max"]
    lon_rej = check_value_range(df, "Longitude", lon_min, lon_max, "outlet_coordinates")
    qm.add_rejections(lon_rej)
    
    df = df.dropna(subset=["Outlet_ID", "Latitude", "Longitude"])
    df = df.drop_duplicates(subset=["Outlet_ID"], keep="first")
    
    qm.flush()
    
    silver_path = config["paths"]["silver"]["outlet_coordinates"]
    write_parquet(df, silver_path)
    logger.info("Finished coordinate cleaning.")
