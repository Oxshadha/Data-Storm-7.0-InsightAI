"""
Bronze Layer — Overture Maps POI Ingestion using DuckDB.

Pulls POI data directly from Overture Maps S3 bucket using DuckDB.
This avoids the 7-hour Overpass API scraping process by using
cloud-native GeoParquet with bounding box pushdown.

Usage:
    python -m src.bronze.ingest_poi
"""

import duckdb
from pathlib import Path
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("bronze.ingest_poi")

def scrape_pois(config: dict | None = None) -> Path:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Bronze: Ingesting POIs from Overture Maps (DuckDB)")

    # Bounding box for Sri Lanka
    lat_min = config["dq_checks"]["outlet_coordinates"]["latitude_min"]
    lat_max = config["dq_checks"]["outlet_coordinates"]["latitude_max"]
    lon_min = config["dq_checks"]["outlet_coordinates"]["longitude_min"]
    lon_max = config["dq_checks"]["outlet_coordinates"]["longitude_max"]

    output_path = resolve_path(config["paths"]["bronze"]["root"]) / "poi_raw.parquet"
    
    # Initialize DuckDB
    con = duckdb.connect()
    
    # Install and load necessary extensions
    logger.info("Installing/Loading DuckDB httpfs & spatial extensions...")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region = 'us-west-2';")

    # Use a recent Overture release
    # We query specific categories relevant to the competition
    overture_url = "s3://overturemaps-us-west-2/release/2026-04-15.0/theme=places/type=place/*"

    query = f"""
    COPY (
        SELECT 
            id as osm_id,
            names.primary AS poi_name,
            categories.primary AS poi_category,
            ST_X(geometry) as longitude,
            ST_Y(geometry) as latitude
        FROM read_parquet('{overture_url}', filename=true, hive_partitioning=1)
        WHERE bbox.xmin BETWEEN {lon_min} AND {lon_max}
          AND bbox.ymin BETWEEN {lat_min} AND {lat_max}
          AND categories.primary IN (
              -- Education / Youth
              'school', 'college_university', 'education', 'preschool',
              'educational_services',
              -- Health
              'hospital',
              -- Transit
              'bus_station', 'train_station', 'transportation', 'gas_station',
              -- Religious (major Sri Lankan footfall drivers)
              'buddhist_temple', 'hindu_temple', 'mosque',
              'church_cathedral', 'catholic_church',
              -- Competitors (beverage sellers)
              'restaurant', 'cafe', 'convenience_store', 'supermarket',
              'grocery_store', 'hotel', 'bakery', 'accommodation', 'resort', 'bar',
              -- Leisure / Tourist
              'park', 'beach', 'playground', 'national_park',
              'landmark_and_historical_building',
              -- Athletic / Sports
              'gym', 'stadium_arena', 'sports_club_and_league',
              'cricket_ground', 'sports_and_recreation_venue',
              -- Niche
              'liquor_store'
          )
    ) TO '{str(output_path)}' (FORMAT PARQUET);
    """

    logger.info(f"Querying Overture Maps in S3 for bounding box: [{lon_min}, {lat_min}] to [{lon_max}, {lat_max}]")
    
    try:
        con.execute(query)
        logger.info(f"Successfully downloaded Overture Maps POIs to: {output_path.name}")
        
        # Verify count
        count = con.execute(f"SELECT count(*) FROM '{str(output_path)}'").fetchone()[0]
        logger.info(f"Total POIs extracted: {count:,}")
        
    except Exception as e:
        logger.error(f"Failed to query Overture Maps: {e}")
        raise

    logger.info("=" * 60)
    return output_path

if __name__ == "__main__":
    scrape_pois()
