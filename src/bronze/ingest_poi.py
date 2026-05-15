"""
Bronze Layer — POI (Point of Interest) scraping from OpenStreetMap.
Uses the Overpass API to fetch POIs.
"""

import time
import requests
import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.io import write_parquet

logger = get_logger("bronze.ingest_poi")


def scrape_pois(config: dict) -> None:
    """Scrape POIs from OpenStreetMap using a single bounding box per category."""
    
    # 1. Construct bounding box for Sri Lanka
    lat_min = config["dq_checks"]["outlet_coordinates"]["latitude_min"]
    lat_max = config["dq_checks"]["outlet_coordinates"]["latitude_max"]
    lon_min = config["dq_checks"]["outlet_coordinates"]["longitude_min"]
    lon_max = config["dq_checks"]["outlet_coordinates"]["longitude_max"]
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    
    url = config["poi_scraping"]["overpass_url"]
    categories = config["poi_scraping"]["categories"]
    
    poi_data = []
    
    logger.info("Starting POI scraping for Sri Lanka bounding box. (Bulk approach to save OSM API load)")
    
    for category_name, tags in categories.items():
        for tag_k, tag_v in tags.items():
            query = f"""
            [out:json][timeout:180];
            (
              node["{tag_k}"="{tag_v}"]({bbox});
              way["{tag_k}"="{tag_v}"]({bbox});
              relation["{tag_k}"="{tag_v}"]({bbox});
            );
            out center;
            """
            
            logger.info(f"Querying {category_name} ({tag_k}={tag_v})...")
            
            try:
                headers = {"User-Agent": "InsightAI/1.0 (DataStorm7)"}
                response = requests.post(url, data={'data': query}, headers=headers, timeout=200)
                response.raise_for_status()
                data = response.json()
                
                elements = data.get("elements", [])
                logger.info(f"  -> Found {len(elements):,} {category_name}.")
                
                for el in elements:
                    lat = el.get("lat") or (el.get("center", {}).get("lat"))
                    lon = el.get("lon") or (el.get("center", {}).get("lon"))
                    if lat and lon:
                        poi_data.append({
                            "osm_id": str(el["id"]),
                            "category": category_name,
                            "lat": lat,
                            "lon": lon,
                            "name": el.get("tags", {}).get("name", "Unknown")
                        })
                
                # Respect rate limit
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to fetch {category_name}: {e}")
                
    df_pois = pd.DataFrame(poi_data)
    
    out_dir = Path(config["paths"]["bronze"]["poi_raw"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_pois_raw.parquet"
    
    write_parquet(df_pois, str(out_path))
    logger.info(f"Saved {len(df_pois):,} raw POIs to {out_path}.")

