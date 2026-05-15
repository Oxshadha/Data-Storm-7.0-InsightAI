"""Gold — POI density & catchment features."""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger("gold.feature_poi")

def build_poi_features(config: dict) -> pd.DataFrame:
    """Build POI count and distance features for each outlet."""
    logger.info("Building POI features...")
    
    outlets = read_parquet(config["paths"]["silver"]["outlet_coordinates"])
    pois = read_parquet(config["paths"]["silver"]["poi"])
    
    # Radius in degrees (1km ~ 0.009 deg)
    radius_deg = config["poi_scraping"]["search_radius_m"] / 111000.0
    
    outlet_coords = outlets[["Latitude", "Longitude"]].values
    poi_coords = pois[["lat", "lon"]].values
    
    if len(poi_coords) == 0:
        logger.warning("No POIs found. Returning empty POI features.")
        return pd.DataFrame({"Outlet_ID": outlets["Outlet_ID"]})
        
    tree = cKDTree(poi_coords)
    
    # Total POI Count
    outlets["poi_count_1km"] = [len(x) for x in tree.query_ball_point(outlet_coords, radius_deg)]
    
    # Nearest distance
    dist, _ = tree.query(outlet_coords, k=1)
    outlets["nearest_poi_km"] = dist * 111.0  # roughly back to km
    
    # By category
    for cat in pois["category"].unique():
        cat_pois = pois[pois["category"] == cat][["lat", "lon"]].values
        if len(cat_pois) == 0:
            outlets[f"poi_{cat}_count_1km"] = 0
            continue
            
        cat_tree = cKDTree(cat_pois)
        outlets[f"poi_{cat}_count_1km"] = [len(x) for x in cat_tree.query_ball_point(outlet_coords, radius_deg)]
        
    logger.info("Finished building POI features.")
    return outlets.drop(columns=["Latitude", "Longitude"])
