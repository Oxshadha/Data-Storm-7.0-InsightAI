"""Gold — Spatial and Geospatial Features (POIs & Gravity)."""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger("gold.feature_spatial")

def build_poi_features(config: dict) -> pd.DataFrame:
    """Build POI count, nearest distance, gravity, and cluster features for each outlet."""
    logger.info("Building Spatial/POI features...")
    
    outlets = read_parquet(config["paths"]["silver"]["outlet_coordinates"])
    pois = read_parquet(config["paths"]["silver"]["poi"])
    
    # 1 degree lat/lon is approx 111 km in Sri Lanka
    DEG_TO_KM = 111.0
    radius_deg = config["poi_scraping"]["search_radius_m"] / (DEG_TO_KM * 1000.0)
    
    outlet_coords = outlets[["Latitude", "Longitude"]].values
    
    if len(pois) == 0:
        logger.warning("No POIs found. Returning default Spatial features.")
        outlets["poi_gravity_score"] = 0.0
        outlets["urban_density_score"] = 0
        outlets["spatial_cluster"] = 0
        return outlets.drop(columns=["Latitude", "Longitude"])

    poi_coords = pois[["lat", "lon"]].values
    tree = cKDTree(poi_coords)
    
    # 1. Total POI Count (Urban Density Proxy)
    outlets["urban_density_score"] = [len(x) for x in tree.query_ball_point(outlet_coords, radius_deg)]
    
    # 2. General Gravity Score
    # sum(weight / (dist_km^2 + epsilon))
    k_neighbors = min(20, len(poi_coords))
    dists, _ = tree.query(outlet_coords, k=k_neighbors)
    dists_km = dists * DEG_TO_KM
    
    # Calculate gravity using distance decay, capping extremely close POIs with +0.01 to avoid div zero
    gravity_scores = np.sum(1.0 / (dists_km**2 + 0.01), axis=1)
    outlets["poi_gravity_score"] = gravity_scores
    
    # 3. Category Specific Features (Counts & Nearest Distance)
    for cat in pois["category"].unique():
        cat_pois = pois[pois["category"] == cat][["lat", "lon"]].values
        if len(cat_pois) == 0:
            outlets[f"poi_{cat}_count_1km"] = 0
            outlets[f"nearest_{cat}_km"] = 50.0  # default far distance
            continue
            
        cat_tree = cKDTree(cat_pois)
        
        # Counts
        outlets[f"poi_{cat}_count_1km"] = [len(x) for x in cat_tree.query_ball_point(outlet_coords, radius_deg)]
        
        # Nearest distance
        dist_cat, _ = cat_tree.query(outlet_coords, k=1)
        outlets[f"nearest_{cat}_km"] = dist_cat * DEG_TO_KM

    # 4. Outlet Spatial Clustering (Macro-Regions)
    # 20 clusters roughly groups neighborhoods / towns together
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    outlets["spatial_cluster"] = kmeans.fit_predict(outlet_coords)

    logger.info(f"Finished building Spatial features for {len(outlets):,} outlets.")
    return outlets.drop(columns=["Latitude", "Longitude"])
