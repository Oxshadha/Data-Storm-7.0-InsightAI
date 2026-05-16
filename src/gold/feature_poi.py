"""
Gold Layer — POI & Catchment Features.

Uses GeoPandas to perform a spatial join (sjoin) between outlets and 
Overture Maps POIs. Creates features representing the density of different
POI categories within the outlet's catchment area (1km radius).

Usage:
    python -m src.gold.feature_poi
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.logger import get_logger

logger = get_logger("gold.feature_poi")


def create_poi_features(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Gold: Engineering POI Catchment Features")

    coords_path = config["paths"]["silver"]["outlet_coordinates"]
    poi_path = config["paths"]["silver"]["poi"]
    out_path = config["paths"]["gold"]["root"] + "/outlet_poi_features.parquet"

    # 1. Load data
    logger.info("Loading outlet coordinates and POIs...")
    coords_df = read_parquet(coords_path)
    poi_df = read_parquet(poi_path)

    if poi_df.empty:
        logger.warning("POI dataset is empty. Cannot generate features.")
        return

    # 2. Convert to GeoDataFrames (EPSG:4326 = WGS84)
    logger.info("Converting to GeoDataFrames...")
    coords_gdf = gpd.GeoDataFrame(
        coords_df, 
        geometry=gpd.points_from_xy(coords_df.Longitude, coords_df.Latitude),
        crs="EPSG:4326"
    )
    
    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df.longitude, poi_df.latitude),
        crs="EPSG:4326"
    )

    # 3. Project to a metric CRS for buffering (EPSG:5234 is Sri Lanka Kandawala)
    # Using 5234 allows us to buffer accurately in meters.
    metric_crs = "EPSG:5234"
    coords_gdf = coords_gdf.to_crs(metric_crs)
    poi_gdf = poi_gdf.to_crs(metric_crs)

    # 4. Buffer outlets to define catchment area
    radius_m = config.get("poi_scraping", {}).get("search_radius_m", 1000)
    logger.info(f"Buffering outlets by {radius_m} meters...")
    
    # We buffer the geometry to create polygons representing the catchment area
    catchment_gdf = coords_gdf.copy()
    catchment_gdf["geometry"] = catchment_gdf.geometry.buffer(radius_m)

    # 5. Spatial Join: Which POIs fall inside which Outlet's catchment?
    logger.info("Performing spatial join (sjoin) to count POIs...")
    joined = gpd.sjoin(poi_gdf, catchment_gdf, how="inner", predicate="within")

    # 6. Aggregate counts per Outlet_ID and POI Category
    counts = (
        joined.groupby(["Outlet_ID", "poi_category"])
        .size()
        .reset_index(name="count")
    )

    # 7. Pivot to wide format (one row per Outlet_ID)
    poi_features = counts.pivot(
        index="Outlet_ID", 
        columns="poi_category", 
        values="count"
    ).fillna(0).astype(int).reset_index()

    # Prefix columns for clarity
    poi_cols = [c for c in poi_features.columns if c != "Outlet_ID"]
    poi_features.columns = ["Outlet_ID"] + [f"poi_count_{c}" for c in poi_cols]

    # Merge back to ensure all outlets are present (even those with 0 POIs)
    final_features = coords_df[["Outlet_ID"]].merge(poi_features, on="Outlet_ID", how="left").fillna(0)
    
    # Cast count columns to int
    count_cols = [c for c in final_features.columns if c.startswith("poi_count_")]
    final_features[count_cols] = final_features[count_cols].astype(int)

    # 8. Compute total POI density
    final_features["poi_total_catchment"] = final_features[count_cols].sum(axis=1)

    logger.info(f"Engineered {len(count_cols) + 1} spatial features for {len(final_features):,} outlets.")
    
    # 9. Write to Gold
    write_parquet(final_features, out_path)
    logger.info(f"Saved POI features to {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_poi_features()
