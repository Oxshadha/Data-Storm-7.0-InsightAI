"""
Gold Layer — Spatial Intelligence Engine (V2: Distance-Decay + Competitive Density).

Replaces the flat-buffer POI counting from Round 1 with enterprise-grade
spatial features using cKDTree vectorized queries and continuous decay functions.

Architecture:
  1. Project all geometries to metric CRS (EPSG:5234 Sri Lanka Kandawala)
  2. Build scipy cKDTree spatial indices (O(N log N) construction)
  3. Compute per-category GRAVITY SCORES using exponential decay: W = e^(-λd)
  4. Compute COMPETITIVE SATURATION INDEX using Huff gravity: W = 1 / (d + 1)^β
  5. Derive composite LATENT OPPORTUNITY RATIO for optimizer consumption
  6. Preserve backward-compatible poi_count_* columns for ABT builder

Mathematical Justification:
  - Exponential decay λ=0.003: influence halves every ~230m, aligning with
    pedestrian walking catchment behavior in Sri Lankan retail corridors.
  - Huff gravity β=2: inverse-square law mirrors empirical retail competition
    research (Huff, 1964; Nakanishi & Cooper, 1974).

Usage:
    python -m src.gold.feature_poi
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.logger import get_logger

logger = get_logger("gold.feature_poi")

# ── Configuration Constants ────────────────────────────────────────────────
# Semantic grouping of POI categories for feature engineering

CATCHMENT_DRIVERS = {
    "youth":     ["school", "education"],
    "health":    ["hospital"],
    "leisure":   ["park", "beach"],
    "transit":   ["transport_hub", "railway_station"],
    "athletic":  ["stadium", "sports_centre", "pitch", "recreation_center", "gym"],
}

COMPETITORS = {
    "retail":    ["supermarket", "convenience_store"],
    "food":      ["restaurant", "cafe", "eatery"],
}


def _project_to_metric(df: pd.DataFrame, lon_col: str, lat_col: str) -> np.ndarray:
    """
    Project lat/lon coordinates to Sri Lanka Kandawala metric CRS (EPSG:5234).
    Returns Nx2 NumPy array of (x_meters, y_meters).
    """
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    gdf_metric = gdf.to_crs(epsg=5234)
    return np.column_stack([gdf_metric.geometry.x.values, gdf_metric.geometry.y.values])


def _compute_gravity_scores(
    shop_coords: np.ndarray,
    poi_coords: np.ndarray,
    max_radius_m: float,
    decay_lambda: float,
    max_k: int = 50,
) -> np.ndarray:
    """
    Compute exponential distance-decay gravity scores using cKDTree.

    For each shop, finds up to max_k POIs within max_radius_m meters,
    then applies: W = e^(-λ * distance)

    Parameters
    ----------
    shop_coords : (N, 2) array of shop positions in meters
    poi_coords : (M, 2) array of POI positions in meters
    max_radius_m : Maximum search radius in meters
    decay_lambda : Decay rate parameter (λ)
    max_k : Maximum number of nearest neighbors to retrieve

    Returns
    -------
    (N,) array of summed gravity scores per shop
    """
    if len(poi_coords) == 0:
        return np.zeros(len(shop_coords))

    # Clamp k to actual number of POIs available
    k = min(max_k, len(poi_coords))

    tree = cKDTree(poi_coords)
    distances, _ = tree.query(shop_coords, k=k, distance_upper_bound=max_radius_m)

    # Handle 1D case when k=1
    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)

    # Exponential decay: e^(-λd), with inf distances → 0 weight
    weights = np.where(np.isinf(distances), 0.0, np.exp(-decay_lambda * distances))

    return weights.sum(axis=1)


def _compute_competitive_saturation(
    shop_coords: np.ndarray,
    comp_coords: np.ndarray,
    max_radius_m: float,
    beta: float,
    max_k: int = 30,
) -> np.ndarray:
    """
    Compute Huff gravity-based competitive saturation index using cKDTree.

    For each shop, finds up to max_k competitors within max_radius_m meters,
    then applies: W = 1 / (d + 1)^β

    The +1 prevents division by zero for co-located competitors.

    Parameters
    ----------
    shop_coords : (N, 2) array of shop positions in meters
    comp_coords : (M, 2) array of competitor positions in meters
    max_radius_m : Maximum search radius for competitors
    beta : Gravity exponent (2 = inverse-square law)
    max_k : Maximum number of nearest competitors to retrieve

    Returns
    -------
    (N,) array of competitive saturation scores per shop
    """
    if len(comp_coords) == 0:
        return np.zeros(len(shop_coords))

    k = min(max_k, len(comp_coords))

    tree = cKDTree(comp_coords)
    distances, _ = tree.query(shop_coords, k=k, distance_upper_bound=max_radius_m)

    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)

    # Huff gravity: 1 / (d + 1)^β, with inf distances → 0
    weights = np.where(np.isinf(distances), 0.0, 1.0 / ((distances + 1.0) ** beta))

    return weights.sum(axis=1)


def _compute_flat_counts(
    shop_coords: np.ndarray,
    poi_coords: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """
    Compute flat POI counts within radius (backward-compatible with R1 poi_count_* columns).
    Uses cKDTree.query_ball_point for efficiency.
    """
    if len(poi_coords) == 0:
        return np.zeros(len(shop_coords), dtype=int)

    tree = cKDTree(poi_coords)
    # query_ball_point returns a list of lists of indices within radius
    results = tree.query_ball_point(shop_coords, r=radius_m)
    return np.array([len(r) for r in results], dtype=int)


def create_poi_features(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Gold: Spatial Intelligence Engine V2 (Distance-Decay + Competitive Density)")

    coords_path = config["paths"]["silver"]["outlet_coordinates"]
    poi_path = config["paths"]["silver"]["poi"]
    out_path = config["paths"]["gold"]["root"] + "/outlet_poi_features.parquet"

    # ── Load Data ──────────────────────────────────────────────────────────
    logger.info("Loading outlet coordinates and POIs...")
    coords_df = read_parquet(coords_path)
    poi_df = read_parquet(poi_path)

    if poi_df.empty:
        logger.warning("POI dataset is empty. Cannot generate spatial features.")
        return

    # ── Spatial Indexing Config ─────────────────────────────────────────────
    spatial_cfg = config.get("spatial_engine", {})
    catchment_radius = spatial_cfg.get("catchment_radius_m", 1500)
    competitor_radius = spatial_cfg.get("competitor_radius_m", 500)
    decay_lambda = spatial_cfg.get("decay_lambda", 0.003)
    huff_beta = spatial_cfg.get("huff_beta", 2.0)
    flat_count_radius = config.get("poi_scraping", {}).get("search_radius_m", 1000)

    logger.info(f"  Catchment radius: {catchment_radius}m | Decay λ={decay_lambda}")
    logger.info(f"  Competitor radius: {competitor_radius}m | Huff β={huff_beta}")
    logger.info(f"  Flat count radius: {flat_count_radius}m (backward compat)")

    # ── Step 1: Project to Metric CRS ──────────────────────────────────────
    logger.info("Projecting to EPSG:5234 (Sri Lanka Kandawala metric CRS)...")
    shop_coords = _project_to_metric(coords_df, "Longitude", "Latitude")
    poi_coords_all = _project_to_metric(poi_df, "longitude", "latitude")

    logger.info(f"  Shops: {len(shop_coords):,} | POIs: {len(poi_coords_all):,}")

    # ── Step 2: Per-Category Gravity Scores (Catchment Drivers) ────────────
    logger.info("Computing per-category exponential decay gravity scores...")

    result_df = coords_df[["Outlet_ID"]].copy()

    # All unique categories in our data
    all_categories = poi_df["poi_category"].unique()
    logger.info(f"  POI categories found: {list(all_categories)}")

    # 2a. Individual category gravity scores
    for cat in sorted(all_categories):
        cat_mask = poi_df["poi_category"] == cat
        cat_poi_coords = poi_coords_all[cat_mask.values]

        # Gravity score (distance-decay)
        gravity = _compute_gravity_scores(
            shop_coords, cat_poi_coords,
            max_radius_m=catchment_radius,
            decay_lambda=decay_lambda,
        )
        result_df[f"gravity_{cat}"] = np.round(gravity, 4)

        # Flat count (backward-compatible with R1 ABT)
        flat_count = _compute_flat_counts(shop_coords, cat_poi_coords, flat_count_radius)
        result_df[f"poi_count_{cat}"] = flat_count

        n_nonzero = (gravity > 0).sum()
        logger.info(f"    {cat}: {len(cat_poi_coords):,} POIs → {n_nonzero:,} shops with signal")

    # 2b. Grouped gravity scores (for spatio-temporal interactions)
    for group_name, categories in CATCHMENT_DRIVERS.items():
        present_cats = [c for c in categories if c in all_categories]
        gravity_cols = [f"gravity_{c}" for c in present_cats if f"gravity_{c}" in result_df.columns]
        if gravity_cols:
            result_df[f"gravity_group_{group_name}"] = result_df[gravity_cols].sum(axis=1)
        else:
            result_df[f"gravity_group_{group_name}"] = 0.0
        logger.info(f"    Group '{group_name}': {len(present_cats)} categories → gravity_group_{group_name}")

    # ── Step 3: Competitive Saturation Index ───────────────────────────────
    logger.info("Computing Huff gravity competitive saturation index...")

    # Identify all competitor POIs
    competitor_categories = [cat for cats in COMPETITORS.values() for cat in cats]
    comp_mask = poi_df["poi_category"].isin(competitor_categories)
    comp_poi_coords = poi_coords_all[comp_mask.values]

    logger.info(f"  Competitor POIs: {comp_mask.sum():,} across categories {competitor_categories}")

    # Overall competitive saturation
    comp_saturation = _compute_competitive_saturation(
        shop_coords, comp_poi_coords,
        max_radius_m=competitor_radius,
        beta=huff_beta,
    )
    result_df["competitive_saturation_index"] = np.round(comp_saturation, 4)

    # Per-group competitive saturation
    for group_name, categories in COMPETITORS.items():
        present_cats = [c for c in categories if c in all_categories]
        group_mask = poi_df["poi_category"].isin(present_cats)
        group_coords = poi_coords_all[group_mask.values]

        group_sat = _compute_competitive_saturation(
            shop_coords, group_coords,
            max_radius_m=competitor_radius,
            beta=huff_beta,
        )
        result_df[f"comp_saturation_{group_name}"] = np.round(group_sat, 4)

    # Flat competitor count (backward compat)
    comp_flat = _compute_flat_counts(shop_coords, comp_poi_coords, flat_count_radius)
    result_df["competitor_count_flat"] = comp_flat

    # ── Step 4: Composite Features ─────────────────────────────────────────
    logger.info("Engineering composite spatial features...")

    # Total catchment gravity (all driver categories)
    driver_gravity_cols = [f"gravity_group_{g}" for g in CATCHMENT_DRIVERS.keys()
                          if f"gravity_group_{g}" in result_df.columns]
    result_df["total_driver_gravity"] = result_df[driver_gravity_cols].sum(axis=1)

    # Backward-compatible aggregate columns
    count_cols = [c for c in result_df.columns if c.startswith("poi_count_")]
    result_df["poi_total_catchment"] = result_df[count_cols].sum(axis=1)

    # Backward-compatible poi_driver_catchment and poi_cannibal_risk
    driver_count_cats = [cat for cats in CATCHMENT_DRIVERS.values() for cat in cats]
    driver_count_cols = [f"poi_count_{c}" for c in driver_count_cats if f"poi_count_{c}" in result_df.columns]
    result_df["poi_driver_catchment"] = result_df[driver_count_cols].sum(axis=1) if driver_count_cols else 0

    cannibal_count_cols = [f"poi_count_{c}" for c in competitor_categories if f"poi_count_{c}" in result_df.columns]
    result_df["poi_cannibal_risk"] = result_df[cannibal_count_cols].sum(axis=1) if cannibal_count_cols else 0

    # THE GOLDEN FEATURE: Latent Opportunity Ratio
    # High driver gravity + Low competition = untapped goldmine
    result_df["latent_opportunity_ratio"] = np.round(
        result_df["total_driver_gravity"] / (result_df["competitive_saturation_index"] + 0.1),
        4
    )

    # Market isolation flag (high footfall gravity, zero or low flat competitors)
    # Uses competitor_count_flat (1km radius) for robustness since Huff saturation
    # is very sparse at 500m for Sri Lankan retail density
    median_gravity = result_df.loc[result_df["total_driver_gravity"] > 0, "total_driver_gravity"].median()
    result_df["is_isolated_goldmine"] = (
        (result_df["total_driver_gravity"] > median_gravity) &
        (result_df["competitor_count_flat"] == 0)
    ).astype(int)

    # ── Step 5: Summary Statistics ─────────────────────────────────────────
    n_with_signal = (result_df["total_driver_gravity"] > 0).sum()
    n_goldmines = result_df["is_isolated_goldmine"].sum()
    n_saturated = (result_df["competitive_saturation_index"] > result_df["competitive_saturation_index"].quantile(0.75)).sum()

    logger.info(f"  Outlets with catchment signal: {n_with_signal:,} / {len(result_df):,}")
    logger.info(f"  Isolated goldmines identified: {n_goldmines:,}")
    logger.info(f"  Highly saturated markets (Q4): {n_saturated:,}")
    logger.info(f"  Total features engineered: {len(result_df.columns) - 1}")

    # ── Save ───────────────────────────────────────────────────────────────
    write_parquet(result_df, out_path)
    logger.info(f"Saved spatial features to {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_poi_features()
