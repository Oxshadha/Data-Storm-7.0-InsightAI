"""
Bronze Layer — POI (Point of Interest) scraping from OpenStreetMap.
Uses the Overpass API to fetch nearby POIs for each outlet.

TODO: Implement in Phase 2
"""

from src.utils.logger import get_logger

logger = get_logger("bronze.ingest_poi")


def scrape_pois(config: dict | None = None) -> None:
    """Scrape POIs from OpenStreetMap for all outlet coordinates."""
    # TODO: Implement Overpass API queries
    # 1. Load outlet_coordinates from Bronze
    # 2. For each outlet, query Overpass for POIs within radius
    # 3. Store raw POI results in data/bronze/poi_raw/
    logger.warning("POI scraping not yet implemented — placeholder.")
    raise NotImplementedError("POI scraping pipeline — implement in Phase 2")


if __name__ == "__main__":
    scrape_pois()
