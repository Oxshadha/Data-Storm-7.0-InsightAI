import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pathlib import Path

# Load data
abt = pd.read_parquet('data/gold/model_input.parquet')
outlet_coords = pd.read_parquet('data/silver/outlet_coordinates.parquet')

# Merge to get lat/lon
plot_df = abt.merge(outlet_coords, on='Outlet_ID', how='inner')

# Filter to Western Province roughly (Colombo area)
# Lat: 6.5 to 7.3, Lon: 79.8 to 80.2
wp_df = plot_df[(plot_df['Latitude'] > 6.5) & (plot_df['Latitude'] < 7.3) & 
                (plot_df['Longitude'] > 79.8) & (plot_df['Longitude'] < 80.2)].copy()

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(wp_df, geometry=gpd.points_from_xy(wp_df.Longitude, wp_df.Latitude), crs="EPSG:4326")

plt.figure(figsize=(10, 12))
# Plot points colored by POI driver catchment
ax = gdf.plot(column='poi_driver_catchment', cmap='YlOrRd', legend=True, 
              markersize=15, alpha=0.7, 
              legend_kwds={'label': "POI Driver Catchment Density", 'orientation': "horizontal"})

plt.title("Spatial Catchment Hotspots (Western Province Focus)", fontsize=14, fontweight='bold')
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Remove grid for cleaner map
ax.set_facecolor('#f0f0f0')
plt.grid(False)

out_dir = Path("output/plots")
out_dir.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(out_dir / "plot_2_spatial_hotspots.png", dpi=300)
print("Saved output/plots/plot_2_spatial_hotspots.png")
