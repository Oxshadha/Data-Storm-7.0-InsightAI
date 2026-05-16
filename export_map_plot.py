import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned POI data
poi_df = pd.read_parquet('data/silver/poi_clean.parquet')

# Set up the plot
plt.figure(figsize=(10, 12))
sns.scatterplot(
    data=poi_df,
    x='longitude',
    y='latitude',
    hue='poi_category',
    palette='viridis',
    s=10,
    alpha=0.6
)

# Set title and labels
plt.title('Spatial Distribution of Overture Maps POIs across Sri Lanka', fontsize=15, fontweight='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='POI Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and save
plt.tight_layout()
out_dir = Path("output/plots")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "plot_5_spatial_hotspots.png", dpi=300)
plt.close()

print("Spatial Hotspot map saved successfully to output/plots/plot_5_spatial_hotspots.png")
