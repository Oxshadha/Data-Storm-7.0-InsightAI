import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="paper")
out_dir = Path("output/plots")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading data for plot exports...")
trans_clean = pd.read_parquet('data/silver/transactions_clean.parquet')
abt = pd.read_parquet('data/gold/model_input.parquet')
outlet_coords = pd.read_parquet('data/silver/outlet_coordinates_clean.parquet')

# --- Plot 1: Ghost Exorcism ---
print("Generating Plot 1: Ghost Exorcism...")
volume_variance = trans_clean.groupby("Outlet_ID")["Volume_Liters"].std().fillna(0)
plt.figure(figsize=(8, 5))
sns.histplot(volume_variance[volume_variance < 500], bins=50, kde=True, color='teal')
plt.title("Variance of Cleaned Transactions (Removal of Extreme System Ghosts)", fontweight='bold')
plt.xlabel("Standard Deviation of Volume (Liters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(out_dir / "plot_1_ghost_exorcism.png", dpi=300)
plt.close()

# --- Plot 2: Spatial Catchment Hotspots ---
print("Generating Plot 2: Spatial Heatmap...")
plot_df = abt.merge(outlet_coords, on='Outlet_ID', how='inner')
# Western Province Focus
wp_df = plot_df[(plot_df['Latitude'] > 6.5) & (plot_df['Latitude'] < 7.3) & 
                (plot_df['Longitude'] > 79.8) & (plot_df['Longitude'] < 80.2)].copy()
gdf = gpd.GeoDataFrame(wp_df, geometry=gpd.points_from_xy(wp_df.Longitude, wp_df.Latitude), crs="EPSG:4326")

plt.figure(figsize=(10, 12))
ax = gdf.plot(column='poi_driver_catchment', cmap='YlOrRd', legend=True, 
              markersize=15, alpha=0.7, 
              legend_kwds={'label': "POI Driver Catchment Density", 'orientation': "horizontal"})
plt.title("Spatial Catchment Hotspots (Western Province Focus)", fontsize=14, fontweight='bold')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
ax.set_facecolor('#f8f9fa')
plt.grid(False)
plt.tight_layout()
plt.savefig(out_dir / "plot_2_spatial_hotspots.png", dpi=300)
plt.close()

# --- Model Training for Features ---
print("Training Model for Technical Proofs...")
features = [
    'poi_driver_catchment', 'poi_cannibal_risk', 'Tuition_Weekend_Surge', 
    'Tourist_Peak_Multiplier', 'Sports_Big_Match_Spike', 'Health_Catchment_Spike',
    'Has_High_Footfall_Catchment', 'Has_Cannibalization_Risk',
    'Number_of_Weekends', 'Holiday_Count'
]
cat_features = []

train_df = abt[abt["Is_Censored"] == 0].copy()
X_train = train_df[features].copy()
y_train = train_df["Total_Volume"]
for c in cat_features:
    X_train[c] = X_train[c].astype('category')

model = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=100, random_state=42, verbose=-1)
model.fit(X_train, y_train, categorical_feature=cat_features)

# --- Plot 3: Feature Importance (Engineering Proof) ---
print("Generating Plot 3: Feature Importance...")
fig, ax = plt.subplots(figsize=(10, 6))
lgb.plot_importance(model, importance_type='gain', max_num_features=12, 
                    title="LightGBM Information Gain (Engineering Proof)", 
                    ax=ax, color='teal', height=0.6)
plt.tight_layout()
plt.savefig(out_dir / "plot_3_feature_importance.png", dpi=300)
plt.close()

# --- Plot 4: Scatter Plot (Mathematical Proof) ---
print("Generating Plot 4: Prediction Performance...")
censored_df = abt[abt["Is_Censored"] == 1].copy()
if len(censored_df) > 0:
    X_censored = censored_df[features].copy()
    for c in cat_features:
        X_censored[c] = X_censored[c].astype('category')
    
    # Raw prediction for "ceiling breakthrough" visual
    raw_preds = model.predict(X_censored)
    # Applied floor for final reality check
    floored_preds = np.maximum(censored_df["Total_Volume"], raw_preds)
    
    plt.figure(figsize=(8, 6))
    sample_idx = np.random.choice(censored_df.index, min(3000, len(censored_df)), replace=False)
    
    # Actuals (ceiling)
    plt.scatter(censored_df.loc[sample_idx, "Total_Volume"], 
                censored_df.loc[sample_idx, "Total_Volume"], 
                alpha=0.4, color='#1f77b4', s=10, label='Actual Historical (Constrained)')
    
    # Predictions (breakthrough)
    plt.scatter(censored_df.loc[sample_idx, "Total_Volume"], 
                floored_preds.loc[sample_idx], 
                alpha=0.6, color='green', marker='*', s=15, label='Predicted Potential (Unconstrained)')
    
    max_val = min(censored_df["Total_Volume"].max(), 500)
    plt.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='Capacity Ceiling')
    
    plt.title("Unconstraining Right-Censored Demand (Mathematical Proof)", fontweight='bold')
    plt.xlabel("Actual Constrained Volume (Historical)")
    plt.ylabel("Predicted Unconstrained Volume (p90)")
    plt.legend()
    plt.xlim(0, max_val)
    plt.ylim(0, max_val * 1.5)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_4_scatter_unconstrained.png", dpi=300)
    plt.close()

# --- Plot 5: K-Means Elbow Method (Statistical Proof) ---
print("Generating Plot 5: Elbow Method validation...")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Aggregate to outlet level for static behavior clustering
# Use a subset of confirmed features
cluster_cols = ['poi_driver_catchment', 'poi_cannibal_risk', 'Tuition_Weekend_Surge']
outlet_static = abt.groupby("Outlet_ID")[cluster_cols].mean().fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(outlet_static)

sse = []
k_range = range(2, 16, 2)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, 'bo-', markerfacecolor='red', markersize=8)
plt.title("Elbow Method for Optimal Peer Clusters (k)", fontweight='bold')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(out_dir / "plot_5_elbow_method.png", dpi=300)
plt.close()

print("All synchronized plots saved successfully to output/plots/")
