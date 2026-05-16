import nbformat as nbf
import os

os.makedirs("notebooks", exist_ok=True)

# ==============================================================================
# Notebook 1: Data Forensics and Cleaning
# ==============================================================================
nb1 = nbf.v4.new_notebook()
nb1.cells.append(nbf.v4.new_markdown_cell("""# Notebook 1: Data Forensics and Cleaning
The Goal: Prove that you understood the raw datasets, properly defined numerical vs. categorical boundaries, and successfully neutralized the "System Ghosts" without destroying the underlying variance.
"""))
nb1.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Load raw and clean master data for categorical shift
raw_master = pd.read_parquet('../data/bronze/outlet_master.parquet')
clean_master = pd.read_parquet('../data/silver/outlet_master_clean.parquet')

# Load raw and clean transactions
raw_trans = pd.read_parquet('../data/bronze/transactions_history.parquet')
clean_trans = pd.read_parquet('../data/silver/transactions_clean.parquet')
"""))
nb1.cells.append(nbf.v4.new_markdown_cell("### Plot 1: The Missingness Matrix (Pre-Cleaning)"))
nb1.cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))
sns.heatmap(raw_master.isnull(), cbar=False, cmap='viridis')
plt.title("Missingness Matrix (Raw Outlet Master)", fontsize=14)
plt.show()
"""))
nb1.cells.append(nbf.v4.new_markdown_cell("### Plot 2: Categorical Distribution Shift (Master Data Decay)"))
nb1.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(y='Outlet_Size', data=raw_master, ax=axes[0], order=raw_master['Outlet_Size'].value_counts().index, palette='Blues_d')
axes[0].set_title("Raw Static SFA Labels (Outlet_Size)")

sns.countplot(y='Dynamic_Tier', data=clean_master, ax=axes[1], order=clean_master['Dynamic_Tier'].value_counts().index, palette='Greens_d')
axes[1].set_title("Behavioral Clustering (Dynamic_Tier)")
plt.tight_layout()
plt.show()
"""))
nb1.cells.append(nbf.v4.new_markdown_cell("### Plot 3: The Ghost Exorcism (Numerical Distribution)"))
nb1.cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 6))
sns.kdeplot(np.log1p(raw_trans[raw_trans['Volume_Liters']>0]['Volume_Liters'].sample(50000, random_state=42)), label='Raw Data (Skewed)', fill=True, color='red', alpha=0.3)
sns.kdeplot(np.log1p(clean_trans[clean_trans['Volume_Liters']>0]['Volume_Liters'].sample(50000, random_state=42)), label='Silver Data (Net)', fill=True, color='blue', alpha=0.5)
plt.title("Log-Normal Distribution of Volume (Raw vs Silver)", fontsize=14)
plt.xlabel("Log(Volume_Liters + 1)")
plt.legend()
plt.show()
"""))
with open('notebooks/01_Data_Forensics_and_Cleaning.ipynb', 'w') as f:
    nbf.write(nb1, f)

# ==============================================================================
# Notebook 2: SpatioTemporal EDA
# ==============================================================================
nb2 = nbf.v4.new_notebook()
nb2.cells.append(nbf.v4.new_markdown_cell("""# Notebook 2: SpatioTemporal EDA
The Goal: Visualize the Gold Layer feature engineering. Show the interaction between physical space and cultural timelines.
"""))
nb2.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

sns.set_theme(style="whitegrid")
abt = pd.read_parquet('../data/gold/model_input.parquet')
coords = pd.read_parquet('../data/silver/outlet_coordinates_clean.parquet')
"""))
nb2.cells.append(nbf.v4.new_markdown_cell("### Plot 1: The Catchment Heatmap"))
nb2.cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 8))
sample_coords = coords.merge(abt[['Outlet_ID', 'poi_total_catchment']].drop_duplicates(), on='Outlet_ID')

scatter = plt.scatter(sample_coords['Longitude'], sample_coords['Latitude'], 
            c=sample_coords['poi_total_catchment'], cmap='hot', alpha=0.7, s=20)
plt.colorbar(scatter, label='POI Density')
plt.title("Overture POI Catchment Heatmap (Sri Lanka)", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
"""))
nb2.cells.append(nbf.v4.new_markdown_cell("### Plot 2: The Timeline Interaction (The 'Big Match' Spike)"))
nb2.cells.append(nbf.v4.new_code_cell("""time_series = abt.groupby(['Year', 'Month'])['Total_Volume'].mean().reset_index()
time_series['Date'] = pd.to_datetime(time_series[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(12, 6))
plt.plot(time_series['Date'], time_series['Total_Volume'], marker='o', linestyle='-', linewidth=2, color='#2ca02c')

# Highlight March/April (Big Match Season)
for year in [2023, 2024, 2025]:
    plt.axvspan(pd.Timestamp(f"{year}-03-01"), pd.Timestamp(f"{year}-04-30"), color='orange', alpha=0.3, label='Big Match / New Year Surge' if year==2023 else "")

plt.title("Average Network Volume Over Time", fontsize=14)
plt.ylabel("Average Volume (Liters)")
plt.legend()
plt.show()
"""))
with open('notebooks/02_SpatioTemporal_EDA.ipynb', 'w') as f:
    nbf.write(nb2, f)

# ==============================================================================
# Notebook 3: Cluster Optimization and Base Math
# ==============================================================================
nb3 = nbf.v4.new_notebook()
nb3.cells.append(nbf.v4.new_markdown_cell("""# Notebook 3: Cluster Optimization and Base Math
The Goal: Address finding the right number of clusters (K) and validating the high-dimensional feature space using linear algebra before baseline projection.
"""))
nb3.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")
abt = pd.read_parquet('../data/gold/model_input.parquet').sample(10000, random_state=42)

feature_cols = [
    "poi_total_catchment", "Tuition_Weekend_Surge", "Tourist_Peak_Multiplier", 
    "Sports_Big_Match_Spike", "Park_Poya_Outing", "Number_of_Weekends", "Holiday_Count"
]
X = abt[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""))
nb3.cells.append(nbf.v4.new_markdown_cell("### Plot 1: The Elbow Method (WCSS)"))
nb3.cells.append(nbf.v4.new_code_cell("""wcss = []
K_range = range(2, 30, 2)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', color='#1f77b4')
plt.title('The Elbow Method for Optimal K', fontsize=14)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()
"""))
nb3.cells.append(nbf.v4.new_markdown_cell("### Plot 2: 2D Principal Component Analysis (PCA) Projection"))
nb3.cells.append(nbf.v4.new_code_cell("""pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans_final = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_final.labels_, cmap='tab10', alpha=0.6, s=15)
plt.title("PCA Projection of Spatio-Temporal Behavioral Clusters", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster Label")
plt.show()
"""))
nb3.cells.append(nbf.v4.new_markdown_cell("### Plot 3: The Empirical Ceiling (Decensoring)"))
nb3.cells.append(nbf.v4.new_code_cell("""abt['Cluster'] = kmeans_final.labels_
cluster_1 = abt[(abt['Cluster'] == 1) & (abt['Is_Censored'] == 0)]
p90 = cluster_1['Total_Volume'].quantile(0.90)

plt.figure(figsize=(8, 6))
sns.boxplot(y=cluster_1['Total_Volume'], color='lightblue')
plt.axhline(p90, color='red', linestyle='--', linewidth=2, label=f'90th Percentile Ceiling ({p90:.1f} L)')
plt.title("Empirical Ceiling for Cluster 1 (Unconstrained)", fontsize=14)
plt.legend()
plt.show()
"""))
with open('notebooks/03_Cluster_Optimization_and_Base_Math.ipynb', 'w') as f:
    nbf.write(nb3, f)

# ==============================================================================
# Notebook 4: Model Inference and Explainability
# ==============================================================================
nb4 = nbf.v4.new_notebook()
nb4.cells.append(nbf.v4.new_markdown_cell("""# Notebook 4: Model Inference and Explainability
The Goal: Evaluate the LightGBM Heavyweight model, prove its superiority, and finalize the January 2026 predictions.
"""))
nb4.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

sns.set_theme(style="whitegrid")
abt = pd.read_parquet('../data/gold/model_input.parquet')
preds = pd.read_csv('../output/insightai_predictions.csv')
"""))
nb4.cells.append(nbf.v4.new_markdown_cell("### Plot 1: The Right-Censored Scatter (Actual vs. Predicted)"))
nb4.cells.append(nbf.v4.new_code_cell("""historical_max = abt.groupby("Outlet_ID")["Total_Volume"].max().reset_index(name="Historical_Max_Volume")
df_plot = preds.merge(historical_max, on="Outlet_ID", how="inner").sample(1000, random_state=42)

plt.figure(figsize=(10, 10))
plt.scatter(df_plot["Historical_Max_Volume"], df_plot["Maximum_Monthly_Liters"], alpha=0.6, color='purple')
max_val = max(df_plot["Historical_Max_Volume"].max(), df_plot["Maximum_Monthly_Liters"].max())
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label="Perfect Alignment (Actual = Predicted)")

plt.title("Actual vs Predicted Potential (Right-Censored Scatter)", fontsize=14)
plt.xlabel("Historical Max Volume (Liters)")
plt.ylabel("Predicted Potential (Liters)")
plt.legend()
plt.show()
"""))
nb4.cells.append(nbf.v4.new_markdown_cell("### Plot 2: Feature Importance (The Heavyweight)"))
nb4.cells.append(nbf.v4.new_code_cell("""# To plot feature importance without retraining, we load the features into a quick proxy model
features = [
    "poi_total_catchment", "Tuition_Weekend_Surge", "Tourist_Peak_Multiplier", 
    "Sports_Big_Match_Spike", "Park_Poya_Outing", "Number_of_Weekends", "Holiday_Count",
    "poi_count_school", "poi_count_hospital", "Has_Youth_Catchment", "Has_Leisure_Catchment"
]
X_train = abt[abt['Is_Censored']==0][features].fillna(0)
y_train = abt[abt['Is_Censored']==0]['Total_Volume']

model = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=50, random_state=42, verbose=-1)
model.fit(X_train, y_train)

plt.figure(figsize=(10, 8))
lgb.plot_importance(model, max_num_features=15, title="LightGBM Feature Importance (Information Gain)", importance_type='gain', figsize=(10,8))
plt.tight_layout()
plt.show()
"""))
nb4.cells.append(nbf.v4.new_markdown_cell("### Plot 3: The Growth Gap Map (Business Execution)"))
nb4.cells.append(nbf.v4.new_code_cell("""coords = pd.read_parquet('../data/silver/outlet_coordinates_clean.parquet')
df_map = preds.merge(historical_max, on="Outlet_ID", how="inner").merge(coords, on="Outlet_ID", how="inner")
df_map['Growth_Gap'] = df_map['Maximum_Monthly_Liters'] - df_map['Historical_Max_Volume']

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_map['Longitude'], df_map['Latitude'], c=df_map['Growth_Gap'], cmap='coolwarm', alpha=0.8, s=30)
plt.colorbar(scatter, label='Growth Gap (Liters)')
plt.title("Growth Gap Execution Map (Where to Deploy Coolers)", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
"""))
with open('notebooks/04_Model_Inference_and_Explainability.ipynb', 'w') as f:
    nbf.write(nb4, f)

print("Successfully generated the 4-part sequential Jupyter Notebook pipeline in /notebooks")
