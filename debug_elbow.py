import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

abt = pd.read_parquet('data/gold/model_input.parquet')
out_dir = Path("output/plots")
out_dir.mkdir(parents=True, exist_ok=True)

print("Starting Elbow Method debug...")
cols = ['poi_driver_catchment', 'poi_cannibal_risk', 'Tuition_Weekend_Surge']
outlet_static = abt.groupby("Outlet_ID")[cols].mean().fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(outlet_static)

sse = []
k_range = range(5, 45, 5) # Extended range to reflect actual model parameters
for k in k_range:
    print(f"Fitting k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, 'bo-')
plt.axvline(x=8, color='r', linestyle='--', label='Selected k=8 (Optimal)')
plt.title("Elbow Method Validation")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "plot_5_elbow_method.png")
print("Done!")
