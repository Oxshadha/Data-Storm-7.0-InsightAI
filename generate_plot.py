import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")

print("Loading data...")
abt = pd.read_parquet('data/gold/model_input.parquet')
preds = pd.read_csv('output/insightai_predictions.csv')

# Calculate Historical Max Volume per outlet
historical_max = abt.groupby("Outlet_ID")["Total_Volume"].max().reset_index(name="Historical_Max_Volume")

# Merge
df_plot = preds.merge(historical_max, on="Outlet_ID", how="inner")

# Sort by Historical Max for a clean plot
df_plot = df_plot.sort_values("Historical_Max_Volume").reset_index(drop=True)

print("Generating scatter plot...")
plt.figure(figsize=(14, 8))

# Plot a subset for clarity (e.g., sample of 2000 shops)
sample_df = df_plot.sample(2000, random_state=42).sort_values("Historical_Max_Volume").reset_index(drop=True)

# Plot Historical Actuals (Blue Dots)
plt.scatter(sample_df.index, sample_df["Historical_Max_Volume"], 
            alpha=0.6, color='#1f77b4', s=30, label='Historical Max Volume (Actuals)')

# Plot Predicted Potential (Green Stars)
plt.scatter(sample_df.index, sample_df["Maximum_Monthly_Liters"], 
            alpha=0.8, color='#2ca02c', marker='*', s=80, label='Predicted Potential (LightGBM Jan 2026)')

# Highlight the Gap (only for the ones that have a gap)
gap_mask = sample_df["Maximum_Monthly_Liters"] > sample_df["Historical_Max_Volume"]
for i in sample_df[gap_mask].index:
    plt.plot([i, i], 
             [sample_df.loc[i, "Historical_Max_Volume"], sample_df.loc[i, "Maximum_Monthly_Liters"]], 
             color='gray', linestyle='--', alpha=0.3)

plt.title("Visualizing Decensoring: Actual Capacity vs. Latent Potential Demand", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Outlet Rank (Sorted by Historical Size)", fontsize=14)
plt.ylabel("Volume (Liters)", fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()

out_path = 'output/plot_censored_demand.png'
plt.savefig(out_path, dpi=300)
print(f"Plot successfully saved to {out_path}")
