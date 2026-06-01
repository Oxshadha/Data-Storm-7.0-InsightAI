import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="paper")
out_dir = Path("output/plots")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading data for Optimization Exports...")
allocs = pd.read_csv("output/insightai_budget_allocations.csv")
preds = pd.read_csv("output/insightai_predictions.csv")
abt = pd.read_parquet("data/gold/model_input.parquet")

# 1. Prepare Data
funded = allocs[allocs["Trade_Spend_Allocation"] > 0].copy()

# Map packages
def map_package(cost):
    if cost == 90000: return "90K Core Asset Injection"
    elif cost == 40000: return "40K Visibility & Refurbish"
    else: return "15K POSM & Discounts"
    
funded["Package"] = funded["Trade_Spend_Allocation"].apply(map_package)

# Merge with predictions and abt to calculate ROI
outlets = abt.sort_values(["Year", "Month"]).groupby("Outlet_ID").last().reset_index()
funded = funded.merge(preds, on="Outlet_ID", how="inner")
funded = funded.merge(outlets[["Outlet_ID", "Avg_Monthly_Volume"]], on="Outlet_ID", how="inner")

funded["Volume_Lift"] = np.maximum(0, funded["Maximum_Monthly_Liters"] - funded["Avg_Monthly_Volume"])
funded["ROI"] = funded["Volume_Lift"] / (funded["Trade_Spend_Allocation"] / 1000)

# --- Plot 6: Budget Allocation by Strategic Package ---
print("Generating Plot 6: Budget Allocation...")
package_spend = funded.groupby("Package")["Trade_Spend_Allocation"].sum().reset_index()
package_spend = package_spend.sort_values("Trade_Spend_Allocation", ascending=True)

plt.figure(figsize=(9, 5))
colors = ['#1e88e5', '#ffb300', '#43a047']
ax = sns.barplot(x="Trade_Spend_Allocation", y="Package", data=package_spend, palette=colors)

for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(width + 50000, p.get_y() + p.get_height()/2. + 0.1, 
            f"LKR {width/1000000:.2f}M", ha="left", fontweight='bold')

plt.title("LKR 5M Budget Allocation by Strategic Package", fontweight='bold', fontsize=14)
plt.xlabel("Total Budget Allocated (LKR)")
plt.ylabel("")
plt.xlim(0, 4000000)
plt.tight_layout()
plt.savefig(out_dir / "plot_6_budget_allocation.png", dpi=300)
plt.close()

# --- Plot 7: ROI Optimization Yield ---
print("Generating Plot 7: ROI Yield...")
roi_yield = funded.groupby("Package")["ROI"].mean().reset_index()
roi_yield = roi_yield.sort_values("ROI", ascending=True)

plt.figure(figsize=(9, 5))
ax = sns.barplot(x="ROI", y="Package", data=roi_yield, palette="rocket")

for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(width + 0.5, p.get_y() + p.get_height()/2. + 0.1, 
            f"{width:.1f} L", ha="left", fontweight='bold')

plt.title("Average ROI Yield (Liters per 1K LKR Spent)", fontweight='bold', fontsize=14)
plt.xlabel("Incremental Liters Generated per 1,000 LKR")
plt.ylabel("")
plt.xlim(0, max(roi_yield["ROI"]) * 1.2)
plt.tight_layout()
plt.savefig(out_dir / "plot_7_roi_yield.png", dpi=300)
plt.close()

print("Optimization plots generated successfully!")
