import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_charts():
    out_dir = Path("output/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Anomaly Summary (from our Rejection Manifest)
    # We'll use the known numbers from our quarantine process
    labels = ["Negative Volumes\n(Returns)", "Zero Volumes", "Coordinates\nOut of Bounds", "Duplicate Holidays"]
    counts = [4753, 100, 480, 93]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=counts, palette="Reds_r")
    plt.title("Data Forensics: Quarantined Records by Type", fontsize=14, pad=20)
    plt.ylabel("Number of Records Quarantined")
    for i, v in enumerate(counts):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "anomaly_summary.png", dpi=150)
    plt.close()
    
    # 2. Outlet Master Cleaned Types
    # Showing how typos were fixed
    types_before = {"Grocery": 2768, "Grocry": 390, "Bakery": 2678, "Bakry": 395, "Eatery": 2667, " Eatery ": 200}
    types_after = {"Grocery": 3158, "Bakery": 3073, "Eatery": 2867}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before
    ax1.barh(list(types_before.keys()), list(types_before.values()), color='lightcoral')
    ax1.set_title("Before Cleaning (Raw Bronze)")
    ax1.set_xlabel("Count")
    
    # After
    ax2.barh(list(types_after.keys()), list(types_after.values()), color='mediumseagreen')
    ax2.set_title("After Cleaning (Clean Silver)")
    ax2.set_xlabel("Count")
    
    plt.suptitle("Outlet Type Standardization (Fixing Typos)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "outlet_typos.png", dpi=150)
    plt.close()
    
    print("Charts generated in output/charts/")

if __name__ == "__main__":
    generate_charts()
