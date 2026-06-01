import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("Loading data...")
try:
    df = pd.read_parquet("data/gold/model_input.parquet")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    censored = df[df["Is_Censored"] == 1]
    uncensored = df[df["Is_Censored"] == 0]
    
    sns.kdeplot(data=uncensored, x="poi_driver_catchment", label="Uncensored (Training)", fill=True, color="#3b82f6", alpha=0.5, ax=ax)
    sns.kdeplot(data=censored, x="poi_driver_catchment", label="Censored (Target)", fill=True, color="#f43f5e", alpha=0.5, ax=ax)
    
    ax.set_title("Selection Bias Defense: Spatial Gravity Overlap", fontsize=14, color="white", pad=15)
    ax.set_xlabel("Spatial Driver Gravity (poi_driver_catchment)", color="#94a3b8")
    ax.set_ylabel("Density", color="#94a3b8")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94a3b8')
    
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor="white")
    
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    os.makedirs("output/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("output/plots/plot_8_selection_bias.png", transparent=True, dpi=300)
    print("Saved plot_8_selection_bias.png")
except Exception as e:
    print(f"Error: {e}")
