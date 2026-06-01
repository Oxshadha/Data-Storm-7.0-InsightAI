import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import os

# Create plots dir if not exists
os.makedirs("output/plots", exist_ok=True)

print("1. Loading data from gold layer...")
abt = pd.read_parquet("data/gold/model_input.parquet")

poi_cols = [c for c in abt.columns if c.startswith("poi_count_")] + ["poi_total_catchment", "poi_driver_catchment", "poi_cannibal_risk"]
catchment_cols = ["Has_High_Footfall_Catchment", "Has_Cannibalization_Risk", "Has_Health_Catchment", "Has_Athletic_Catchment", "Has_Youth_Catchment", "Has_Leisure_Catchment"]
temporal_cols = ["Number_of_Weekends", "Holiday_Count", "Is_Cultural_Month", "Is_High_Season"]
interaction_cols = [
    "Tuition_Weekend_Surge", 
    "Tourist_Peak_Multiplier", 
    "Sports_Big_Match_Spike", 
    "Health_Catchment_Spike"
]
cat_features = ["Outlet_Type"]

features = poi_cols + catchment_cols + temporal_cols + interaction_cols + cat_features
target = "Total_Volume"

print("2. Filtering for uncensored 'Star' shops...")
full_df = abt[abt["Is_Censored"] == 0].copy()

X_train = full_df[features].copy()
y_train = full_df[target]

for c in cat_features:
    X_train[c] = X_train[c].astype('category')

print("3. Training LightGBM Quantile (p90) Model...")
model = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.90,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model.fit(X_train, y_train, categorical_feature=cat_features)

print("4. Generating SHAP values (sampling 5,000 rows for memory efficiency)...")
explainer = shap.TreeExplainer(model)
X_sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
shap_values = explainer.shap_values(X_sample)

print("5. Plotting SHAP Beeswarm Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
plt.title("SHAP Feature Importance (Beeswarm Plot)", fontsize=16)
plt.tight_layout()

output_path = "output/plots/shap_beeswarm.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"6. Success! SHAP Beeswarm saved to {output_path}")
