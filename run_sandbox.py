import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== SANDBOX EVALUATION DATA ===")

abt = pd.read_parquet('data/gold/model_input.parquet')

# 1. Censoring Threshold
print(f"Total Rows: {len(abt[abt['Total_Volume'] > 0])}")
print(f"Rows caught at CV < 0.15: {len(abt[(abt['Volume_CV'] < 0.15) & (abt['Total_Volume'] > 0)])}")
print(f"Rows caught at CV < 0.25: {len(abt[(abt['Volume_CV'] < 0.25) & (abt['Total_Volume'] > 0)])}")
print(f"Rows caught at CV < 0.35: {len(abt[(abt['Volume_CV'] < 0.35) & (abt['Total_Volume'] > 0)])}")

# 2. Tuning LightGBM Alpha
feature_cols = [
    "poi_total_catchment", "Tuition_Weekend_Surge", "Tourist_Peak_Multiplier", 
    "Sports_Big_Match_Spike", "Park_Poya_Outing", "Number_of_Weekends", "Holiday_Count"
]
features = feature_cols + ["poi_count_school", "Has_Youth_Catchment", "Has_Leisure_Catchment", "Outlet_Type", "Dynamic_Tier"]
train_df = abt.sample(10000, random_state=42).copy()
for c in ["Outlet_Type", "Dynamic_Tier"]:
    train_df[c] = train_df[c].astype('category')

X_train = train_df[features].fillna(0)
y_train = train_df['Total_Volume']

alphas = [0.75, 0.85, 0.90]
results = {}

for alpha in alphas:
    model = lgb.LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=50, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    results[f'Alpha={alpha}'] = preds.sum()

historical_sum = train_df.groupby("Outlet_ID")["Total_Volume"].max().sum()

print("\n--- Alpha Tuning ---")
print(f"Historical Max Sum: {historical_sum:,.2f}")
for k, v in results.items():
    pct = ((v - historical_sum) / historical_sum) * 100
    print(f"{k}: {v:,.2f} (+{pct:.1f}%)")
