import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== DATA SCIENTIST HONEST EVALUATION ===")

# 1. Prediction Reality Check
abt = pd.read_parquet('data/gold/model_input.parquet')
preds = pd.read_csv('output/insightai_predictions.csv')

historical_monthly = abt.groupby(['Year', 'Month'])['Total_Volume'].sum().reset_index()
avg_monthly = historical_monthly['Total_Volume'].mean()
max_monthly = historical_monthly['Total_Volume'].max()
pred_total = preds['Maximum_Monthly_Liters'].sum()

print(f"\n1. PREDICTION SANITY CHECK:")
print(f"Historical Avg Monthly Volume: {avg_monthly:,.2f} L")
print(f"Historical Max Monthly Volume: {max_monthly:,.2f} L")
print(f"Predicted Jan 2026 Potential:  {pred_total:,.2f} L")
print(f"Percentage increase over Max:  {((pred_total - max_monthly)/max_monthly)*100:.2f}%")

# 2. Censoring Math Check
print(f"\n2. CENSORING FLAG CHECK:")
censored = abt[abt['Is_Censored'] == 1]
print(f"Total rows flagged as censored: {len(censored)} out of {len(abt)} ({len(censored)/len(abt)*100:.4f}%)")
if len(censored) > 0:
    print(f"Average CV of censored: {censored['Volume_CV'].mean():.4f}")
    print(f"Average CV of uncensored: {abt[abt['Is_Censored'] == 0]['Volume_CV'].mean():.4f}")

# 3. K-Means Check
print(f"\n3. CLUSTER OPTIMIZATION CHECK:")
feature_cols = [
    "poi_total_catchment", "Tuition_Weekend_Surge", "Tourist_Peak_Multiplier", 
    "Sports_Big_Match_Spike", "Number_of_Weekends", "Holiday_Count"
]
sample_X = abt[feature_cols].fillna(0).sample(min(10000, len(abt)), random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sample_X)

wcss = []
for k in [10, 20, 30, 40, 50, 60]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
print(f"WCSS for K=10,20,30,40,50,60: {wcss}")

# 4. Feature Importance Check
print(f"\n4. FEATURE IMPORTANCE CHECK:")
features = feature_cols + ["poi_count_school", "Has_Youth_Catchment", "Has_Leisure_Catchment"]
X_train = abt[abt['Is_Censored']==0][features].fillna(0)
y_train = abt[abt['Is_Censored']==0]['Total_Volume']

model = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=50, random_state=42, verbose=-1)
model.fit(X_train, y_train)
importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
importances = importances.sort_values('Importance', ascending=False).head(5)
print(importances)
