import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== DIAGNOSING THE TRUE MODEL ===")
abt = pd.read_parquet('data/gold/model_input.parquet')

# Features to use (same as lgbm_quantile.py)
poi_cols = [c for c in abt.columns if c.startswith("poi_count_")] + ["poi_total_catchment"]
catchment_cols = ["Has_Youth_Catchment", "Has_Leisure_Catchment", "Has_Athletic_Catchment"]
temporal_cols = ["Number_of_Weekends", "Holiday_Count", "Is_Cultural_Month", "Is_High_Season"]
interaction_cols = [
    "Tuition_Weekend_Surge", 
    "Tourist_Peak_Multiplier", 
    "Sports_Big_Match_Spike", 
    "Park_Poya_Outing"
]

cat_features = ["Outlet_Type", "Dynamic_Tier"]

features = poi_cols + catchment_cols + temporal_cols + interaction_cols + cat_features

# Evaluate the unconstrained shops
unconstrained = abt[abt['Is_Censored'] == 0].dropna()
X = unconstrained[features].copy()
y = unconstrained['Total_Volume']

for c in cat_features:
    X[c] = X[c].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
model.fit(X_train, y_train, categorical_feature=cat_features)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"True R2 Score: {r2}")
print(f"True MAE: {mae}")

# Calculate proper correlation using numerical encoding for categorical
abt_corr = abt.copy()
abt_corr["Dynamic_Tier_Num"] = abt_corr["Dynamic_Tier"].astype('category').cat.codes
corr_matrix = abt_corr[["Total_Volume", "Dynamic_Tier_Num", "poi_total_catchment", "Tuition_Weekend_Surge"]].corr()
print("\nCorrelation with Total Volume:")
print(corr_matrix['Total_Volume'])
