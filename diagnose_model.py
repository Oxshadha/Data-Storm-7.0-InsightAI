import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== DIAGNOSING THE WORST MODEL ===")
abt = pd.read_parquet('data/gold/model_input.parquet')

features_to_test = [
    'Total_Volume', 'poi_total_catchment', 'Tuition_Weekend_Surge', 
    'Tourist_Peak_Multiplier', 'Sports_Big_Match_Spike', 'Park_Poya_Outing',
    'Has_Youth_Catchment', 'Holiday_Count'
]
corr_matrix = abt[features_to_test].corr()
print("\nCorrelation with Total Volume:")
print(corr_matrix['Total_Volume'].sort_values(ascending=False))

unconstrained = abt[abt['Is_Censored'] == 0].dropna()
features = [
    "poi_total_catchment", "Tuition_Weekend_Surge", "Tourist_Peak_Multiplier", 
    "Sports_Big_Match_Spike", "Park_Poya_Outing", "Number_of_Weekends", "Holiday_Count",
    "poi_count_school"
]
X = unconstrained[features]
y = unconstrained['Total_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nR2 Score: {r2}")
print(f"MAE: {mae}")
