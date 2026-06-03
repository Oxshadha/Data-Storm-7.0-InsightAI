import time
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import warnings
warnings.filterwarnings('ignore')

def load_dashboard_data():
    abt = pd.read_parquet('data/gold/model_input.parquet')
    outlets = abt.sort_values(['Year', 'Month']).groupby('Outlet_ID').last().reset_index()
    preds = pd.read_csv('output/insightai_predictions.csv')
    outlets = outlets.merge(preds, on='Outlet_ID', how='inner')
    
    def get_province(dist_id):
        if str(dist_id).startswith("DIST_W"): return "Western"
        elif str(dist_id).startswith("DIST_C"): return "Central"
        elif str(dist_id).startswith("DIST_NW"): return "North-Western"
        elif str(dist_id).startswith("DIST_S"): return "Southern"
        return "Other"
    
    outlets["Province"] = outlets["Distributor_ID"].apply(get_province)
    outlets["Raw_Volume_Lift"] = np.maximum(0, outlets["Maximum_Monthly_Liters"] - outlets["Avg_Monthly_Volume"])
    outlets["Volume_Lift"] = outlets["Raw_Volume_Lift"] 
    return outlets

print('loading data...')
df = load_dashboard_data()
print('data loaded')

optim_budget = 5000000

wp_outlets = df[df["Province"] == "Western"].copy()

def assign_tier(row):
    cooler = float(row.get("Cooler_Count", 0.0))
    lift = float(row.get("Volume_Lift", 0.0))
    if cooler == 0 and lift > 500: return 90000
    elif cooler >= 3 or (cooler == 0 and lift <= 500): return 15000
    else: return 40000

wp_outlets["investment_cost"] = wp_outlets.apply(assign_tier, axis=1)
candidates = wp_outlets[wp_outlets["Volume_Lift"] > 10.0].copy().reset_index(drop=True)

print(f"Number of candidates: {len(candidates)}")

solver = pywraplp.Solver.CreateSolver("SCIP")
variables = {}
for idx, row in candidates.iterrows():
    variables[idx] = solver.IntVar(0, 1, f"x_{idx}")

solver.Add(solver.Sum([variables[idx] * int(row["investment_cost"]) for idx, row in candidates.iterrows()]) <= optim_budget)

min_spend = optim_budget * 0.20
for dist in candidates["Distributor_ID"].unique():
    dist_shops = candidates[candidates["Distributor_ID"] == dist]
    if not dist_shops.empty:
        solver.Add(solver.Sum([variables[idx] * int(row["investment_cost"]) for idx, row in dist_shops.iterrows()]) >= min_spend)

t1_shops = candidates[candidates["investment_cost"] == 90000]
t2_shops = candidates[candidates["investment_cost"] == 40000]
if not t1_shops.empty:
    solver.Add(solver.Sum([variables[idx] for idx, row in t1_shops.iterrows()]) >= 10)
if not t2_shops.empty:
    solver.Add(solver.Sum([variables[idx] for idx, row in t2_shops.iterrows()]) >= 20)

objective = solver.Objective()
for idx, row in candidates.iterrows():
    mult = 1.2 if row["is_isolated_goldmine"] == 1 else 1.0
    objective.SetCoefficient(variables[idx], float(row["Volume_Lift"] * mult))
objective.SetMaximization()

print("Solving...")
start = time.time()
status = solver.Solve()
end = time.time()
print(f"Status: {status}, Time taken: {end-start:.2f}s")
