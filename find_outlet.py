import pandas as pd
import numpy as np

df = pd.read_parquet("data/gold/model_input.parquet")
gravity_cols = [c for c in df.columns if c.startswith("gravity_group_")]

best_outlet = None
max_std = -1

for idx, row in df.iterrows():
    pcts = []
    for c in gravity_cols:
        val = float(row.get(c, 0.0))
        pct = (df[c] <= val).mean() * 100
        pcts.append(pct)
    std = np.std(pcts)
    if std > max_std and np.mean(pcts) > 50:
        max_std = std
        best_outlet = row["Outlet_ID"]

print(best_outlet)
