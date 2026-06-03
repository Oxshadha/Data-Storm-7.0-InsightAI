import pandas as pd
df = pd.read_parquet("data/gold/model_input.parquet")
gravity_cols = [c for c in df.columns if c.startswith("gravity_group_")]
outlet_row = df[df["Outlet_ID"] == "OUT_08605"].iloc[0].to_dict()
single_pcts = []
for c in gravity_cols:
    val = float(outlet_row.get(c, 0.0))
    pct = (df[c] <= val).mean() * 100
    single_pcts.append(pct)
print(single_pcts)
