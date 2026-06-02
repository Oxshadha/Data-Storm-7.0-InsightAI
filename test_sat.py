import pandas as pd
df = pd.read_parquet("data/gold/model_input.parquet")
print("Total rows:", len(df))
print("Rows > 0:", len(df[df["competitive_saturation_index"] > 0]))
print("Rows > 0.05:", len(df[df["competitive_saturation_index"] > 0.05]))
print("Min > 0:", df[df["competitive_saturation_index"] > 0]["competitive_saturation_index"].min())
print("Max:", df["competitive_saturation_index"].max())
