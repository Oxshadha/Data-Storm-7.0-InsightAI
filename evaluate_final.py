import pandas as pd
print("=== FINAL TUNED PIPELINE METRICS ===")
abt = pd.read_parquet('data/gold/model_input.parquet')
preds = pd.read_csv('output/insightai_predictions.csv')

censored = abt[abt['Is_Censored'] == 1]
print(f"Censored Flag Check:")
print(f"- Rows flagged as mathematically constrained: {len(censored)} ({len(censored)/len(abt)*100:.2f}%)")

historical_monthly = abt.groupby(['Year', 'Month'])['Total_Volume'].sum().reset_index()
max_monthly = historical_monthly['Total_Volume'].max()
pred_total = preds['Maximum_Monthly_Liters'].sum()

print(f"\nDemand Projection Check (Jan 2026):")
print(f"- Historical Max Monthly Network Capacity: {max_monthly:,.2f} L")
print(f"- New Projected Jan 2026 Network Potential: {pred_total:,.2f} L")
print(f"- Target Growth Percentage: +{((pred_total - max_monthly)/max_monthly)*100:.1f}%")
