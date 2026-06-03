import pandas as pd
alloc = pd.read_csv("output/insightai_budget_allocations.csv")
t1 = alloc[alloc["Trade Spend Allocation (LKR)"] == 90000].head(5)["Outlet_ID"].tolist()
t2 = alloc[alloc["Trade Spend Allocation (LKR)"] == 40000].head(5)["Outlet_ID"].tolist()
t3 = alloc[alloc["Trade Spend Allocation (LKR)"] == 15000].head(5)["Outlet_ID"].tolist()
print("90K Outlets:", t1)
print("40K Outlets:", t2)
print("15K Outlets:", t3)
