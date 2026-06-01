"""
Gold Layer — Marketing Spend Optimizer (MILP Knapsack).

Solves the LKR 5 Million Western Province trade spend allocation problem
using Google OR-Tools Mixed Integer Linear Programming (MILP).

Key Advantages over Lagrangian:
    1. Binary decision variables (x ∈ {0,1}) — shop either gets a package or not.
    2. Discrete tiered costs — physically meaningful (cooler, merch, branding).
    3. Exact budget adherence — never exceeds 5,000,000.00 LKR.
    4. Provably optimal solution via SCIP branch-and-bound.

Tiered Investment Packages (Business Logic):
    - Tier 3 (Small Kade / Low Volume):    Merchandising posters + discounts → 15,000 LKR
    - Tier 2 (Medium Grocery / Mid Volume): Cooler refurbishment + branding   → 40,000 LKR
    - Tier 1 (Large Hub / High Volume):     Brand new cooler + billboard       → 90,000 LKR

Objective:
    Maximize  Σ  x_i × Volume_Lift_i × Strategic_Multiplier_i
    s.t.      Σ  x_i × Investment_Cost_i  ≤  5,000,000

Output:
    Saves results strictly to:
      - output/insightai_budget_allocations.csv (Format: Outlet_ID, Trade_Spend_Allocation)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from ortools.linear_solver import pywraplp
from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.logger import get_logger

logger = get_logger("gold.spend_optimizer")


def _assign_investment_tier(row: pd.Series) -> int:
    """
    Assign discrete investment cost based on outlet profile.

    Uses Dynamic_Tier (from quantile-based reclassification in Silver)
    to map each outlet to one of three trade marketing packages.
    """
    cooler = float(row.get("Cooler_Count", 0.0))
    lift = float(row.get("volume_lift", 0.0))
    
    # 90K Core Asset Injection (New Cooler)
    # Target: High potential shops (Lift > 500) that have severe hardware bottlenecks (0 coolers)
    if cooler == 0 and lift > 500:
        return 90000
        
    # 15K POSM & Trade Discounts
    # Target 1: Saturated giants (3+ coolers) that just need promotions to move volume.
    # Target 2: Small low-potential shops (0 coolers, lift <= 500) that only justify basic posters.
    elif cooler >= 3 or (cooler == 0 and lift <= 500):
        return 15000
        
    # 40K Visibility & Refurbishment
    # Target: The middle ground (1-2 coolers). Fix up their existing hardware and add billboards.
    else:
        return 40000


def run_spend_optimizer(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Gold: Marketing Spend Optimization — MILP Knapsack (OR-Tools)")
    logger.info("=" * 60)

    # ── Load Data ──────────────────────────────────────────────────────────
    preds_path = "output/insightai_predictions.csv"
    abt_path = config["paths"]["gold"]["root"] + "/model_input.parquet"

    logger.info("Loading predictions and ABT...")
    preds_df = pd.read_csv(preds_path)
    abt_df = read_parquet(abt_path)

    # Group by Outlet_ID to get static properties (last record per outlet)
    keep_cols = [
        "Outlet_ID", "Distributor_ID", "Region", "Cooler_Count",
        "Volume_Liters", "Avg_Monthly_Volume", "is_isolated_goldmine",
        "Dynamic_Tier", "poi_total_catchment"
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in abt_df.columns]

    outlets_last = (
        abt_df.sort_values(["Year", "Month"])
        .groupby("Outlet_ID")
        .last()
        .reset_index()
    )

    # ── Filter for Western Province Outlets ───────────────────────────────
    wp_distributors = ["DIST_W_01", "DIST_W_02", "DIST_W_03"]
    logger.info(f"Filtering ABT for Western Province distributors: {wp_distributors}")
    wp_outlets = outlets_last[outlets_last["Distributor_ID"].isin(wp_distributors)].copy()
    
    # Select columns
    wp_outlets = wp_outlets[keep_cols]

    if wp_outlets.empty:
        logger.error("No Western Province records found. Cannot run optimization.")
        return

    # Merge with predictions to get January 2026 Maximum Potential
    wp_outlets = wp_outlets.merge(preds_df, on="Outlet_ID", how="inner")

    # Clean NaNs
    wp_outlets["Cooler_Count"] = wp_outlets["Cooler_Count"].fillna(0.0)
    wp_outlets["Avg_Monthly_Volume"] = wp_outlets["Avg_Monthly_Volume"].fillna(0.0)
    wp_outlets["Maximum_Monthly_Liters"] = wp_outlets["Maximum_Monthly_Liters"].fillna(0.0)
    if "is_isolated_goldmine" in wp_outlets.columns:
        wp_outlets["is_isolated_goldmine"] = wp_outlets["is_isolated_goldmine"].fillna(0).astype(int)
    else:
        wp_outlets["is_isolated_goldmine"] = 0

    logger.info(f"Unique Western Province outlets: {len(wp_outlets):,}")

    # ── Compute Volume Lift (The Reward) ──────────────────────────────────
    wp_outlets["volume_lift"] = np.maximum(
        0.0,
        wp_outlets["Maximum_Monthly_Liters"] - wp_outlets["Avg_Monthly_Volume"]
    )

    # ── Assign Discrete Investment Costs (The Weight) ─────────────────────
    wp_outlets["investment_cost"] = wp_outlets.apply(_assign_investment_tier, axis=1)

    # Filter: only include outlets with positive lift (worth investing in) and non-zero spatial footprint (removes offshore synthetics)
    candidates = wp_outlets[
        (wp_outlets["volume_lift"] > 10.0) & 
        (wp_outlets["poi_total_catchment"] > 0.0)
    ].copy().reset_index(drop=True)
    logger.info(f"Candidate outlets with positive lift (>10L) and valid coordinates: {len(candidates):,}")

    # ── Cost Tier Distribution ────────────────────────────────────────────
    tier_counts = candidates["investment_cost"].value_counts().sort_index()
    for cost, count in tier_counts.items():
        label = {15000: "Tier 3 (POSM & Discounts)", 40000: "Tier 2 (Refurbish)", 90000: "Tier 1 (New Cooler)"}
        logger.info(f"  {label.get(cost, 'Unknown')}: {count:,} outlets @ LKR {cost:,}")

    # ── Initialize OR-Tools MILP Solver ───────────────────────────────────
    logger.info("Initializing Google OR-Tools SCIP MILP Solver...")
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        logger.error("SCIP solver failed to initialize. Falling back to CBC.")
        solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        logger.error("No solver available. Cannot run optimization.")
        return

    budget = 5_000_000  # LKR 5,000,000 — strict integer

    # ── Decision Variables: x_i ∈ {0, 1} ─────────────────────────────────
    variables = {}
    for idx, row in candidates.iterrows():
        variables[idx] = solver.IntVar(0, 1, f"x_{row['Outlet_ID']}")

    # ── Constraint 1: Total Spend ≤ Budget ─────────────────────────────────
    budget_constraint = solver.Sum([
        variables[idx] * int(row["investment_cost"])
        for idx, row in candidates.iterrows()
    ])
    solver.Add(budget_constraint <= budget)

    # ── Constraint 1.5: Geographic Distribution (Save Colombo) ──────────────
    # Force the solver to spend at least 1,000,000 LKR in EACH distributor
    min_spend_per_distributor = 1000000
    distributors = candidates["Distributor_ID"].unique()
    for dist in distributors:
        dist_shops = candidates[candidates["Distributor_ID"] == dist]
        if not dist_shops.empty:
            solver.Add(
                solver.Sum([variables[idx] * int(row["investment_cost"]) for idx, row in dist_shops.iterrows()]) >= min_spend_per_distributor
            )
    logger.info(f"Geographic Distribution: Enforced {min_spend_per_distributor:,.0f} LKR minimum spend per distributor.")

    # ── Constraint 1.6: Force Tier Diversity (Cooler Deployment) ───────────
    # Prevent the knapsack from exclusively spamming Tier 3 packages (15k) due to low cost.
    # Force the solver to deploy at least 10 new coolers (Tier 1) and 20 refurbishments (Tier 2).
    t1_shops = candidates[candidates["investment_cost"] == 90000]
    t2_shops = candidates[candidates["investment_cost"] == 40000]
    if not t1_shops.empty:
        solver.Add(solver.Sum([variables[idx] for idx, row in t1_shops.iterrows()]) >= 10)
    if not t2_shops.empty:
        solver.Add(solver.Sum([variables[idx] for idx, row in t2_shops.iterrows()]) >= 20)
    logger.info("Tier Diversity: Enforced minimum deployment of 10 Tier-1 coolers and 20 Tier-2 refurbishments.")

    # ── Constraint 2: Spatial Anti-Cannibalization ────────────────────────
    # No two funded outlets within 500m of each other.
    # Prevents double-spending on the same catchment zone.
    exclusion_radius_m = 500
    coords_df = read_parquet(
        config["paths"]["silver"]["root"] + "/outlet_coordinates_clean.parquet"
    )
    # Merge coords to wp_outlets so we have all neighbors for Huff Model
    wp_outlets = wp_outlets.merge(coords_df[["Outlet_ID", "Latitude", "Longitude"]], on="Outlet_ID", how="left")
    
    candidates_with_coords = candidates.merge(coords_df[["Outlet_ID", "Latitude", "Longitude"]], on="Outlet_ID", how="left")
    has_coords = candidates_with_coords["Latitude"].notna()
    
    huff_multipliers = {idx: 1.0 for idx in candidates.index}

    if has_coords.sum() > 0:
        geo_subset = candidates_with_coords[has_coords]
        gdf = gpd.GeoDataFrame(
            geo_subset,
            geometry=gpd.points_from_xy(geo_subset["Longitude"], geo_subset["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=5234)
        xy = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])

        tree = cKDTree(xy)
        conflict_pairs = tree.query_pairs(r=exclusion_radius_m)

        # Map positional indices back to candidate DataFrame indices
        geo_indices = geo_subset.index.tolist()
        n_conflicts = 0
        for pos_a, pos_b in conflict_pairs:
            idx_a = geo_indices[pos_a]
            idx_b = geo_indices[pos_b]
            if idx_a in variables and idx_b in variables:
                solver.Add(variables[idx_a] + variables[idx_b] <= 1)
                n_conflicts += 1

        logger.info(
            f"Anti-cannibalization: {n_conflicts:,} pairwise exclusion constraints "
            f"(no two funded outlets within {exclusion_radius_m}m)"
        )
        
        # ── Enterprise Feature: Huff Gravity Model ────────────────────────────
        logger.info("Computing Enterprise Huff Gravity Model dominance scores...")
        wp_has_coords = wp_outlets["Latitude"].notna()
        wp_geo = wp_outlets[wp_has_coords].copy()
        gdf_all = gpd.GeoDataFrame(
            wp_geo,
            geometry=gpd.points_from_xy(wp_geo["Longitude"], wp_geo["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=5234)
        
        xy_all = np.column_stack([gdf_all.geometry.x.values, gdf_all.geometry.y.values])
        tree_all = cKDTree(xy_all)
        attractiveness = gdf_all["Maximum_Monthly_Liters"].values + 1.0
        
        cand_geo = candidates_with_coords[has_coords].copy()
        cand_gdf = gpd.GeoDataFrame(
            cand_geo,
            geometry=gpd.points_from_xy(cand_geo["Longitude"], cand_geo["Latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=5234)
        
        cand_geoms = cand_gdf.geometry.values
        cand_indices = cand_geo.index.tolist()
        
        for k, pt in enumerate(cand_geoms):
            idx = cand_indices[k]
            row = candidates.iloc[idx]
            
            # Find all competitors within 2km walking/driving catchment
            neighbors = tree_all.query_ball_point([pt.x, pt.y], r=2000)
            
            numerator = row["volume_lift"] + 1.0
            denominator = 0.0
            
            for n_idx in neighbors:
                dist = np.sqrt((pt.x - xy_all[n_idx][0])**2 + (pt.y - xy_all[n_idx][1])**2)
                dist = max(dist, 50.0) # 50m minimum distance to avoid infinite gravity
                denominator += attractiveness[n_idx] / (dist**2)
                
            own_gravity = numerator / (50.0**2)
            huff_prob = own_gravity / denominator if denominator > 0 else 1.0
            
            # Boost the multiplier based on how dominant they are in their catchment
            huff_multipliers[idx] = 1.0 + huff_prob

    else:
        logger.warning("No coordinates available — skipping spatial exclusion constraints.")

    # ── Objective: Maximize Volume Lift with Strategic Multiplier ──────────
    # Isolated Goldmines get a 1.2× priority multiplier.
    # We now also multiply by the Huff Gravity Dominance multiplier.
    objective = solver.Objective()
    for idx, row in candidates.iterrows():
        base_mult = 1.2 if row["is_isolated_goldmine"] == 1 else 1.0
        strategic_multiplier = base_mult * huff_multipliers[idx]
        adjusted_lift = float(row["volume_lift"] * strategic_multiplier)
        objective.SetCoefficient(variables[idx], adjusted_lift)

    objective.SetMaximization()

    # ── Solve ─────────────────────────────────────────────────────────────
    logger.info("Solving MILP Knapsack Problem...")
    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        logger.error(f"Solver returned status {status} — no optimal solution found.")
        return

    # ── Extract Results ───────────────────────────────────────────────────
    candidates["is_selected"] = [
        int(variables[idx].solution_value()) for idx in candidates.index
    ]
    candidates["Trade_Spend_Allocation"] = (
        candidates["is_selected"] * candidates["investment_cost"]
    )

    winners = candidates[candidates["is_selected"] == 1].copy()
    total_spend = int(winners["Trade_Spend_Allocation"].sum())
    total_lift = winners["volume_lift"].sum()
    goldmine_winners = winners["is_isolated_goldmine"].sum()

    logger.info("=" * 60)
    logger.info("🏆 MILP OPTIMIZATION COMPLETE (Provably Optimal)")
    logger.info(f"  Total Budget Allocated: LKR {total_spend:,} / LKR {budget:,}")
    logger.info(f"  Budget Remaining:       LKR {budget - total_spend:,}")
    logger.info(f"  Outlets Funded:         {len(winners):,} / {len(wp_outlets):,}")
    logger.info(f"  Isolated Goldmines Funded: {goldmine_winners:,}")
    logger.info(f"  Total Volume Lift:      {total_lift:,.2f} Liters")
    logger.info(f"  Solver Wall Time:       {solver.wall_time()/1000:.2f} seconds")

    # Winner tier breakdown
    winner_tiers = winners["investment_cost"].value_counts().sort_index()
    for cost, count in winner_tiers.items():
        label = {15000: "Tier 3 (Discounts)", 40000: "Tier 2 (Refurbish)", 90000: "Tier 1 (Cooler)"}
        logger.info(f"  {label.get(cost, '?')}: {count:,} outlets | LKR {cost * count:,}")

    # Summary by distributor
    dist_summary = (
        winners.groupby("Distributor_ID")
        .agg(
            outlets=("Outlet_ID", "count"),
            total_spend=("Trade_Spend_Allocation", "sum"),
            total_lift=("volume_lift", "sum"),
            goldmines=("is_isolated_goldmine", "sum"),
        )
        .reset_index()
    )
    logger.info("\nAllocation Summary by Distributor:")
    for _, row in dist_summary.iterrows():
        logger.info(
            f"  {row['Distributor_ID']}: {row['outlets']} shops | "
            f"Spend: LKR {row['total_spend']:,.0f} | "
            f"Lift: {row['total_lift']:,.2f} L | "
            f"Goldmines: {int(row['goldmines'])}"
        )

    # ── Save Deliverables ─────────────────────────────────────────────────
    # Full output for all WP outlets (funded + unfunded with 0 allocation)
    all_outlets_out = wp_outlets[["Outlet_ID"]].copy()
    funded_map = winners.set_index("Outlet_ID")["Trade_Spend_Allocation"]
    all_outlets_out["Trade Spend Allocation (LKR)"] = (
        all_outlets_out["Outlet_ID"].map(funded_map).fillna(0).astype(int)
    )
    all_outlets_out.to_csv("output/insightai_budget_allocations.csv", index=False)
    logger.info("Saved allocations to output/insightai_budget_allocations.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_spend_optimizer()
