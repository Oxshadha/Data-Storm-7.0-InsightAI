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
    tier = str(row.get("Dynamic_Tier", "Tier-4"))
    cooler = float(row.get("Cooler_Count", 0))

    # Tier 1: High-volume hubs (large grocery, major outlets with coolers)
    if tier in ("Tier-1",) or cooler >= 4:
        return 90000
    # Tier 2: Medium outlets
    elif tier in ("Tier-2",) or cooler >= 2:
        return 40000
    # Tier 3: Small kades, minimal infrastructure
    else:
        return 15000


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

    # ── Filter for Western Province Outlets ───────────────────────────────
    wp_distributors = ["DIST_W_01", "DIST_W_02", "DIST_W_03"]
    logger.info(f"Filtering ABT for Western Province distributors: {wp_distributors}")
    wp_abt = abt_df[abt_df["Distributor_ID"].isin(wp_distributors)]

    if wp_abt.empty:
        logger.error("No Western Province records found in ABT. Cannot run optimization.")
        return

    # Group by Outlet_ID to get static properties (last record per outlet)
    logger.info("Extracting static outlet properties...")
    keep_cols = [
        "Outlet_ID", "Distributor_ID", "Cooler_Count", "Dynamic_Tier",
        "Avg_Monthly_Volume", "is_isolated_goldmine",
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in wp_abt.columns]

    wp_outlets = (
        wp_abt.sort_values(["Year", "Month"])
        .groupby("Outlet_ID")
        .last()
        .reset_index()[keep_cols]
    )

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

    # Filter: only include outlets with positive lift (worth investing in)
    candidates = wp_outlets[wp_outlets["volume_lift"] > 10.0].copy().reset_index(drop=True)
    logger.info(f"Candidate outlets with positive lift (>10L): {len(candidates):,}")

    # ── Cost Tier Distribution ────────────────────────────────────────────
    tier_counts = candidates["investment_cost"].value_counts().sort_index()
    for cost, count in tier_counts.items():
        label = {15000: "Tier 3 (Merchandising)", 40000: "Tier 2 (Refurbish)", 90000: "Tier 1 (New Cooler)"}
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

    # ── Constraint: Total Spend ≤ Budget ──────────────────────────────────
    budget_constraint = solver.Sum([
        variables[idx] * int(row["investment_cost"])
        for idx, row in candidates.iterrows()
    ])
    solver.Add(budget_constraint <= budget)

    # ── Objective: Maximize Volume Lift with Strategic Multiplier ──────────
    # Isolated Goldmines get a 1.2× priority multiplier:
    # This forces the solver to mathematically prefer shops with massive
    # footfall and zero nearby competition.
    objective = solver.Objective()
    for idx, row in candidates.iterrows():
        strategic_multiplier = 1.2 if row["is_isolated_goldmine"] == 1 else 1.0
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
        label = {15000: "Tier 3 (Merch)", 40000: "Tier 2 (Refurb)", 90000: "Tier 1 (Cooler)"}
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
    all_outlets_out["Trade_Spend_Allocation"] = (
        all_outlets_out["Outlet_ID"].map(funded_map).fillna(0).astype(int)
    )
    all_outlets_out.to_csv("output/insightai_budget_allocations.csv", index=False)
    logger.info("Saved allocations to output/insightai_budget_allocations.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_spend_optimizer()
