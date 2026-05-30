"""
Gold Layer — Marketing Spend Optimizer.

Solves the LKR 5 Million Western Province trade spend allocation problem.
Uses a mathematically rigorous Lagrange Multiplier framework to maximize
incremental volume lift subject to budget and operational constraints.

Response Model:
    Incremental_Lift_i = Growth_Gap_i * (1 - e^(-alpha_i * S_i))
    where:
      - Growth_Gap_i = Max_Potential_i (predicted Jan 2026) - Historical_Baseline_i
      - S_i = allocated spend in LKR
      - alpha_i = efficiency coefficient reflecting spatial and assets:
          alpha_i = alpha_base * (1 + 2 * scaled_ratio_i) * (1 + 0.5 * Cooler_Count_i)

Optimization Algorithm:
    Since the objective is concave and separable, we find the exact global optimum
    using Lagrange multipliers and binary search over the shadow price lambda.
    This solves the 9,000-variable optimization problem in milliseconds.

Output:
    Saves results strictly to:
      - output/insightai_budget_allocations.csv (Format: Outlet_ID, Trade_Spend_Allocation)
      - output/InsightAI_budget_allocations.csv (duplicate for case-insensitive matching)
"""

import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.logger import get_logger

logger = get_logger("gold.spend_optimizer")


def run_spend_optimizer(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Gold: Marketing Spend Optimization (Western Province LKR 5M)")
    logger.info("=" * 60)

    # ── Load Data ──────────────────────────────────────────────────────────
    preds_path = "output/insightai_predictions.csv"
    abt_path = config["paths"]["gold"]["root"] + "/model_input.parquet"

    logger.info("Loading predictions and ABT...")
    preds_df = pd.read_csv(preds_path)
    abt_df = read_parquet(abt_path)

    # ── Filter for Western Province Outlets ───────────────────────────────
    # Western Province distributors: DIST_W_01, DIST_W_02, DIST_W_03
    wp_distributors = ["DIST_W_01", "DIST_W_02", "DIST_W_03"]
    logger.info(f"Filtering ABT for Western Province distributors: {wp_distributors}")
    wp_abt = abt_df[abt_df["Distributor_ID"].isin(wp_distributors)]

    if wp_abt.empty:
        logger.error("No Western Province records found in ABT. Cannot run optimization.")
        return

    # Group by Outlet_ID to get static properties (last record per outlet)
    logger.info("Extracting static outlet properties...")
    wp_outlets = (
        wp_abt.sort_values(["Year", "Month"])
        .groupby("Outlet_ID")
        .last()
        .reset_index()[["Outlet_ID", "Distributor_ID", "Cooler_Count", "Avg_Monthly_Volume", "latent_opportunity_ratio"]]
    )

    # Merge with predictions to get January 2026 Maximum Potential
    wp_outlets = wp_outlets.merge(preds_df, on="Outlet_ID", how="inner")
    
    # Clean NaNs to prevent solver failure
    wp_outlets["Cooler_Count"] = wp_outlets["Cooler_Count"].fillna(0.0)
    wp_outlets["Avg_Monthly_Volume"] = wp_outlets["Avg_Monthly_Volume"].fillna(0.0)
    wp_outlets["latent_opportunity_ratio"] = wp_outlets["latent_opportunity_ratio"].fillna(0.0)
    wp_outlets["Maximum_Monthly_Liters"] = wp_outlets["Maximum_Monthly_Liters"].fillna(0.0)
    
    logger.info(f"Unique Western Province outlets in predictions: {len(wp_outlets):,}")

    # ── Mathematical Formulation ────────────────────────────────────────────
    # 1. Growth Gap (Max possible incremental volume)
    # G_i = max(0, Potential - Historical_Avg)
    wp_outlets["growth_gap"] = np.maximum(
        0.0,
        wp_outlets["Maximum_Monthly_Liters"] - wp_outlets["Avg_Monthly_Volume"]
    )

    # 2. Scaled Latent Opportunity Ratio (using log-scaling to compress outliers)
    max_ratio = wp_outlets["latent_opportunity_ratio"].max()
    wp_outlets["scaled_ratio"] = np.log1p(wp_outlets["latent_opportunity_ratio"]) / np.log1p(max_ratio)

    # 3. Efficiency Coefficient (alpha_i)
    # alpha_base = 0.00005 (so LKR 10,000 gets ~39% lift for base shop, ~90% for top shop)
    alpha_base = 0.00005
    wp_outlets["alpha"] = (
        alpha_base *
        (1.0 + 2.0 * wp_outlets["scaled_ratio"]) *
        (1.0 + 0.5 * wp_outlets["Cooler_Count"].fillna(0))
    )

    # ── Optimization Parameters ────────────────────────────────────────────
    budget = 5000000.0  # LKR 5,000,000
    max_spend_per_outlet = 50000.0  # LKR 50,000 (operational constraint)

    G = wp_outlets["growth_gap"].values
    alpha = wp_outlets["alpha"].values
    outlets = wp_outlets["Outlet_ID"].values

    # Marginal ROI at S_i = 0 is G_i * alpha_i
    marginal_roi_zero = G * alpha

    # ── Solver: Binary Search over Lagrange shadow price lambda ────────────
    logger.info("Running optimization solver (Lagrangian binary search)...")

    def compute_spend(lambd: float) -> np.ndarray:
        # S_i = min(S_max, max(0, 1/alpha * ln(G * alpha / lambda)))
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.log((G * alpha) / lambd) / alpha
        spend = np.clip(val, 0.0, max_spend_per_outlet)
        # For zero growth gap or zero alpha, spend must be 0
        spend[np.isnan(spend) | np.isinf(spend)] = 0.0
        spend[G <= 0.0] = 0.0
        return spend

    # Binary search bounds
    low = 1e-12
    high = float(np.max(marginal_roi_zero)) + 1e-5

    if high <= low:
        logger.warning("No outlets have a positive growth gap. Budget cannot be allocated.")
        wp_outlets["Trade_Spend_Allocation"] = 0.0
    else:
        # Solve
        for iteration in range(100):
            mid = (low + high) / 2.0
            spends = compute_spend(mid)
            total_spend = spends.sum()

            if abs(total_spend - budget) < 1.0:  # Within 1 LKR
                break
            elif total_spend > budget:
                low = mid  # Need higher lambda (higher shadow price -> lower spend)
            else:
                high = mid  # Need lower lambda

        wp_outlets["Trade_Spend_Allocation"] = np.round(compute_spend(mid), 2)

    # ── Calculate Impact & Metrics ─────────────────────────────────────────
    wp_outlets["incremental_lift"] = np.round(
        wp_outlets["growth_gap"] * (1.0 - np.exp(-wp_outlets["alpha"] * wp_outlets["Trade_Spend_Allocation"])),
        2
    )

    allocated_outlets = wp_outlets[wp_outlets["Trade_Spend_Allocation"] > 0]
    total_allocated = wp_outlets["Trade_Spend_Allocation"].sum()
    total_lift = wp_outlets["incremental_lift"].sum()
    max_alloc_reached = (wp_outlets["Trade_Spend_Allocation"] >= max_spend_per_outlet - 1.0).sum()

    logger.info(f"Optimization complete:")
    logger.info(f"  Total Budget Allocated: LKR {total_allocated:,.2f} / LKR {budget:,.2f}")
    logger.info(f"  Outlets Receiving Budget: {len(allocated_outlets):,} / {len(wp_outlets):,}")
    logger.info(f"  Total Incremental Volume Lift: {total_lift:,.2f} Liters")
    logger.info(f"  Outlets hitting max spend ({max_spend_per_outlet} LKR): {max_alloc_reached:,}")

    # Summary by distributor
    dist_summary = (
        wp_outlets.groupby("Distributor_ID")
        .agg(
            outlets=("Outlet_ID", "count"),
            funded_outlets=("Trade_Spend_Allocation", lambda x: (x > 0).sum()),
            total_spend=("Trade_Spend_Allocation", "sum"),
            avg_spend=("Trade_Spend_Allocation", lambda x: x[x > 0].mean()),
            total_lift=("incremental_lift", "sum"),
        )
        .reset_index()
    )
    logger.info("\nAllocation Summary by Distributor:")
    for _, row in dist_summary.iterrows():
        logger.info(
            f"  {row['Distributor_ID']}: {row['funded_outlets']}/{row['outlets']} shops funded | "
            f"Spend: LKR {row['total_spend']:,.2f} | Lift: {row['total_lift']:,.2f} L"
        )

    # ── Save strictly to required outputs ─────────────────────────────────
    out_df = wp_outlets[["Outlet_ID", "Trade_Spend_Allocation"]]
    out_df.to_csv("output/insightai_budget_allocations.csv", index=False)
    out_df.to_csv("output/InsightAI_budget_allocations.csv", index=False)
    logger.info("Saved allocations to output/insightai_budget_allocations.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_spend_optimizer()
