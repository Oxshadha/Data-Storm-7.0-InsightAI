"""
Predictive Modeling — Route 2: Quantile Gradient Boosting (LightGBM)

Implements the predictive heavyweight algorithm using LightGBM.
Optimizes the pinball loss function across a full quantile surface 
(alpha=[0.50, 0.75, 0.90]) to map the non-linear unconstrained demand surface.
Provides specific potential estimates for January 2026 without artificial safety floors.

Steps:
1. Train on historical uncensored data (Is_Censored == 0).
2. Generate synthetic grid for January 2026 (incorporating specific Jan holidays, weekends, seasonality).
3. Predict the 50th, 75th, and 90th percentile capacities for all outlets in Jan 2026.
4. Export final predictions CSV containing the full surface.

Usage:
    python -m src.model.lgbm_quantile
"""

import pandas as pd
import numpy as np
import calendar
import lightgbm as lgb
from src.utils.config import load_config, resolve_path
from src.utils.io import read_parquet
from src.utils.logger import get_logger

logger = get_logger("model.lgbm_quantile")

def get_weekend_days(year: int, month: int) -> int:
    cal = calendar.monthcalendar(year, month)
    return sum(1 for week in cal if week[5] != 0) + sum(1 for week in cal if week[6] != 0)

def run_lgbm_model(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Modeling: Running LightGBM Quantile Regressor")

    paths = config["paths"]

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    logger.info("Loading ABT and Master Data...")
    abt = read_parquet(paths["gold"]["root"] + "/model_input.parquet")
    
    # Extract unique outlet configurations (Distributor, Master, POIs)
    # We take the most recent configuration for each outlet
    outlet_static = abt.sort_values(["Year", "Month"]).groupby("Outlet_ID").tail(1).reset_index(drop=True)
    
    # Features to use
    poi_cols = [c for c in abt.columns if c.startswith("poi_count_")] + ["poi_total_catchment"]
    catchment_cols = ["Has_Youth_Catchment", "Has_Leisure_Catchment", "Has_Athletic_Catchment"]
    temporal_cols = ["Number_of_Weekends", "Holiday_Count", "Is_Cultural_Month", "Is_High_Season"]
    interaction_cols = [
        "Tuition_Weekend_Surge", 
        "Tourist_Peak_Multiplier", 
        "Sports_Big_Match_Spike", 
        "Park_Poya_Outing"
    ]
    
    cat_features = ["Outlet_Type", "Dynamic_Tier"]
    
    features = poi_cols + catchment_cols + temporal_cols + interaction_cols + cat_features
    target = "Total_Volume"

    # ── 2. The Training Phase ─────────────────────────────────────────────────
    logger.info("Preparing Training Data (Uncensored 'Stars' Only)...")
    
    # Filter to strictly uncensored historical rows
    full_df = abt[abt["Is_Censored"] == 0].copy()
    
    # Sort chronologically for Time-Series Split
    full_df = full_df.sort_values(["Year", "Month"])
    
    # We use the final historical month as the Validation set for Early Stopping
    last_year = full_df["Year"].max()
    last_month = full_df[full_df["Year"] == last_year]["Month"].max()
    val_mask = (full_df["Year"] == last_year) & (full_df["Month"] == last_month)
    
    train_df = full_df[~val_mask]
    val_df = full_df[val_mask]
    
    # Prepare X and y
    X_train = train_df[features].copy()
    y_train = train_df[target]
    
    X_val = val_df[features].copy()
    y_val = val_df[target]
    
    # LightGBM handles categoricals natively, but they need to be type 'category'
    for c in cat_features:
        X_train[c] = X_train[c].astype('category')
        X_val[c] = X_val[c].astype('category')
        
    logger.info(f"Training LightGBM on {len(X_train):,} training records and {len(X_val):,} validation records...")
    
    # We will train 3 separate models for a full Quantile Surface
    alphas = [0.50, 0.75, 0.90]
    models = {}
    
    for alpha in alphas:
        logger.info(f"Training Quantile Regressor (alpha={alpha})...")
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=300, # Max estimators, early stopping will halt it
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Train with Early Stopping to prevent overfitting
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        
        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            eval_metric='quantile',
            categorical_feature=cat_features,
            callbacks=callbacks
        )
        
        best_iter = model.best_iteration_
        logger.info(f"Model for alpha {alpha} stopped at iteration {best_iter}.")
        models[alpha] = model

    # ── 3. The Inference Phase (January 2026 Grid) ────────────────────────────
    logger.info("Building synthetic Future Grid for January 2026...")
    
    target_year = config["project"]["target_year"]    # 2026
    target_month = config["project"]["target_month"]  # 1
    
    future_grid = outlet_static[["Outlet_ID", "Distributor_ID", "Outlet_Type", "Dynamic_Tier"] + poi_cols + catchment_cols].copy()
    future_grid["Year"] = target_year
    future_grid["Month"] = target_month
    
    # Apply Temporal Triggers for Jan 2026
    season_df = read_parquet(paths["silver"]["distributor_seasonality"])
    future_season = season_df[(season_df["Year"] == target_year) & (season_df["Month"] == target_month)]
    future_grid = future_grid.merge(future_season[["Distributor_ID", "Seasonality_Index"]], on="Distributor_ID", how="left")
    future_grid["Seasonality_Index"] = future_grid["Seasonality_Index"].fillna("Moderate")
    
    holiday_df = read_parquet(paths["silver"]["holiday_list"])
    holiday_df["Year"] = pd.to_datetime(holiday_df["Date"]).dt.year
    holiday_df["Month"] = pd.to_datetime(holiday_df["Date"]).dt.month
    jan_holidays = holiday_df[(holiday_df["Year"] == target_year) & (holiday_df["Month"] == target_month)]
    
    future_grid["Holiday_Count"] = len(jan_holidays)
    future_grid["Number_of_Weekends"] = get_weekend_days(target_year, target_month)
    future_grid["Is_Cultural_Month"] = 0  # Jan is not March/April
    future_grid["Is_High_Season"] = (future_grid["Seasonality_Index"] == "Favorable").astype(int)
    
    future_grid["Tuition_Weekend_Surge"] = future_grid["Has_Youth_Catchment"] * future_grid["Number_of_Weekends"]
    future_grid["Tourist_Peak_Multiplier"] = future_grid["Has_Leisure_Catchment"] * future_grid["Is_High_Season"]
    future_grid["Sports_Big_Match_Spike"] = future_grid["Has_Athletic_Catchment"] * (future_grid["Is_Cultural_Month"] * 1.5 + future_grid["Number_of_Weekends"])
    future_grid["Park_Poya_Outing"] = future_grid["Has_Leisure_Catchment"] * future_grid["Holiday_Count"]
    
    X_test = future_grid[features].copy()
    for c in cat_features:
        X_test[c] = X_test[c].astype('category')
        
    logger.info("Predicting Latent Demand Surface for January 2026...")
    
    # Predict all quantiles
    for alpha in alphas:
        col_name = f"Predicted_Volume_p{int(alpha*100)}"
        future_grid[col_name] = models[alpha].predict(X_test).round(2)
        
    # We removed the Historical Safety Floor to prevent upward statistical bias.
    # The models' predictions are allowed to fall below historical max if temporal features dictate.
    
    # ── 4. The Final Output ───────────────────────────────────────────────────
    final_cols = ["Outlet_ID", "Predicted_Volume_p50", "Predicted_Volume_p75", "Predicted_Volume_p90"]
    final_output = future_grid[final_cols].copy()
    
    out_path = resolve_path(paths["output"]["predictions"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_output.to_csv(out_path, index=False)
    
    logger.info("=" * 60)
    logger.info("🚀 PHASE 5 COMPLETE!")
    logger.info(f"Total projected network demand for Jan 2026 (p75): {final_output['Predicted_Volume_p75'].sum():,.2f} Liters")
    logger.info(f"Final predictions saved strictly to: {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_lgbm_model()
