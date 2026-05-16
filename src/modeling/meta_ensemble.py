"""Modeling — Stage 5 & 6: Meta Ensemble and Business Guardrails."""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from src.utils.logger import get_logger

logger = get_logger("modeling.meta_ensemble")

def blend_predictions(df_val: pd.DataFrame, preds_val: pd.DataFrame, 
                      df_test: pd.DataFrame, preds_test: pd.DataFrame) -> np.ndarray:
    """
    Stage 5: Train a Ridge Regression meta-blender on validation predictions,
    and apply it to the test predictions.
    
    preds_val and preds_test should contain columns: 
    ['pred_aft', 'pred_quantile', 'pred_peer']
    """
    logger.info("Training Meta-Blender (Ridge Regression) on validation predictions (Stage 5)...")
    
    X_val = preds_val[['pred_aft', 'pred_quantile', 'pred_peer']].values
    
    # Target is observed volume in the validation set.
    # We use sample weights to heavily prioritize the blender learning from 
    # unconstrained records, where observed volume is closest to true latent demand.
    y_val = df_val['total_volume'].values
    weights = np.clip(1.0 - df_val['censor_probability'].fillna(0), 0.1, 1.0)
    
    # Fit Ridge Regression to find the optimal combination weights
    blender = Ridge(alpha=1.0, fit_intercept=True)
    blender.fit(X_val, y_val, sample_weight=weights)
    
    logger.info(f"Fitted Meta-Blender Weights -> AFT: {blender.coef_[0]:.3f}, Quantile: {blender.coef_[1]:.3f}, Peer: {blender.coef_[2]:.3f}")
    
    # Predict on the target test grid
    X_test = preds_test[['pred_aft', 'pred_quantile', 'pred_peer']].values
    final_preds = blender.predict(X_test)
    
    # Prevent negative predictions
    final_preds = np.maximum(final_preds, 0)
    
    return final_preds

def apply_business_guardrails(df_test: pd.DataFrame, raw_predictions: np.ndarray, config: dict) -> np.ndarray:
    """
    Stage 6: Apply Business Guardrails (Stable Floor and Extreme Sanity Ceiling).
    """
    logger.info("Applying Business Guardrails (Stage 6)...")
    
    # 1. STABLE FLOOR: prediction >= robust historical upper bound
    # We use historical 95th percentile adjusted by target month seasonality
    hist_p95 = df_test["historical_p95"].fillna(0)
    
    # Map seasonal score (-1, 0, 1) to a +/- 10% multiplier
    seas_adj = 1.0 + (df_test["distributor_seasonal_score"].fillna(0) * 0.10)
    
    # Add a nominal 5% annual market growth assumption
    market_growth = 1.05
    
    stable_floor = hist_p95 * seas_adj * market_growth
    
    # 2. SANITY CEILING: prediction <= extreme peer ceiling
    # Prevents runaway AFT extrapolations by capping at 1.5x the Peer 95th percentile benchmark
    if "pred_peer" in df_test.columns:
        sanity_ceiling = df_test["pred_peer"] * 1.5
    else:
        sanity_ceiling = np.inf
    
    # Apply Clamp Logic
    clamped_preds = np.maximum(raw_predictions, stable_floor)
    
    # Only apply ceiling if it's strictly greater than the floor to prevent inversions
    valid_ceiling_mask = sanity_ceiling > stable_floor
    clamped_preds[valid_ceiling_mask] = np.minimum(clamped_preds[valid_ceiling_mask], sanity_ceiling[valid_ceiling_mask])
    
    num_floored = np.sum(clamped_preds > raw_predictions)
    num_ceilinged = np.sum(clamped_preds < raw_predictions)
    
    logger.info(f"Guardrails applied: Raised {num_floored:,} predictions (Hit Floor). Capped {num_ceilinged:,} predictions (Hit Ceiling).")
    
    return clamped_preds
