"""Modeling — Model A: XGBoost AFT (Latent Demand Estimation)."""

import pandas as pd
import numpy as np
import xgboost as xgb
from src.utils.logger import get_logger

logger = get_logger("modeling.model_xgb_aft")

def train_predict_xgb_aft(df_train: pd.DataFrame, df_test: pd.DataFrame, features: list, config: dict) -> np.ndarray:
    """Train XGBoost AFT survival model and predict latent potential."""
    logger.info("Training XGBoost AFT Model (Stage 4 - Model A)...")
    
    # 1. Target Setup (Calibrated Elasticity Bounds)
    # Elasticity is calibrated from historical volatility (cv_volume) rather than hardcoded.
    # We clip it between 5% and 30% to prevent wild boundary explosions.
    elasticity = 1.0 + np.clip(df_train["volatility_cv"].fillna(0), 0.05, 0.30)
    
    # Threshold for deciding strict bounds based on censor probability
    censor_thresh = 0.65
    
    y_lower = df_train["total_volume"].copy()
    y_upper = df_train["total_volume"].copy()
    
    # For highly constrained outlets, upper bound is infinity
    is_constrained = df_train["censor_probability"] > censor_thresh
    y_upper[is_constrained] = np.inf
    
    # For unconstrained outlets, upper bound allows for dynamic elasticity headroom
    is_unconstrained = ~is_constrained
    y_upper[is_unconstrained] = y_lower[is_unconstrained] * elasticity[is_unconstrained]
    
    # 2. XGBoost Setup
    dtrain = xgb.DMatrix(df_train[features], enable_categorical=True)
    dtrain.set_float_info('label_lower_bound', y_lower)
    dtrain.set_float_info('label_upper_bound', y_upper)
    
    params = {
        'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik',
        'aft_loss_distribution': 'normal',
        'aft_loss_distribution_scale': 1.20,
        'tree_method': 'hist',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'random_state': config.get("modeling", {}).get("random_seed", 42)
    }
    
    # 3. Fit
    logger.info(f"Fitting XGB AFT with {is_constrained.sum():,} censored and {is_unconstrained.sum():,} uncensored records.")
    bst = xgb.train(params, dtrain, num_boost_round=300)
    
    # 4. Predict
    logger.info("Predicting with XGB AFT...")
    dtest = xgb.DMatrix(df_test[features], enable_categorical=True)
    preds = bst.predict(dtest)
    
    return preds
