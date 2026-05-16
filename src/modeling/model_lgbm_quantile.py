"""Modeling — Model B: Quantile Boosting (Upper Envelope Estimation)."""

import pandas as pd
import numpy as np
import lightgbm as lgb
from src.utils.logger import get_logger

logger = get_logger("modeling.model_lgbm_quantile")

def train_predict_lgbm_quantile(df_train: pd.DataFrame, df_test: pd.DataFrame, features: list, config: dict) -> np.ndarray:
    """Train LightGBM Quantile Regressor and predict upper demand envelope."""
    logger.info("Training LightGBM Quantile Model (Stage 4 - Model B)...")
    
    # Target is exactly the observed historical volume
    X_train = df_train[features].copy()
    y_train = df_train["total_volume"]
    
    # Identify categorical features for LightGBM
    cat_features = [c for c in features if df_train[c].dtype == 'object' or 'Type' in c or 'Cluster' in c]
    for c in cat_features:
        X_train[c] = X_train[c].astype('category')
        if c in df_test.columns:
            df_test[c] = df_test[c].astype('category')
            
    # Weighted Training Implementation
    # Likely constrained/censored rows get lower weight so they don't drag down the upper envelope.
    # Likely unconstrained rows get higher weight because they represent true demand distribution.
    # We invert censor_probability. Max weight 1.0, min weight 0.2.
    censor_prob = df_train["censor_probability"].fillna(0)
    sample_weights = np.clip(1.0 - censor_prob, 0.2, 1.0)
    
    # Model configuration for the 95th Percentile Envelope
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=0.95,  # 95th percentile
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_weight=5,
        subsample=0.8,
        random_state=config.get("modeling", {}).get("random_seed", 42),
        n_jobs=-1,
        verbose=-1
    )
    
    logger.info(f"Fitting LGBM Quantile Regressor (alpha=0.95) using weighted training...")
    model.fit(
        X_train, y_train, 
        sample_weight=sample_weights,
        categorical_feature=cat_features
    )
    
    logger.info("Predicting with LGBM Quantile Regressor...")
    preds = model.predict(df_test[features])
    
    return preds
