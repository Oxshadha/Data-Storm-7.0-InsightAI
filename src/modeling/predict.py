"""Modeling — Pipeline Orchestrator (Stages 4, 5, 6)."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from src.utils.logger import get_logger
from src.utils.io import read_parquet
from src.modeling.model_xgb_aft import train_predict_xgb_aft
from src.modeling.model_lgbm_quantile import train_predict_lgbm_quantile
from src.modeling.model_peer_benchmark import calculate_peer_benchmark
from src.modeling.meta_ensemble import blend_predictions, apply_business_guardrails

logger = get_logger("modeling.predict")

def generate_predictions(config: dict) -> None:
    logger.info("Executing Pipeline Stage 4-6: Predict (Meta-Ensemble)...")
    
    df_train = read_parquet(config["paths"]["gold"]["model_input"])
    
    # For this hackathon, our test grid is structurally identical to the latest state of the outlets
    df_test = df_train.copy()
    
    # Convert object columns to category type for ML models
    for c in df_train.columns:
        if df_train[c].dtype == 'object':
            df_train[c] = df_train[c].astype('category')
            df_test[c] = df_test[c].astype('category')
            
    # Features to use (Exclude IDs, targets, and our synthetic probability column)
    exclude_cols = ["Outlet_ID", "total_volume", "total_bill", "censor_probability"]
    features = [c for c in df_train.columns if c not in exclude_cols]
    
    # ---------------------------------------------------------
    # 1. Generate Full Test Predictions
    # ---------------------------------------------------------
    logger.info("Generating Full Test Predictions...")
    preds_test = pd.DataFrame(index=df_test.index)
    preds_test["pred_aft"] = train_predict_xgb_aft(df_train, df_test, features, config)
    preds_test["pred_quantile"] = train_predict_lgbm_quantile(df_train, df_test, features, config)
    preds_test["pred_peer"] = calculate_peer_benchmark(df_train, df_test)
    
    # Add peer benchmark back to df_test for guardrails
    df_test["pred_peer"] = preds_test["pred_peer"]
    
    # ---------------------------------------------------------
    # 2. Generate Out-Of-Fold (OOF) Validation Predictions
    # ---------------------------------------------------------
    logger.info("Generating OOF predictions for Meta-Blender training...")
    kf = KFold(n_splits=5, shuffle=True, random_state=config.get("modeling", {}).get("random_seed", 42))
    preds_val = pd.DataFrame(index=df_train.index)
    preds_val["pred_aft"] = 0.0
    preds_val["pred_quantile"] = 0.0
    
    fold_num = 1
    for train_idx, val_idx in kf.split(df_train):
        logger.info(f"OOF Fold {fold_num}/5...")
        t_fold, v_fold = df_train.iloc[train_idx].copy(), df_train.iloc[val_idx].copy()
        
        preds_val.loc[val_idx, "pred_aft"] = train_predict_xgb_aft(t_fold, v_fold, features, config)
        preds_val.loc[val_idx, "pred_quantile"] = train_predict_lgbm_quantile(t_fold, v_fold, features, config)
        fold_num += 1
        
    preds_val["pred_peer"] = calculate_peer_benchmark(df_train, df_train)
    
    # ---------------------------------------------------------
    # 3. Blend & Guardrails
    # ---------------------------------------------------------
    # Stage 5: Blend
    final_raw_preds = blend_predictions(df_train, preds_val, df_test, preds_test)
    
    # Stage 6: Business Guardrails
    final_clamped_preds = apply_business_guardrails(df_test, final_raw_preds, config)
    
    df_test["Maximum_Monthly_Liters"] = np.round(final_clamped_preds, 2)
    final_output = df_test[["Outlet_ID", "Maximum_Monthly_Liters"]]
    
    # Export to the new distinct CSV requested
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "insightai_predictions.csv"
    
    final_output.to_csv(out_path, index=False)
    
    logger.info("=" * 60)
    logger.info("🚀 PHASE 6 COMPLETE: Final Model Ensembled")
    logger.info(f"Total projected network demand for Jan 2026: {final_output['Maximum_Monthly_Liters'].sum():,.2f} Liters")
    logger.info(f"Final predictions securely saved to: {out_path}")
    logger.info("=" * 60)
