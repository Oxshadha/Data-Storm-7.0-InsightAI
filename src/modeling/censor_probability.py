"""Modeling — Stage 3: Censor Probability Engine."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from src.utils.logger import get_logger

logger = get_logger("modeling.censor_probability")

def calculate_censor_probability(df: pd.DataFrame) -> pd.Series:
    """
    Calculate a continuous censor_probability in [0, 1] using a hybrid heuristic + clustering validation.
    """
    logger.info("Calculating Censor Probability (Stage 3)...")
    
    # 1. Base Plateau Component (0 to 1)
    plateau = df["plateau_score"].fillna(0)
    
    # 2. Operational Anomalies
    ops_score = df["stockout_proxy_score"].fillna(0) + df["high_return_anomaly"].fillna(0) * 0.2
    
    # 3. Structural Mismatches
    mismatch = (df["high_poi_low_sales_mismatch"].fillna(0) + df["low_cooler_high_peer_mismatch"].fillna(0)) * 0.3
    
    # 4. Sudden Drops
    drop = df["sudden_sales_drop"].fillna(0) * 0.2
    
    # Composite Heuristic Score
    raw_score = plateau + ops_score + mismatch + drop
    
    # 5. Clustering Validation
    # We run a quick KMeans on the constraint features to find the "highly constrained" blob.
    # If an outlet falls into the highly constrained cluster, we boost its probability.
    cluster_features = ["plateau_score", "stockout_proxy_score", "low_cooler_high_peer_mismatch", "volatility_cv"]
    X = df[cluster_features].fillna(df[cluster_features].median())
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Identify which cluster represents "censored" (the one with higher average plateau score)
    centers = kmeans.cluster_centers_
    plateau_idx = cluster_features.index("plateau_score")
    censored_cluster = 0 if centers[0, plateau_idx] > centers[1, plateau_idx] else 1
    
    is_in_censored_cluster = (clusters == censored_cluster).astype(float)
    
    # Final Blend: 70% Heuristic, 30% Clustering validation
    scaler = MinMaxScaler()
    normalized_heuristic = scaler.fit_transform(raw_score.values.reshape(-1, 1)).flatten()
    
    prob = (normalized_heuristic * 0.7) + (is_in_censored_cluster * 0.3)
    
    # Ensure bounds
    prob = np.clip(prob, 0, 1)
    
    logger.info(f"Censor probability generated. Mean: {prob.mean():.3f}, Max: {prob.max():.3f}")
    return pd.Series(prob, index=df.index)
