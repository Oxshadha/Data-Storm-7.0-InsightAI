# Technical Architecture Deep Dive: Latent Demand Model

This document serves as an internal technical guide to explain *exactly* how the InsightAI Latent Demand model was built, the specific algorithms used, and the mathematical reasoning behind each choice. Use this to deeply understand the codebase, prepare for Q&A, or present the technical architecture.

---

## 1. The Core Problem: Right-Censored Data
The fundamental challenge of Data Storm 7.0 is that the target variable (historical volume) is **right-censored**. This means for many outlets, the physical sales they recorded were artificially capped by stockouts, broken coolers, or credit limits. Their *true latent demand* is higher than what is observed in the data.

Standard Machine Learning models (like Random Forests or Neural Networks) trained to minimize Mean Squared Error (MSE) will fail because they learn to predict the *capped* volume, not the true potential.

To solve this, we built a highly specialized **Probability-Weighted Tri-Model Ensemble**.

---

## 2. Unsupervised Censoring Detection: KMeans Clustering
Before we can predict demand, we need to mathematically identify *which* outlets were artificially capped. We did this using a hybrid Unsupervised Learning engine.

**Algorithm Used:** `sklearn.cluster.KMeans` + Rule-based Heuristics
**Location in Codebase:** `src/modeling/censoring_detection.py`

**How it works:**
1.  **Feature Extraction:** We extract behavioral signals for each outlet:
    *   *Zero-Sales Fraction:* Percentage of weeks with exactly 0 sales (indicates stockouts).
    *   *Demand Volatility:* Coefficient of Variation. High volatility usually indicates irregular supply, not irregular demand.
    *   *Capacity Utilization:* How much volume they push per square foot of cooler space.
2.  **KMeans Clustering:** We pass these features into a K-Means algorithm to group the outlets into "High Constraint", "Medium Constraint", and "Low Constraint" clusters.
3.  **Continuous Probability:** The distance to the cluster centers is converted into a continuous `censor_probability` score ranging from $0.0$ to $1.0$.
    *   *0.0:* Outlet is completely unconstrained (historical sales = true demand).
    *   *1.0:* Outlet is heavily constrained (true demand is much higher than historical sales).

---

## 3. The Geospatial Gravity Model
To understand true demand, we must understand foot traffic. Instead of just counting POIs (Points of Interest), we built a "Gravity Model" backed by real-world OpenStreetMap data.

**Algorithm Used:** `scipy.spatial.cKDTree` & `OpenStreetMap Overpass API`
**Location in Codebase:** `src/bronze/ingest_poi.py` & `src/gold/feature_poi.py`

**How it works:**
1.  **Data Extraction (Bronze Layer):** We dynamically calculate a master bounding box that covers all our outlet GPS coordinates (Sri Lanka). Using this bounding box, we send a `POST` request to the **OpenStreetMap (OSM) Overpass API**, safely extracting nodes, ways, and relations for commercial centers, transit hubs, and educational institutions.
2.  **Distance Calculation (Gold Layer):** We use a KD-Tree (`scipy.spatial.cKDTree`) to rapidly calculate the exact distance between every outlet and every extracted OSM POI.
3.  **Inverse-Distance Decay:** We apply a mathematical decay function:
    $Gravity Score = \sum (1 / Distance^2)$

An outlet 100 meters from a train station gets a massive gravity score, while an outlet 5 kilometers away gets almost zero. This allows the model to map latent pedestrian potential independently of historical sales.

---

## 4. The Tri-Model Predictive Architecture
With our features and censorship probabilities ready, we feed the data into three separate algorithms, each designed to view the demand problem from a different mathematical angle.

### Model A: LightGBM Quantile Regressor
**Algorithm:** `lightgbm.LGBMRegressor(objective='quantile', alpha=0.95)`
*   **The Goal:** Predict the 95th percentile upper envelope of demand.
*   **The Secret Weapon:** We pass the KMeans `censor_probability` into LightGBM as **Sample Weights**. Outlets that were *unconstrained* receive heavy weights, while constrained outlets are down-weighted. This forces LightGBM to learn what the true upper limit of demand looks like based only on healthy, unconstrained stores.

### Model B: XGBoost Accelerated Failure Time (AFT)
**Algorithm:** `xgboost.XGBRegressor(objective='survival:aft')`
*   **The Goal:** Explicitly model the right-censorship.
*   **How it works:** Survival models are typically used in medicine to predict how long a patient will live. We adapted this for FMCG demand. We tell XGBoost: *"For outlets with a high censor_probability, the true demand is an unknown number between their observed volume and infinity."* The AFT model mathematically extrapolates unbounded latent potential.

### Model C: Feature-Space Peer Benchmark
**Algorithm:** Deterministic Pandas Grouping
*   **The Goal:** Establish a safe, reality-based business anchor.
*   **How it works:** For every outlet, we find its exact "twins" in the network (e.g., Small Groceries in Territory X with exactly 1 cooler). We calculate the 95th percentile volume achieved by those twins. If an outlet is severely constrained, its true potential should roughly mirror the top performance of its exact structural peers.

---

## 5. The Meta-Blender & Guardrails (Stage 5 & 6)
We now have three predictions. We must intelligently blend them.

**Algorithm Used:** `sklearn.linear_model.Ridge`
**Location in Codebase:** `src/modeling/meta_ensemble.py`

**How it works:**
We train a Ridge Regression Meta-Blender using Out-of-Fold (OOF) cross-validation. The blender compares the three models against historical sales and assigns weights.

*   **The Paradox:** The blender assigned **92.7%** weight to LightGBM and **0%** to XGBoost AFT. Why? Because the blender validates against *historical capped sales*. Since the AFT model correctly predicts massive uncapped numbers, the blender penalized it for "over-predicting" reality. LightGBM, which smoothly hugs the 95th percentile, won the blender's favor.
*   *This proves our LightGBM model successfully mapped the unconstrained upper bounds without breaking mathematical validation rules.*

**The Business Guardrails:**
Finally, no ML model goes to production without safety checks:
1.  **Stable Floor:** The prediction cannot drop below the historical 95th percentile (adjusted for Jan seasonality + 5% market growth).
2.  **Sanity Ceiling:** We clamped 143 extreme outlet predictions by forcing the ML output to never exceed **1.5x the Peer Benchmark**. This prevents the model from predicting millions of liters for a tiny grocery store.
