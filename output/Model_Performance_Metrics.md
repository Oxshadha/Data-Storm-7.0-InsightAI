# InsightAI Model Performance & Training Metrics

## 1. Training Environment and Data Shape
The model was trained on the fully integrated Analytics Base Table (ABT) combining transaction history, spatial POI features, and seasonal data.

- **Total Dataset Size:** 450,633 records × 130 columns
- **Training Set (Uncensored):** 428,509 records
- **Validation Set (Censored/Out-of-Time):** 12,314 records

## 2. Quantile Regression Configuration
To properly model **Right-Censored Latent Demand** (outlets that hit their supply ceiling), the system trained a sequence of LightGBM Quantile Regressors using Pinball Loss to estimate peak capability rather than median tendency.

- **$\alpha = 0.50$ (Median):** Stopped at iteration 300.
- **$\alpha = 0.75$ (75th Percentile):** Stopped at iteration 300.
- **$\alpha = 0.90$ (90th Percentile Peak Capacity):** Stopped at iteration 300.

*The $p90$ Quantile Regressor was selected as the final production model to drive the MILP optimizer, representing a 90% statistical confidence ceiling on true unconstrained volume potential.*

## 3. Network Projection
- **Total Projected Network Demand (Jan 2026):** 12,755,448.08 Liters
- **Objective Output:** Peak Volume Potential ($G_i$) generated for all Candidate Outlets.

## 4. Top Feature Importances (Gain)
The LightGBM model isolated the most critical features driving latent volume capability. Spatial and contextual drivers heavily outweighed raw historical constraints.

| Rank | Feature Name | Gain Importance | Feature Group |
|---|---|---|---|
| 1 | `poi_driver_catchment` | 5394.32 | Spatial Gravity |
| 2 | `poi_total_catchment` | 4247.99 | Spatial Gravity |
| 3 | `Outlet_Type` | 3783.67 | Master Data |
| 4 | `Is_High_Season` | 3704.61 | Temporal |
| 5 | `poi_count_hotel` | 2946.92 | Spatial Gravity |
| 6 | `poi_count_bakery` | 2600.10 | Spatial Gravity |
| 7 | `poi_count_restaurant` | 2262.46 | Spatial Gravity |
| 8 | `poi_count_buddhist_temple` | 2122.09 | Spatial Gravity |
| 9 | `poi_cannibal_risk` | 2003.11 | Spatial Gravity (Penalty) |

*Observation: `poi_driver_catchment` and `poi_cannibal_risk` being in the top 10 validates the core hypothesis of the project: spatial context and competitive isolation are the strongest predictors of unconstrained volume lift.*
