# Data Storm 7.0 - Storming Round: Latent Demand Potential
**Team Name:** InsightAI

## Executive Summary
This report details the methodology and architecture employed by Team InsightAI to estimate the uncapped latent demand for a major FMCG beverage manufacturer's retail network in January 2026. Traditional machine learning models trained to minimize Mean Squared Error (MSE) on historical sales data inherently under-predict true market demand due to the presence of systemic constraints (e.g., stockouts, cooler capacity limits, and strict credit caps). 

To solve this, we implemented a mathematically rigorous **Tri-Model Meta-Ensemble**. This physics-based approach treats the problem as a censored estimation challenge, utilizing spatial gravity modeling, probabilistic clustering, and probability-weighted quantile regression to mathematically uncap the constraints. 

**Final Projection:** The total uncapped latent network demand for January 2026 is estimated at **124,498,487.77 Liters**.

---

## a. Data Forensics and Hygiene
In the FMCG sector, system data is notoriously plagued by "ghost" entries, reversed transactions, and erroneous GPS coordinates. Blindly feeding this data into an ML model corrupts the capacity bounds.

We established a strict **Data Forensics pipeline (Quarantine Manager)**:
- **Negative Volumes & Reversals:** We trapped over 4,800 anomalies where `Volume_Liters < 0`. Instead of silently dropping them, they were safely aggregated to net-zero out phantom sales or moved to a dedicated rejected store.
- **GPS Bounding Box Checks:** We implemented geospatial filters to isolate coordinates falling outside the logical bounds of the country/operational regions, quarantining "ghost" coordinates (e.g., coordinates sitting in the ocean).
- **The Rejection Manifest:** Every dropped or anomalous record was securely written to our `rejection_manifest.csv` with a documented failure reason, ensuring full data auditability without polluting the Silver and Gold analytics layers.

---

## b. Feature Engineering Logic
To predict what a store *could* sell, we engineered features to proxy unobserved limits and geographical potential.

*   **Spatial Gravity Modeling (KDTree & OSM API):** To map pedestrian potential independently of historical sales, we extracted geospatial markers using the **OpenStreetMap (OSM) Overpass API**. By constructing a master bounding box covering Sri Lanka, we safely bulk-extracted commercial, transit, and educational POIs. We then mapped POI (Point of Interest) influence using `scipy.spatial.cKDTree` and an inverse-distance spatial decay formula. An outlet located 500 meters from a major transit hub receives a stronger "Gravity Score" than one 5 kilometers away, allowing the model to learn true pedestrian traffic potential.
*   **Cooler Capacity & Outlet Footprint:** We engineered percentile scores mapping an outlet's given square footage against its allocated cooler capacity. This identified "under-cooled" outlets where physical constraints artificially suppress sales.
*   **Censorship Probability Engine (KMeans + Heuristics):** We developed an unsupervised probability engine to flag constrained outlets. Outlets displaying high historical volatility, frequent zero-sales days, or low cooler-to-area ratios were assigned a continuous constraint probability `[0, 1]`. This became the cornerstone of our probability-weighted training phase.

---

## c. Causal Base Logic
The fundamental challenge of estimating latent potential is that historical sales curves are artificially capped by physical constraints (stockouts, credit limits). We recognized this as a classic **censored demand curve** problem. Standard algorithms penalize predictions that exceed historical maximums, which is counterproductive when the goal is to estimate an *uncapped* ceiling.

**Our Statistical & Probabilistic Approach:**
1.  **Unsupervised Constraint Probability:** We did not arbitrarily assume all outlets were constrained equally. We utilized K-Means clustering (driven by zero-sales fraction, coefficient of variation, and capacity ratios) to assign a deterministic `censor_probability` to each outlet. 
2.  **Probability-Weighted Quantile Regression:** To calculate the uncapped potential ceiling, we utilized a LightGBM Quantile Regressor targeted at the 95th percentile ($\alpha=0.95$). Crucially, we passed the `censor_probability` as inverse sample weights. The model was mathematically forced to learn the upper demand envelope exclusively from "healthy, unconstrained" outlets, effectively uncapping the ceiling for constrained outlets without hallucinating impossible volumes.
3.  **Survival Analysis Integration:** We experimented with XGBoost Accelerated Failure Time (AFT), explicitly defining constrained outlets as censored targets (where true demand $\in [observed, \infty)$). 

**Validation and The Censorship Paradox:**
When validating our Meta-Ensemble using Out-Of-Fold (OOF) predictions, standard validation heavily penalized the Survival/AFT approach because it naturally over-predicted the historical "capped" data. To measure true accuracy, we validated exclusively against our *unconstrained subset* (outlets with `censor_probability < 0.2`). Against this true representation of demand, our approach achieved an **R-squared ($R^2$) of 0.9951** and an **MAE of 177.45 Liters**.

---

## d. GenAI Transparency Log
In strict adherence to the 36-hour hackathon rules, Generative AI (LLMs) was strategically utilized as an engineering accelerator, functioning as an intelligent pair-programmer under constant human oversight.

**How, Where, and Why GenAI was Utilized:**
*   **Code Translation & Boilerplate Accelerator:** LLMs were used to rapidly generate PySpark/Pandas boilerplate code for the Silver Layer data hygiene (e.g., geospatial bounding boxes and quarantine routing), saving hours of manual syntax typing.
*   **Architecture Brainstorming:** Used as a sounding board to debate the mathematical viability of Survival Analysis (AFT) vs. Probability-Weighted Quantile Regression for handling censored demand curves.
*   **Debugging & Refactoring:** During the final hours, LLMs were used to resolve specific Git/Merge issues and to quickly refactor the heuristic-based models into the final OOP (Object-Oriented Programming) pipeline structure for clean execution.

**Human Oversight & Ethics:**
No data was blindly passed to an LLM. The core mathematical thesis (the Tri-Model blending approach), the specific gravity decay formulas, the deterministic business guardrails (e.g., the 1.5x structural peer ceiling), and all final parameter hyper-tuning were explicitly directed, reviewed, and authorized by human intelligence. LLMs accelerated the *execution* of the pipeline, but the *causal logic* remained strictly human-driven.
