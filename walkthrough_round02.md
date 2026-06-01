# Data Storm 7.0 — Round 02 Technical Walkthrough

> **Team:** InsightAI  
> **Phase:** Final Round (Round 02)  
> **Date:** May 2026

---

## 1. Dataset Verification

Before beginning Round 2 development, we verified that the competition datasets are **identical** to Round 1:

| Dataset | Rows | Columns | Status |
|---------|------|---------|--------|
| `transactions_history_final.csv` | 2,376,389 | 7 | ✅ Identical (byte-for-byte) |
| `outlet_master.csv` | 20,000 | 4 | ✅ Identical (196 nulls preserved) |
| `outlet_coordinates.csv` | 20,000 | 3 | ✅ Identical |
| `distributor_seasonality_details.csv` | 360 | 4 | ✅ Identical |
| `holiday_list.csv` | 349 | 3 | ✅ Identical |

**Conclusion:** Round 2 uses the same raw data. The challenge is building more advanced features and new deliverables on top of the existing pipeline.

---

## 2. Pipeline Foundation (Carried from Round 1)

Our Round 1 Medallion Lakehouse pipeline remains intact and unchanged:

- **Bronze:** CSV → Parquet ingestion (zero transformations)
- **Silver:** Forensic cleaning with parameterized DQ framework + quarantine store
  - Ghost exorcism (zero-volume, duplicate retries)
  - Negative return tagging (nets out in aggregation)
  - Lazy Rep detection (low SKU diversity + high volume)
  - Dynamic Tiering (quantile-based reclassification replacing stale Outlet_Size)
  - Typo corrections, format/range/referential integrity checks
- **Gold → Model:** Collaborative filtering, POI features, K-Means clustering, LightGBM p90 quantile regression

All Silver layer code, quarantine system, and config are unchanged from Round 1.

---

## 3. Round 2 Upgrade: Spatial Intelligence Engine V2

### 3.1 Problem Statement

Round 2 explicitly requires:
- **Spatial Distance-Decay Modeling:** "Simply counting nearby POIs within a fixed radius is not enough. You should apply non-linear methods such as distance-decay functions (Gravity Models, Gaussian decay, or exponential decay)."
- **Competitive Catchment Density:** "Estimate how crowded or isolated a store is in its local market."

### 3.2 Our Approach

We completely rewrote `src/gold/feature_poi.py` from a GeoPandas `sjoin` flat-buffer counter into an enterprise-grade **Spatial Intelligence Engine** using vectorized KD-Tree queries.

#### Architecture Decision: cKDTree over GeoPandas sjoin

| Criteria | R1 (GeoPandas sjoin) | R2 (scipy cKDTree) |
|----------|---------------------|---------------------|
| Complexity | O(N × M) with spatial index overhead | O(N log M) via binary space partitioning |
| Output type | Integer count (flat) | Continuous gravity score (distance-weighted) |
| Speed (19,760 × 12,678) | ~15 seconds | **1.75 seconds** |
| Information loss | High (bus stop 20m away = bus stop 900m away) | None (each POI weighted by exact distance) |

#### Mathematical Framework

**Catchment Driver Gravity (Exponential Decay):**
```
W = e^(-λd)    where λ = 0.003, d = distance in meters
```
- A school 10m away: W ≈ 0.97 (nearly full influence)
- A school 230m away: W ≈ 0.50 (half influence)
- A school 800m away: W ≈ 0.09 (minimal influence)
- A school >1500m away: W = 0 (excluded by KD-Tree cutoff)

**Competitive Saturation (Huff Gravity Model):**
```
W = 1 / (d + 1)^β    where β = 2.0 (inverse-square law)
```
- Competitor 0m away: W = 1.0 (maximum cannibalization)
- Competitor 50m away: W ≈ 0.0004 (significant penalty)
- Competitor >500m away: W = 0 (excluded by radius)

Reference: Huff, D.L. (1964). "Defining and Estimating a Trading Area." *Journal of Marketing.*

#### Parameters (Config-Driven)

All spatial parameters are in `config/pipeline_config.yaml` under `spatial_engine:`:

```yaml
spatial_engine:
  catchment_radius_m: 1500      # Max POI search distance
  decay_lambda: 0.003           # Halves every ~230m (pedestrian catchment)
  competitor_radius_m: 500      # Competition is hyperlocal
  huff_beta: 2.0                # Inverse-square law
```

### 3.3 Features Engineered (33 per outlet)

| Feature Type | Examples | Count |
|---|---|---|
| Per-category gravity | `gravity_school`, `gravity_hospital`, `gravity_restaurant`... | 9 |
| Per-category flat count | `poi_count_school`, `poi_count_hospital`... (R1 backward compat) | 9 |
| Group gravity | `gravity_group_youth`, `gravity_group_health`, `gravity_group_leisure` | 5 |
| Competition | `competitive_saturation_index`, `comp_saturation_retail`, `comp_saturation_food` | 3 |
| Composite | `total_driver_gravity`, `latent_opportunity_ratio`, `is_isolated_goldmine` | 3 |
| Legacy compat | `poi_total_catchment`, `poi_driver_catchment`, `poi_cannibal_risk`, `competitor_count_flat` | 4 |

**Key composite feature — Latent Opportunity Ratio:**
```
latent_opportunity_ratio = total_driver_gravity / (competitive_saturation_index + 0.1)
```
High ratio = high footfall drivers + low competition = untapped goldmine for the budget optimizer.

### 3.4 ABT Builder Upgrade (build_model_input.py)

We added **V2 continuous gravity interactions** alongside the existing R1 binary ones. This gives the LightGBM model a much richer continuous signal:

| R1 Binary Interaction | R2 Continuous Interaction |
|---|---|
| `Tuition_Weekend_Surge` = Has_Youth (0/1) × Weekends | `Tuition_Weekend_Gravity` = gravity_group_youth (continuous) × Weekends |
| `Tourist_Peak_Multiplier` = Has_Leisure (0/1) × High_Season | `Tourist_Peak_Gravity` = gravity_group_leisure (continuous) × High_Season |
| `Sports_Big_Match_Spike` = Has_Athletic (0/1) × Cultural | `Sports_Match_Gravity` = gravity_group_athletic (continuous) × Cultural |
| `Health_Catchment_Spike` = Has_Health (0/1) × Weekends | `Health_Pulse_Gravity` = gravity_group_health (continuous) × Weekends |
| *(none)* | `Competition_Season_Pressure` = saturation × off-season *(new)* |

Censoring detection was also enhanced to use `total_driver_gravity > 0.5` as an additional signal alongside the R1 binary interaction score.

### 3.5 Verification Results

```
⏱️ Execution time:   1.75 seconds (19,760 shops × 12,678 POIs)
✅ Outlets with catchment signal:   9,499 / 19,760
✅ Isolated goldmines identified:   922
✅ Highly saturated markets (Q4):   828
✅ 33 spatial features engineered per outlet
✅ All R1 backward-compatible columns preserved (poi_count_*)
```

---

## 4. Files Changed in This Phase

| File | Change Type | Description |
|------|------------|-------------|
| `src/gold/feature_poi.py` | **Rewrite** | Flat-buffer → cKDTree + exponential decay + Huff gravity |
| `src/gold/build_model_input.py` | **Enhanced** | Added V2 gravity interactions + enhanced censoring |
| `config/pipeline_config.yaml` | **Extended** | Added `spatial_engine` config section |
| `ai_log/genai_transparency_log.md` | **Updated** | Logged Round 2 AI usage with prompts |
| `walkthrough_round02.md` | **New** | This document |

---

## 5. Remaining Round 2 Deliverables (TODO)

- [ ] Re-run full pipeline (Gold → Model) with new spatial features
- [ ] Marketing Spend Optimization (5M LKR, Western Province)
- [ ] Outlet Intelligence Web App (interactive dashboard)
- [ ] XAI Module with LLM-based explainability
- [ ] Expand Technical Report to 10 pages
- [ ] Executive Pitch Deck (10 slides)
