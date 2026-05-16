# 📋 Development Log — InsightAI

> Step-by-step record of what was built, when, and current status.
> Updated as we progress through each phase.

---

## Phase 1: Scaffolding ✅
**Date:** 2026-05-15
**Status:** Complete

### What was done:
1. **Initialized Git repo** with remote `origin` → `github.com/Oxshadha/Data-Storm-7.0-InsightAI`
2. **Created full Lakehouse directory structure:**
   - `data/bronze/` → Raw ingestion (zero transforms)
   - `data/silver/` + `data/silver/rejected/` → Cleaned data + quarantine store
   - `data/gold/` → Feature-engineered, model-ready data
3. **Implemented core utilities (fully working):**
   - `src/utils/config.py` — Config loader from `pipeline_config.yaml`
   - `src/utils/logger.py` — Structured logging (console + file)
   - `src/utils/io.py` — Parquet/CSV read/write with auto-logging
4. **Implemented DQ framework (fully working):**
   - `src/silver/dq_checks.py` — 7 reusable, parameterizable check functions:
     - `check_duplicates()` — Composite key duplicate detection
     - `check_nulls()` — Mandatory field null detection
     - `check_referential_integrity()` — Foreign key validation
     - `check_value_range()` — Min/max boundary check
     - `check_format()` — Regex pattern validation
     - `check_negative_volumes()` — Return/reversal tagging
     - `check_zero_volumes()` — System adjustment tagging
   - `src/silver/quarantine.py` — `QuarantineManager` class (collects, stores, manifests)
5. **Implemented Bronze ingestion (fully working):**
   - `src/bronze/ingest_internal.py` — Reads all 5 CSVs → Parquet as-is
6. **Created pipeline orchestrator:**
   - `run_pipeline.py` — CLI with `--stage bronze|silver|gold|predict`
7. **Created config + tracking files:**
   - `config/pipeline_config.yaml` — All paths, thresholds, parameters
   - `requirements.txt` — Python dependencies
   - `ai_log/genai_transparency_log.md` — GenAI usage tracker
   - `experiments/experiment_log.md` — Model experiment tracker
   - `README.md` — Full architecture documentation

### What's NOT done (stubs only):
- `src/bronze/ingest_poi.py` — POI scraping (Overpass API)
- `src/silver/clean_*.py` — All 6 dataset cleaning scripts
- `src/gold/feature_*.py` — All feature engineering scripts
- `src/modeling/*.py` — Censoring detection + demand estimation
- `notebooks/` — No notebooks created yet

---

## Phase 2: Bronze Ingestion ✅
**Date:** 2026-05-16
**Status:** Complete

### What was done:
- [x] `src/bronze/ingest_internal.py` — **Fully implemented** Bronze ingestion:
  - Reads CSVs from `data/bronze/`
  - Converts to Parquet with category dtypes for memory optimization (71% memory reduction)
  - Locks down schema (numeric types enforced)
- [x] `src/bronze/ingest_poi.py` — **Overture Maps Integration** (Pivot from Overpass API):
  - Uses `duckdb` to query the official Overture Maps cloud Parquet files (AWS S3) directly.
  - Pulls all POIs (education, transportation, commercial, tourism, etc.) for the bounding box of Sri Lanka in under 2 minutes (instead of 7 hours).
  - Bounding box constraints used directly in the SQL WHERE clause.
  - Saves raw results as Parquet in `data/bronze/poi_raw.parquet`.

### Still TODO:
- [ ] Run `ingest_poi.py` to extract Overture Maps data.

---

## Phase 3: Silver — Forensic Cleaning ✅
**Date:** 2026-05-16
**Status:** Complete

### What was done:
- [x] Completely rewrote `dq_checks.py` to use vectorized pandas operations instead of loops.
- [x] Completely rewrote `quarantine.py` to save full original rows plus the failure reason.
- [x] `clean_transactions.py` — Handled System Ghosts (negatives, zeros, dupes, outliers, lazy reps). Negative returns are tagged but left in the clean data to aggregate correctly. Lazy Reps are tagged based on a volume-to-SKU threshold.
- [x] `clean_outlet_master.py` — Fixed typos, case, nulls, and engineered `Dynamic_Tier` using 6-month average volumes from transactions to handle Master Data Decay.
- [x] `clean_coordinates.py` — Geo-bounds validation, co-location detection.
- [x] `clean_seasonality.py` — Validated seasonality index values.
- [x] `clean_holidays.py` — Parsed ISO dates, deduplicated.
- [x] `clean_poi.py` — Standardized POI data.
- [x] Successfully ran the Silver pipeline. Quarantined records properly separated from clean data.

---

## Phase 4: Gold — Feature Engineering ✅
**Date:** 2026-05-16
**Status:** Complete

### What was done:
- [x] **POI Catchment Features**: Mapped Overture POIs into custom catchments (Youth/Education, Leisure, Athletic) using a 1km GeoPandas buffer around each outlet.
- [x] **Temporal Triggers**: Extracted `Number_of_Weekends`, `Is_Cultural_Month`, `Holiday_Count`, and `Seasonality_Index` per month.
- [x] **Spatio-Temporal Interactions**: 
  - `Tuition_Weekend_Surge`: (Youth POIs × Weekends)
  - `Tourist_Peak_Multiplier`: (Leisure POIs × High Season)
  - `Sports_Big_Match_Spike`: (Athletic POIs × (Cultural Month + Weekends))
  - `Park_Poya_Outing`: (Leisure POIs × Holidays)
- [x] **Censoring Signal Detection**: Flagged `Is_Censored` for outlets with zero volume variance but high spatial demand.
- [x] **ABT Generation**: Joined everything into `data/gold/model_input.parquet` (450,633 rows × 38 features).

---

## Phase 5: Predictive Modeling (Demand Estimation) ✅
**Date:** 2026-05-16
**Status:** Complete

### What was done:
- [x] **The Spatial Heuristic (K-Means Baseline)**:
  - Standardized the 38-dimensional feature space.
  - Ran K-Means clustering ($K=50$) to group outlets into behavioral peer groups based on spatio-temporal features.
  - Calculated the 90th percentile `Total_Volume` for the *uncensored* shops in each cluster.
  - Projected this ceiling onto the *censored* shops (`Predicted = max(Actual, Cluster_P90)`).
  - Saved baseline predictions to `model_baseline_output.parquet`.
- [x] **Quantile Gradient Boosting (LightGBM)**:
  - Trained `LGBMRegressor(objective='quantile', alpha=0.90)` strictly on *uncensored* historical data.
  - Generated a synthetic future grid for January 2026, incorporating specific temporal triggers (10 weekend days, specific Jan holidays, seasonality).
  - Passed the future grid into the trained model to predict the 90th percentile maximum potential.
  - Applied the historical maximum safety floor (`max(Prediction, Historical_Max)`).
  - Generated the final deliverable `output/insightai_predictions.csv`.

---

## Phase 6: Deliverables 🔴
**Date:** —
**Status:** Not started

### TODO:
- [ ] Final predictions CSV
- [ ] 5-page PDF report
- [ ] Final README review
- [ ] GenAI log complete
