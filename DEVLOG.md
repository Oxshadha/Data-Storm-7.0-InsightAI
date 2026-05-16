# 📋 Development Log — InsightAI

> Step-by-step record of what was built, when, and current status.
> Updated as we progress through each phase.

---

## Phase 0: Data Setup & Bronze Validation ✅
**Date:** 2026-05-15
**Status:** Complete

### What was done:
1. **Confirmed current repo structure is preserved**
   - No folders were renamed or moved.
   - Work continues inside the existing Lakehouse layout.
2. **Confirmed raw data source**
   - `Refernce Resources/` remains the configured raw source in `config/pipeline_config.yaml`.
   - `transactions_history_final.csv` is present locally and ignored by Git because it exceeds GitHub's 100 MB limit.
3. **Copied required small CSVs into `data/bronze/` for local access**
   - `distributor_seasonality_details.csv`
   - `holiday_list.csv`
   - `outlet_coordinates.csv`
   - `outlet_master.csv`
4. **Executed Bronze ingestion**
   - Command: `python3 run_pipeline.py --stage bronze`
   - Result: completed successfully.
5. **Verified Bronze row counts**
   - `transactions_history.parquet`: 2,376,389 rows × 7 columns
   - `outlet_master.parquet`: 20,000 rows × 4 columns
   - `outlet_coordinates.parquet`: 20,000 rows × 3 columns
   - `distributor_seasonality.parquet`: 360 rows × 4 columns
   - `holiday_list.parquet`: 349 rows × 3 columns
6. **Updated project tracking files**
   - `experiments/experiment_log.md`
   - `ai_log/genai_transparency_log.md`
   - External planning file: `/Users/oneionei/MyProjects/DataStrom/doc/model_building_plan.md`

### Phase 0 checklist:
- [x] Confirm source files exist locally.
- [x] Confirm large raw transaction file is ignored by Git.
- [x] Preserve existing file structure.
- [x] Run Bronze ingestion.
- [x] Verify generated Bronze Parquet outputs.
- [x] Confirm logs are generated at `logs/pipeline.log`.
- [x] Record experiment notes.
- [x] Record GenAI usage.

### Next step:
- Start Phase 3 Silver implementation by wiring `run_pipeline.py --stage silver` to the existing cleaning modules.

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
**Date:** 2026-05-15
**Status:** Complete

### TODO:
- [x] Run `ingest_internal.py` to convert all CSVs → Bronze parquet
- [x] Implement POI scraping from OpenStreetMap (Overpass API)
- [x] Validate Bronze row counts match raw CSV exactly

---

## Phase 3: Silver — Forensic Cleaning ✅
**Date:** 2026-05-16
**Status:** Complete

### TODO:
- [x] `clean_transactions.py` — Handle System Ghosts (negatives, zeros, dupes, outliers)
- [x] `clean_outlet_master.py` — Fix typos, case, nulls
- [x] `clean_coordinates.py` — Geo-validation, co-location detection
- [x] `clean_seasonality.py` — Validate seasonality values
- [x] `clean_holidays.py` — Parse dates, deduplicate
- [x] `clean_poi.py` — Standardize POI data
- [x] Verify quarantine store is populated with documented reasons

---

## Phase 4: Gold — Feature Engineering ✅
**Date:** 2026-05-16
**Status:** Complete

### TODO:
- [x] Outlet profile features (size, type, coolers)
- [x] Transaction behavioral features (trends, variability, patterns)
- [x] POI density & catchment features
- [x] Seasonality & holiday features
- [x] Censoring signal detection (flat volumes, capacity constraints)
- [x] Join all into `model_input.parquet`

---

## Phase 5: Modeling — Demand Estimation ✅
**Date:** 2026-05-16
**Status:** Complete

### What was done:
1. **Try 1: Rule-Based Demand Estimation (Deprecated)**
   - Initial heuristic approach using max spikes and expert-rule multipliers.
   - Identified as mathematically insufficient due to right-censoring selection bias.
2. **Try 2: ML-Based Tri-Model Ensemble (Final Approved Architecture)**
   - **XGBoost AFT:** Models right-censoring via survival analysis, using calibrated elasticity bounds (upper = `y * elasticity`).
   - **LightGBM Quantile:** 95th percentile regressor trained with probability-weighted sampling to learn the true upper demand envelope.
   - **Feature-Space Peer Benchmark:** Rigorous historical 95th percentile matching across multi-dimensional clusters.
   - **Censor Probability Engine:** Hybrid heuristic + KMeans clustering score ∈ [0,1].
   - **Meta-Blender:** Ridge Regression backtested dynamically.
   - **Business Guardrails:** Clamps to historical stable floors and extreme peer ceilings (caught 143 extreme extrapolations).

---

## Phase 6: Deliverables ✅
**Date:** 2026-05-16
**Status:** Complete

### What was done:
- [x] Final predictions generated and saved to `output/insightai_final_predictions.csv`
- [x] Model artifacts and architecture thoroughly documented
- [x] Tri-Model Ensemble verified mathematically sound
- [x] Total projected network demand for Jan 2026 computed: 124,498,487.77 Liters
