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
- [x] `src/bronze/ingest_poi.py` — **Fully implemented** POI scraper:
  - Uses Overpass API (no extra library — pure `requests`)
  - Fetches ALL 8 POI categories in a single API call per outlet
  - Exponential backoff on rate limit errors (429/504)
  - **Checkpoint/resume system** — saves every 50 outlets; restarts from where it left off
  - Saves intermediate batches as `data/bronze/poi_raw/poi_batch_XXXX.parquet`
  - Final combined output: `data/bronze/poi_raw.parquet`
  - Custom `User-Agent` header (Overpass API requirement)

### Still TODO:
- [ ] Run `ingest_poi.py` to scrape POIs for all 20,000 outlets (~7hrs estimated)

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

## Phase 4: Gold — Feature Engineering 🔴
**Date:** —
**Status:** Not started

### TODO:
- [ ] Outlet profile features (size, type, coolers)
- [ ] Transaction behavioral features (trends, variability, patterns)
- [ ] POI density & catchment features
- [ ] Seasonality & holiday features
- [ ] Censoring signal detection (flat volumes, capacity constraints)
- [ ] Join all into `model_input.parquet`

---

## Phase 5: Modeling — Demand Estimation 🔴
**Date:** —
**Status:** Not started

### TODO:
- [ ] Censoring detection (identify constrained outlets)
- [ ] Implement demand estimation (Tobit / Quantile / Bayesian)
- [ ] Generate `insightai_predictions.csv`

---

## Phase 6: Deliverables 🔴
**Date:** —
**Status:** Not started

### TODO:
- [ ] Final predictions CSV
- [ ] 5-page PDF report
- [ ] Final README review
- [ ] GenAI log complete
