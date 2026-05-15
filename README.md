# Data Storm 7.0 — InsightAI

> **Competition:** Data Storm 7.0 — Storming Round (36-hour hackathon)
> **Team:** InsightAI
> **Objective:** Estimate the latent maximum monthly volume potential (in liters) for ~20,000 retail outlets for January 2026.

---

## 🏗️ Architecture — Lakehouse Pipeline

This repository follows the **Medallion Architecture** (Bronze → Silver → Gold) with a forensic data detective approach.

```
Bronze (Raw)  →  Silver (Cleaned)  →  Gold (Enriched)  →  Predictions
     ↓                  ↓
  As-Is CSV       Quarantined
  → Parquet       Rejected Records
```

### Directory Structure

```
├── config/                     # Pipeline configuration (YAML)
├── data/
│   ├── bronze/                 # Raw ingestion — zero transformations
│   ├── silver/                 # Cleaned + DQ-checked data
│   │   └── rejected/          # Quarantined records with documented reasons
│   └── gold/                   # Feature-engineered, model-ready data
├── src/
│   ├── bronze/                 # Ingestion scripts
│   ├── silver/                 # DQ checks + forensic cleaning
│   ├── gold/                   # Feature engineering
│   ├── modeling/               # Demand estimation models
│   └── utils/                  # Config, logging, I/O helpers
├── notebooks/                  # EDA, forensics, model experiments
├── ai_log/                     # GenAI transparency log (required deliverable)
├── experiments/                # Experiment tracking
├── output/                     # Final predictions + PDF report
├── run_pipeline.py             # Master pipeline orchestrator
└── Refernce Resources/         # Original competition data (read-only)
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ensure Raw Data

Place `transactions_history_final.csv` (~169MB) in the `Refernce Resources/` directory. This file is excluded from Git due to size limits.

### 3. Run the Pipeline

```bash
# Full pipeline: Bronze → Silver → Gold → Predict
python run_pipeline.py

# Or run individual stages:
python run_pipeline.py --stage bronze
python run_pipeline.py --stage silver
python run_pipeline.py --stage gold
python run_pipeline.py --stage predict
```

---

## 🔬 Data Detective Philosophy

We treat data anomalies as **evidence**, not garbage:

| Ghost Type | Action | NOT This |
|---|---|---|
| Negative Returns | Tag as `RETURN`, aggregate for net volume | ~~Drop~~ |
| Zero-Volume Rows | Tag as `SYSTEM_ADJUSTMENT`, quarantine | ~~Ignore~~ |
| Duplicate Retries | Detect, keep one, quarantine rest with reason | ~~Deduplicate silently~~ |
| Extreme Outliers | Cross-reference with outlet profile | ~~Cap at percentile~~ |

---

## 📊 Deliverables

1. **`output/insightai_predictions.csv`** — Outlet_ID + Maximum_Monthly_Liters for Jan 2026
2. **This repository** — Reproducible codebase with Bronze → Silver → Gold structure
3. **`output/report/InsightAI_Report.pdf`** — 5-page technical summary

---

## 🤖 GenAI Usage

All AI interactions are documented in [`ai_log/genai_transparency_log.md`](ai_log/genai_transparency_log.md).
