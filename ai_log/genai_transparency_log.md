# GenAI Transparency Log — InsightAI

> **Competition:** Data Storm 7.0 — Storming Round
> **Team:** InsightAI
> **Purpose:** Document how, where, and why Generative AI was used during the 36-hour hackathon.

---

## Usage Log

| # | Timestamp | Phase | Tool Used | Purpose | Prompt Summary | Output Used? | Validation Done |
|---|-----------|-------|-----------|---------|----------------|-------------|-----------------|
| 1 | 2026-05-15 22:00 | Planning | Claude (Antigravity) | Repository architecture design | "Plan the repo as a data scientist using Lakehouse architecture" | ✅ Adapted | Reviewed against competition rubric — aligned with Bronze/Silver/Gold requirements |
| 2 | 2026-05-15 22:30 | Scaffolding | Claude (Antigravity) | Code scaffolding | "Create DQ check framework, quarantine system, config loader" | ✅ Modified | Reviewed function signatures, tested imports |
| 3 | 2026-05-15 23:20 | Phase 0 validation | Codex | Plan creation, raw data placement review, Bronze-stage execution, and checklist completion | "Create an industry-standard model building plan, commit current changes, start Phase 0, and complete checklist" | ✅ Adapted | Verified raw file presence, Git ignore rules, successful `python3 run_pipeline.py --stage bronze`, row counts, Parquet outputs, and pipeline log creation |
| 4 | 2026-05-15 23:50 | Silver Layer | Claude (Antigravity) | Silver forensics implementation | "Implement clean_transactions, coordinate bounds checking, handle negative/zero volumes with QuarantineManager" | ✅ Adapted | Verified 4,753 negative returns caught and rejected, pipeline executed |
| 5 | 2026-05-16 00:20 | Gold Layer | Claude (Antigravity) | POI scraping and gold features | "Implement bulk POI scraping using bounding box, build model input combining POI KDTree, transactions, and censoring logic" | ✅ Adapted | Verified 26,170 POIs downloaded, KDTree counts accurate, pipeline produced `model_input.parquet` |
| 6 | 2026-05-16 00:35 | Predict Layer | Claude (Antigravity) | Latent potential estimation logic | "Create an expert rule-based demand estimation using peer group 90th percentile and historical peak volume adjusted for seasonality" | ✅ Adapted | Checked calculation logic and successfully generated 20,000 predictions in `insightai_predictions.csv` |
| 7 | 2026-05-16 10:00 | Visualization & Modeling | Claude (Antigravity) | Generate EDA and Model Training Notebooks | "Create a Jupyter Notebook to visualize data anomalies and step through the demand estimation model mathematically" | ✅ Adapted | Generated and reviewed visualizations for quarantined records and censoring thresholds |
| 8 | 2026-05-16 14:00 | Machine Learning | Claude (Antigravity) | Final Tri-Model Ensemble Architecture | "Implement ML-based Tri-Model Ensemble (XGBoost AFT, LGBM Quantile, Peer) replacing rule-based heuristics" | ✅ Adapted | Validated mathematically via OOF backtesting; confirmed Guardrails successfully capped extreme extrapolations |

---

## Guidelines

- **Every AI interaction** that produces code, analysis, or design decisions should be logged here.
- Mark whether the output was used **as-is**, **adapted**, or **rejected**.
- Document what **validation** was performed on AI-generated output.
- Save significant prompts/responses in `ai_log/prompt_archive/`.
