# Experiment Log — InsightAI

> Track all modeling experiments, approaches, and results.

---

## Experiments

| # | Date | Approach | Description | Metrics | Notes |
|---|------|----------|-------------|---------|-------|
| 0 | 2026-05-15 | Phase 0 setup validation | Verified source data availability, Git ignore rules for large raw files, and executed Bronze ingestion from `Refernce Resources/` to `data/bronze/*.parquet`. | Bronze row counts: transactions 2,376,389; outlet master 20,000; coordinates 20,000; seasonality 360; holidays 349. | `python3 run_pipeline.py --stage bronze` completed successfully. Generated Parquet outputs and logs are reproducible and ignored by Git. |
| 1 | 2026-05-16 | Try 1: Rule-Based Heuristics | Applied hardcoded seasonality and historical peak multipliers to estimate uncapped potential. | Generated 20k predictions. | Rejected: Heuristics suffer from right-censoring selection bias and don't mathematically account for infinite headroom. |
| 2 | 2026-05-16 | Try 2: Tri-Model Ensemble | XGBoost AFT (Survival), LGBM Quantile Regressor, and Feature-Space Peer bounds blended via Ridge. | Projected total network Jan 2026 demand: 124,498,487.77 Liters. | Accepted: AFT models true unbounded potential; LGBM learns 95th percentile envelope. Guardrails caught 143 extreme anomalies. |

---

## Approach Notes

<!-- Document your thinking process, hypotheses, and findings here -->

### Phase 0 Notes

- The configured raw source remains `Refernce Resources/`.
- `transactions_history_final.csv` is kept local and ignored by Git because it exceeds GitHub's 100 MB limit.
- Bronze Parquet outputs are the pipeline interface for downstream Silver and Gold stages.
- The initial implementation should now focus on wiring and completing the Silver cleaning modules.
