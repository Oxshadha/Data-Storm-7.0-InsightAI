# GenAI Transparency Log — InsightAI

> **Competition:** Data Storm 7.0  
> **Team:** InsightAI  
> **Purpose:** Document how, where, and why Generative AI was used during the competition. Evidence of critical evaluation, iterative prompting, and rigorous validation.

---

## Round 1 — Storming Round (36-hour Sprint)

| # | Timestamp | Phase | Tool Used | Purpose | Prompt Summary | Output Used? | Validation Done |
|---|-----------|-------|-----------|---------|----------------|-------------|-----------------|
| 1 | 2026-05-15 22:00 | Planning | Claude (Antigravity) | Repository architecture design | "Plan the repo as a data scientist using Lakehouse architecture" | ✅ Adapted | Reviewed against competition rubric — aligned with Bronze/Silver/Gold requirements |
| 2 | 2026-05-15 22:30 | Scaffolding | Claude (Antigravity) | Code scaffolding | "Create DQ check framework, quarantine system, config loader" | ✅ Modified | Reviewed function signatures, tested imports |
| 3 | 2026-05-15 23:30 | Data Engineering | Claude (Antigravity) | DuckDB SQL Optimization | "Write spatial join queries to extract Overture POIs via DuckDB S3 Parquet" | ✅ Adapted | Manually validated query results against OSM visual maps |
| 4 | 2026-05-16 02:00 | Feature Engineering | Claude (Antigravity) | Python Prototyping | "Implement 3-month rolling mean smoothing and Lazy Rep flag logic" | ✅ Modified | Verified smoothing output against manual Excel calculation for 5 sample outlets |
| 5 | 2026-05-16 06:00 | Statistical Validation | Claude (Antigravity) | Plotting Scripts | "Generate matplotlib scripts for K-Means Elbow method and LightGBM Gain charts" | ✅ Adapted | Cross-checked elbow plot against sklearn inertia values |
| 6 | 2026-05-16 08:00 | Logic Validation | Claude (Antigravity) | Peer-Review Simulation | "Stress-test the Structural Safety Floor formula against edge cases" | ✅ Used | Tested edge cases: zero-volume outlets, negative CV, outlets with no POIs |

**Round 1 Policy:** At no point did GenAI replace human architectural decision-making. All core strategies — p90 Quantile Regression, specific cultural POI targeting, collaborative filtering logic, and the Structural Safety Floor — were strictly human-led.

---

## Round 2 — Final Round

| # | Timestamp | Phase | Tool Used | Purpose | Prompt Summary | Output Used? | Validation Done |
|---|-----------|-------|-----------|---------|----------------|-------------|-----------------|
| 7 | 2026-05-30 21:30 | Analysis | Claude (Antigravity) | Repository audit & gap analysis | "Read Round 2 problem PDF and our R1 report. Compare both branches, verify datasets are identical, identify what needs to be built." | ✅ Used | Verified dataset identity programmatically (byte-for-byte comparison). Confirmed branch code identity via `git diff`. |
| 8 | 2026-05-30 22:06 | Architecture | Claude (Antigravity) | Spatial engine design review | "We proposed a cKDTree + exponential decay approach. Is this better or need another way?" — shared full code snippet for review. | ✅ Critically reviewed & refined | AI identified 3 issues: (1) collapsing all POI categories into one score loses interaction signals, (2) hardcoded parameters, (3) not integrated with pipeline. Human decided to keep per-category scores. |
| 9 | 2026-05-30 22:08 | Implementation | Claude (Antigravity) | Spatial Intelligence Engine V2 | "Implement refined cKDTree spatial engine with per-category gravity, Huff competitive saturation, config-driven parameters, backward-compatible poi_count_* columns" | ✅ Adapted | Executed on full dataset: 1.75s runtime, verified 33 features, 922 goldmines, backward compat confirmed. Manually spot-checked gravity scores for OUT_08605 (highest ratio) against OSM map. |
| 10 | 2026-05-30 22:10 | Integration | Claude (Antigravity) | ABT builder upgrade | "Update build_model_input.py to use new gravity scores in spatio-temporal interactions alongside R1 binary interactions" | ✅ Modified | Verified new columns (Tuition_Weekend_Gravity etc.) appear in ABT. Confirmed R1 binary columns still computed identically. |

### Round 2 — Key Prompt–Response Interactions

#### Prompt 8: Spatial Engine Design Review

**Human prompt (summarized):**
> "To build an enterprise-level spatial engine, we have to solve the N-Squared Problem. 20,000 shops × 100,000 POIs = 2 billion calculations. Enterprise companies use cKDTree. Here is the exact step-by-step approach with exponential decay and Huff gravity. Is this better or need to do another way?"

**AI response (key critique):**
> "The approach is fundamentally excellent. Three refinements needed: (1) Don't collapse all POI categories into one score — your ABT has per-category interactions like Tuition_Weekend_Surge that depend on separate school/education/hospital signals. (2) λ and β should be in YAML config, not hardcoded. (3) Must preserve backward-compatible poi_count_* columns."

**Human decision:** Accepted all 3 refinements. Kept per-category gravity approach. Selected λ=0.003 based on pedestrian walking distance research (half-influence at ~230m matches Sri Lankan retail corridor density).

#### Prompt 9: Implementation Validation

After implementation, we ran the full engine and manually validated:
- `OUT_08605` has `total_driver_gravity = 27.98` (highest) — checked on OpenStreetMap, confirmed this outlet is near a dense cluster of schools and hospitals in Colombo
- `is_isolated_goldmine` flagged 922 outlets — these have above-median footfall gravity but zero competitors within 1km, confirming they are genuinely underserved markets

---

## AI Usage Philosophy

1. **AI as Thought Partner:** We used GenAI to stress-test our mathematical approaches and identify blind spots (e.g., the category-collapse issue in Prompt 8).
2. **AI as Accelerator:** Boilerplate code (KD-Tree queries, NumPy vectorization, config integration) was generated by AI and then reviewed/modified.
3. **Human-Led Architecture:** The choice of exponential decay over Gaussian, λ=0.003, Huff β=2.0, and the concept of "isolated goldmines" were human decisions informed by retail analytics literature.
4. **Rigorous Validation:** Every AI-generated output was tested on the full dataset and spot-checked against geographic ground truth before integration.
