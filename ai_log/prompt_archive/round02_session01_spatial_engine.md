# Prompt Archive — Round 2, Session 1 (2026-05-30)

## Prompt 7: Repository Audit & Gap Analysis

**Timestamp:** 2026-05-30 21:30 IST

**User Prompt:**
> "Can you read the Round 02 Problem Statement PDF? First need to identify our repo. There's a kalindu-branch where my teammate is cleaning data his way, approach-oshadha has my work. In first round we did what they asked in the Storming Round PDF and built our approach documented in InsightAI_Final_Report.tex."
> 
> "First thing I have concern on my data cleaning pipeline — is Kalindu's approach better or do we end up with the same dataset? How do we handle collaborative filtering, duplicate handling, demand distribution, null values? Also check if Round 2 datasets are same as Round 1."

**AI Actions:**
1. Read both problem PDFs (Round 1 + Round 2)
2. Switched between branches, read all src/silver/, src/gold/, src/model/ files on both
3. Ran `git diff approach-kalindu..approach-oshadha` — confirmed ZERO differences in code
4. Compared all 5 datasets via pandas (shape, columns, head, content)

**Key Finding:** Both branches have byte-for-byte identical source code. Only differences are in output artifacts (report PDF, PCA plot). All datasets identical between rounds.

---

## Prompt 8: Spatial Engine Design Review

**Timestamp:** 2026-05-30 22:06 IST

**User Prompt (full code snippet provided):**
> Shared a complete cKDTree + exponential decay implementation covering:
> - Step 1: CRS projection to EPSG:5234
> - Step 2: KD-Tree spatial indexing
> - Step 3: Exponential decay W=e^(-λd) with λ=0.003
> - Step 4: Huff competitive saturation W=1/(d+1)^β
> - Step 5: Latent opportunity ratio composition
> 
> "Is this better or need to do another way?"

**AI Response (key points):**
1. ✅ cKDTree approach is correct and enterprise-grade
2. ⚠️ Issue: Collapses all POIs into one aggregate score — loses per-category interaction signals
3. ⚠️ Issue: Parameters hardcoded — should be YAML config-driven
4. ⚠️ Issue: Not integrated with existing pipeline (loads from geojson, not parquet)
5. Recommended: Per-category gravity scores + backward-compatible flat counts

**Human Decision:** Accepted all refinements. Implemented per-category approach.

---

## Prompt 9–10: Implementation & Integration

**Timestamp:** 2026-05-30 22:08–22:12 IST

**AI implemented:**
- Complete rewrite of `src/gold/feature_poi.py` (116 → 339 lines)
- Updated `src/gold/build_model_input.py` with V2 gravity interactions
- Added `spatial_engine` section to `config/pipeline_config.yaml`

**Validation performed:**
- Full execution: 1.75 seconds, 33 features, 19,760 outlets
- 922 isolated goldmines identified
- Backward compatibility confirmed (all poi_count_* columns preserved)
