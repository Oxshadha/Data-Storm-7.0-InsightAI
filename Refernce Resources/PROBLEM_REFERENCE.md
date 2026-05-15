# Data Storm 7.0 — Storming Round Problem Reference

> **Source:** `Data Storm 7.0 - Storming Round Problem.pdf` — OCTAVE, John Keells Group
> **Competition:** Data Storm 7.0 (Storming Round)
> **Duration:** 36 hours hackathon

---

## Table of Contents

1. [Background](#1-background)
2. [Preliminary Problem Statement](#2-preliminary-problem-statement)
3. [Data Description](#3-data-description)
4. [Technical & Architecture Requirements](#4-technical--architecture-requirements)
5. [Deliverables](#5-deliverables)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Initial Data Profiling (Our Observations)](#7-initial-data-profiling)

---

## 1. Background

### 1.1 Business Context

A **leading beverage manufacturer in Sri Lanka** operates a massive distribution network spanning over **80,000 traditional retail outlets** — from bustling urban grocery stores in Colombo to small **"kades"** (corner shops) and local eateries in rural outstations.

**Current Problem:**
- Sales teams allocate trade marketing budgets, coolers, and promotional discounts based on **historical sales averages**
- Leadership realizes this is **fundamentally flawed**: historical sales only reflect what an outlet **did sell**, not what it **could sell**
- A high-traffic town-center kade might be **underperforming** due to poor stock management or credit constraints
- A small village shop might already be **maxed out**

**Desired Shift:**
- From **Historical-Based** resource allocation → **Potential-Based Allocation**
- Need to predict the **Maximum Monthly Purchase Potential** of every traditional trade outlet
- Goal: optimize trade spend and cooler deployments

### 1.2 Current Scope

Focus: **20,000 traditional trade outlets** (kades, groceries, eateries, pharmacies, etc.) serviced by **10 key distributors** across **4 key provinces** in Sri Lanka.

| Province | # Distributors | Distributor IDs |
|----------|---------------|-----------------|
| Western | 3 | `DIST_W_01`, `DIST_W_02`, `DIST_W_03` |
| Central | 3 | `DIST_C_01`, `DIST_C_02`, `DIST_C_03` |
| North-Western | 2 | `DIST_NW_01`, `DIST_NW_02` |
| Southern | 2 | `DIST_S_01`, `DIST_S_02` |

---

## 2. Preliminary Problem Statement

> [!IMPORTANT]
> **Objective:** Design a robust analytical framework and model to estimate the **latent maximum monthly volume potential (in liters)** for the retail network for the **month of January 2026**.

### Key Realities

1. **There is no perfect "ground truth"** — no secret answer key
2. **"Potential" is a theoretical ceiling** — a hidden variable
3. **You are NOT given a target variable (y)**
4. **Historical volume is censored** — it represents the minimum of either:
   - **(A)** The true consumer demand
   - **(B)** Systemic Constraints (credit limits, stockouts, or delivery caps)

> [!CAUTION]
> **Your goal:** Build the most **logical, defensible, and mathematically sound** framework to **uncap this latent demand**. This is a left-censored demand problem.

---

## 3. Data Description

### 3.1 Dataset Description

Participants are provided with the following datasets:

| # | File | Description |
|---|------|-------------|
| 1 | `transactions_history_final.csv` | Granular outlet-level transaction data |
| 2 | `outlet_master.csv` | Outlet related metadata |
| 3 | `outlet_coordinates.csv` | Longitude and Latitude for each outlet |
| 4 | `distributor_seasonality_details.csv` | Month-specific seasonality for distributors |
| 5 | `holiday_list.csv` | List of holidays |

#### Detailed Column Descriptions

**`transactions_history_final.csv`** (Main transaction file)

| Column | Data Type | Description |
|--------|-----------|-------------|
| `Outlet_ID` | String | Unique identifier for the retail outlet |
| `Year` | Integer | Transaction year (2023, 2024, 2025) |
| `Month` | Integer | Transaction month (1 to 12) |
| `Distributor_ID` | String | Identifier for the 10 distributors |
| `SKU_ID` | String | Unique identifier for the product |
| `Volume_Liters` | Float | Total volume sold in liters |
| `Total_Bill_Value` | Float | Total transaction value in LKR |

> [!NOTE]
> The PDF lists `Product_Name` as a column in the dataset description Excel but it is **NOT present** in the actual CSV. Only `SKU_ID` is provided (SKU_01 through SKU_10).

**`outlet_master.csv`**

| Column | Data Type | Description |
|--------|-----------|-------------|
| `Outlet_ID` | String | Unique identifier for the retail outlet |
| `Outlet_Size` | String | Categorical size (Small, Medium, Large, Extra Large) |
| `Cooler_Count` | Integer | Number of company-owned refrigerators |
| `Outlet_Type` | String | Category of trade (e.g., Grocery, Eatery, Pharmacy) |

**`outlet_coordinates.csv`**

| Column | Data Type | Description |
|--------|-----------|-------------|
| `Outlet_ID` | String | Unique identifier for the retail outlet |
| `Latitude` | Float | GPS Latitude coordinate |
| `Longitude` | Float | GPS Longitude coordinate |

**`distributor_seasonality_details.csv`**

| Column | Data Type | Description |
|--------|-----------|-------------|
| `Distributor_ID` | String | Identifier for the 10 distributors |
| `Year` | Integer | Corresponding Year |
| `Month` | Integer | Corresponding Month |
| `Seasonality_Index` | String | Tag: Favorable, Moderate, Un-Favorable |

**`holiday_list.csv`**

| Column | Data Type | Description |
|--------|-----------|-------------|
| `Date` | Date | Date of the specific holiday |
| `Holiday_Name` | String | Holiday name along with description |
| `Holiday_Type` | String | Type: Public, Bank, Poya Day, Mercantile |

### 3.2 The External Data Challenge (POI Scraping)

> [!IMPORTANT]
> **POI data is NOT provided.** Teams must acquire it externally.

- Use public APIs: **OpenStreetMap**, **Overpass API**, or web scraping
- Identify nearby: **schools, bus stands, hospitals, tourist attractions**, etc.
- These POIs help estimate the **potential** (catchment/footfall drivers)

### 3.3 The Data Quality Reality

> [!WARNING]
> All datasets are **unprocessed, raw exports** from legacy **Sales Force Automation (SFA)** and **distributor ERP systems**. Data has **NOT been sanitized**.

FMCG data issues to expect:
- **Connectivity blackouts** (missing data periods)
- **Human data-entry shortcuts** (typos, inconsistencies)
- **Automated ghost entries** (system-generated artifacts)
- **Master-data decay** (stale/incorrect master records)

> **First major task = Data Forensics:** Separate true market signals from system artifacts and human errors BEFORE building potential framework.

---

## 4. Technical & Architecture Requirements

### 4.1 Lakehouse Pipeline Architecture

> [!IMPORTANT]
> Codebase MUST be structured into **three clearly defined layers** mirroring industry-standard Lakehouse practices.

| Layer | Name | Purpose |
|-------|------|---------|
| **Bronze** | Raw Ingestion | Ingest all flat files **as-is** with **no transformations**. Preserve original data exactly. |
| **Silver** | Cleaned | Apply all DE checks and data cleaning. Records failing checks **must be quarantined** into a **separate rejected records store** with **documented failure reason**. Must NOT be silently dropped or carried forward. |
| **Gold** | Enriched | Feature engineering + model-ready output. Draws from all Silver-layer datasets including internally provided AND externally scraped data. |

> [!NOTE]
> A cleanly structured **local directory setup** (storing intermediate `.csv` or `.parquet` files in dedicated folders) is **perfectly sufficient**. No need for actual cloud Lakehouse infrastructure.

### 4.2 Reusable Data Quality Checks

> [!IMPORTANT]
> DQ checks must be implemented as **reusable, parameterizable functions** applied **consistently across ALL datasets**.

**Required check types (minimum — not comprehensive):**

| Check Type | Description |
|------------|-------------|
| **Duplicate Check** | Detect duplicate records based on configurable primary key (single or composite) |
| **Null Check** | Flag records where mandatory fields contain null or empty values |
| **Referential Integrity Check** | Validate foreign key values in one dataset exist in a reference dataset |
| **Value Range Check** | Assert numeric fields fall within expected min/max boundary |
| **Format / Type Check** | Validate fields conform to expected data type or format (e.g., dates, IDs) |

> Teams are **encouraged to identify and implement additional checks** relevant to the data.

---

## 5. Deliverables

> [!CAUTION]
> Evaluation is **100% qualitative** — there is no hidden "answer key" to optimize against. Focus is on **data forensics methodology**, **Lakehouse implementation**, **code & analytical robustness**, and **business logic**.

### Deliverable 1: The "Latent Potential" Output (CSV)

- **File:** `teamname_predictions.csv`
- **Contents:**
  - `Outlet_ID` — the outlet identifier
  - `Maximum_Monthly_Liters` — your predicted potential for **January 2026**

### Deliverable 2: Reproducible Codebase (GitHub Link or Zipped Repo)

Complete code (Python scripts / Jupyter Notebooks, etc.) for data cleaning, POI scraping, and modeling.

**Must include:**
- (a) **`README.md`** with clear instructions on how to run the pipeline end-to-end
- (b) **Clearly structured codebase** reflecting the **Bronze → Silver → Gold** layer separation
- (c) **Clearly structured modeling/analytical pipelines** from simple EDAs to final prediction

### Deliverable 3: PDF Summary Report (Max 5 Pages incl. Cover)

A comprehensive technical document that **must explicitly address:**

| Section | What to Document |
|---------|-----------------|
| **(a) Data Forensics & Hygiene** | Specific system anomalies trapped, how records were quarantined, DE checks setup |
| **(b) POI Data Acquisition** | Technical approach to acquiring external geospatial signals (APIs, scraping methods/tools), types of POIs targeted as catchment drivers, how mapped to internal outlets |
| **(c) Causal Base Logic** | Initial statistical/probabilistic approach to solving the **left-censored demand curve**. Exact methodology used to calculate the uncapped potential ceiling |
| **(d) GenAI Transparency Log** | Dedicated log of how, where, and why GenAI was used during the 36 hours. How LLMs were used as **engineering accelerators** |

---

## 6. Evaluation Metrics

| Criteria | Weight |
|----------|--------|
| **Data Engineering & Forensics (Cleaning)** | **40%** |
| **Methodology & Base Math** | **40%** |
| **Generative AI Utilization & Workflow** | **20%** |

### 6.1 Data Engineering & Forensics (40%)

Judges will evaluate:
- ✅ Did the team implement a clearly structured **Bronze → Silver → Gold pipeline** with a **rejected records store**?
- ✅ Was the DE checks implementation **reusable, parameterizable**, and applied **consistently** across datasets?
- ✅ Did the team successfully **identify, clean and neutralize legacy system artifacts**?
- ✅ How robust was the **web-scraping / API pipeline** for external POI data?
- ✅ Did the team engineer/create **highly relevant features** to isolate true market signals?

### 6.2 Methodology & Base Math (40%)

Judges will evaluate:
- ✅ How did the team **conceptualize the problem** of "Latent Potential"?
- ✅ What **mathematical or statistical approach** was used to handle the **missing target variable** and **right-censored data**?

> [!NOTE]
> The PDF mentions "right-censored" here but earlier describes it as "left-censored." The concept is that observed sales are **censored from above** — we see sales ≤ true potential. This is technically **right-censoring** of the demand distribution (the upper tail is truncated by constraints).

### 6.3 Generative AI Utilization & Workflow (20%)

Judges will evaluate:
- ✅ Did the team **clearly document** how, where, and why they used LLMs (e.g., Gemini, Copilot, ChatGPT) during the hackathon?
- ✅ Was AI used intelligently as an **accelerator** (brainstorming causal frameworks, writing boilerplate scraping code, debugging complex statistical logic) rather than a **crutch**?
- ✅ Is there evidence that the team **critically evaluated**, **iteratively prompted**, and **rigorously validated** the AI-generated code and assumptions, rather than **blindly trusting** the output?

---

## 7. Initial Data Profiling (Our Observations)

### 7.1 File Sizes & Row Counts

| File | Rows | Columns | Size |
|------|------|---------|------|
| `transactions_history_final.csv` | **2,376,389** | 7 | ~161 MB |
| `outlet_master.csv` | **20,000** | 4 | ~508 KB |
| `outlet_coordinates.csv` | **20,000** | 3 | ~575 KB |
| `distributor_seasonality_details.csv` | **360** | 4 | ~10 KB |
| `holiday_list.csv` | **349** | 3 | ~20 KB |

### 7.2 Data Quality Issues Discovered

> [!WARNING]
> These are the **messy data artifacts** the organizers intentionally planted. These MUST be handled in the Silver layer.

#### `outlet_master.csv`

| Issue | Detail |
|-------|--------|
| **Case inconsistency in `Outlet_Size`** | Both `"Small"` (9,672) and `"small"` (600) exist — need standardization |
| **Null `Outlet_Size`** | 196 records have missing size values |
| **Typos in `Outlet_Type`** | `"Grocry"` (390) should be `"Grocery"`, `"Bakry"` (395) should be `"Bakery"`, `" Eatery "` (200, with spaces) should be `"Eatery"` |

**Outlet Type Distribution:**

| Outlet_Type | Count | Notes |
|-------------|-------|-------|
| Hotel | 2,797 | ✅ Clean |
| Grocery | 2,768 | ✅ Clean |
| SMMT | 2,723 | ✅ Clean |
| Pharmacy | 2,691 | ✅ Clean |
| Kiosk | 2,691 | ✅ Clean |
| Bakery | 2,678 | ✅ Clean |
| Eatery | 2,667 | ✅ Clean |
| Bakry | 395 | ⚠️ Typo → "Bakery" |
| Grocry | 390 | ⚠️ Typo → "Grocery" |
| ` Eatery ` | 200 | ⚠️ Leading/trailing spaces → "Eatery" |

**Outlet Size Distribution:**

| Outlet_Size | Count | Notes |
|-------------|-------|-------|
| Small | 9,672 | ✅ |
| Medium | 5,702 | ✅ |
| Large | 2,887 | ✅ |
| Extra Large | 943 | ✅ |
| small | 600 | ⚠️ Case → "Small" |
| NaN | 196 | ⚠️ Missing values |

**Cooler Count Values:** 0, 1, 2, 3, 4, 5

#### `transactions_history_final.csv`

| Issue | Detail |
|-------|--------|
| **Negative volumes** | **4,753 records** with negative `Volume_Liters` (min: -956.44 L). Likely returns/reversals |
| **Zero volumes** | **100 records** with exactly 0.0 liters |
| **Negative bill values** | **4,753 records** (same count as negative volumes — perfectly correlated) |
| **Extreme outliers** | Max volume = **9,438.58 L** in a single transaction — needs investigation |
| **Missing `Product_Name`** | Column listed in dataset description but NOT in actual CSV |
| **No nulls** | Zero nulls across all columns |

**Volume Liters Stats:**

| Stat | Value |
|------|-------|
| Mean | 52.62 L |
| Std | 95.49 L |
| Min | -956.44 L |
| 25% | 10.18 L |
| Median | 23.16 L |
| 75% | 54.43 L |
| Max | 9,438.58 L |

**Data spans:** 3 years × 12 months = **36 months** (Jan 2023 → Dec 2025), ~65,500–67,000 records per month

**SKUs:** SKU_01 through SKU_10 (10 products)

#### `distributor_seasonality_details.csv`

| Seasonality_Index | Count |
|-------------------|-------|
| Moderate | 249 |
| Favorable | 81 |
| Un-Favorable | 30 |

- 360 rows = 10 distributors × 3 years × 12 months ✅ Complete

#### `holiday_list.csv`

- **349 holidays** across 2023–2025
- **76 unique dates**, **41 unique holiday names**
- Types: Bank (99), Public (98), Mercantile (93), Poya Day (59)
- Date format: ISO 8601 with timezone (`2023-01-06T00:00:00Z`)
- Multiple holiday types can fall on the same date (same date appears with different types)

#### `outlet_coordinates.csv`

- All 20,000 outlets have coordinates ✅
- 19,836 unique latitudes, 19,691 unique longitudes (some outlets share locations — could be co-located or data quality issue)
- Coordinates fall within Sri Lanka boundaries

---

## 8. Critical Success Factors (Summary)

> [!TIP]
> **What will differentiate a winning submission:**

### Must-Haves (Non-Negotiable)
1. ✅ **Bronze → Silver → Gold** directory/pipeline structure
2. ✅ **Rejected records store** with documented failure reasons (quarantine, don't silently drop)
3. ✅ **Reusable, parameterizable DQ check functions**
4. ✅ **POI scraping pipeline** (OpenStreetMap/Overpass)
5. ✅ **Prediction CSV** with `Outlet_ID` and `Maximum_Monthly_Liters` for Jan 2026
6. ✅ **Sound mathematical framework** for uncapping censored demand
7. ✅ **GenAI transparency log**
8. ✅ **README.md** with end-to-end run instructions
9. ✅ **PDF Report** (max 5 pages)

### Differentiators
- **Data Forensics depth:** How many and what types of anomalies you caught
- **Statistical rigor:** Tobit regression, survival analysis, quantile regression, or other censored-data methods
- **Feature engineering quality:** POI density, competitor proximity, catchment area analysis
- **Business logic soundness:** Does your "potential" make business sense?
- **Code quality:** Clean, modular, well-documented

---

## 9. Conceptual Approach Notes

### The Censored Demand Problem

This is fundamentally a **censored regression** problem:

```
Observed_Sales = min(True_Demand, Constraint_Ceiling)
```

When `Observed_Sales < Constraint_Ceiling` → we see the true demand (uncensored)
When `Observed_Sales = Constraint_Ceiling` → demand is censored (we only know demand ≥ observed)

**Possible statistical approaches:**
- **Tobit Model** (Type I) — Standard approach for censored data
- **Survival Analysis** (Kaplan-Meier, Cox PH) — Treat potential as "time to event"
- **Quantile Regression** — Estimate upper quantiles of demand distribution
- **Mixture Models** — Model constrained vs. unconstrained outlets separately
- **Bayesian Hierarchical Models** — Pool information across similar outlets

### Identifying Censoring Indicators

Signals that an outlet is **constrained** (censored):
- **Hitting credit limits** (need to infer from patterns)
- **Stockouts** (gaps in purchasing patterns)
- **Delivery caps** (volume plateaus)
- **Flat monthly volumes** (suspiciously consistent = likely capped)
- **Low cooler count relative to peers** (capacity constraint)
- **High-traffic POI location but low sales** (under-potential)

### POI Feature Engineering Ideas

- Count of schools within X km radius
- Count of bus stands / transport hubs
- Proximity to hospitals
- Tourist attraction density
- Population density proxy
- Competitor outlet density
- Road type / accessibility
