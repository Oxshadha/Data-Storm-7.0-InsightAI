# InsightAI: Data Storm 7.0 — Storming Round Report

---

## Page 1: Executive Summary & Architecture

### The Strategic Pivot: From Rearview to Potential-Based Allocation
The core business challenge posed by this competition is a classic problem of **right-censored demand**. FMCG supply chains naturally hit capacity limits (coolers, inventory drops), forcing historical sales data to artificially flatline. Relying on traditional "rearview mirror" historical allocation models merely teaches algorithms to predict these systemic out-of-stocks, resulting in misallocated resources and squandered revenue.

Team InsightAI pivoted the company strategy entirely toward **Potential-Based Allocation**. By mapping unconstrained latent demand, we empower the business to deploy capital and coolers precisely where the "Growth Gap" (Potential minus Actuals) is highest. 

### The Forensic Lakehouse Implementation
To achieve this, we engineered a production-grade **Medallion Lakehouse Architecture**:
1. **Bronze (Raw):** Integrated historical transactions, master data, and executed a highly performant cloud-native ingestion pipeline for raw spatial data.
2. **Silver (Sanitized):** Implemented strict forensic hygiene. Rather than dropping dirty data, we established a quarantine store, explicitly tagging anomalies (Lazy Reps, Master Data Decay) while preserving data integrity.
3. **Gold (Enriched):** Transformed clean data into a high-dimensional analytical base table (ABT). We projected Sri Lankan spatial nodes (e.g., mass tuition centers, tourist beaches) against temporal realities (weekends, holidays) to mathematically capture complex human behavior.

### The Business Result
By optimizing an asymmetric Quantile Gradient Boosting machine learning architecture on 450,600 uncensored historical records, we successfully mapped the upper boundary of unconstrained demand.
- **Projected Network Demand (January 2026):** **9,641,821 Liters**
- The precise identification of unconstrained demand unlocks granular, actionable insights for every specific outlet, maximizing the ROI of future capital deployments.

---

## Page 2: Data Forensics and Hygiene (The Detective)

A robust analytical framework relies entirely on the integrity of its foundation. We adopted a strict **"No-Drop" forensic philosophy**, recognizing that in supply chain data, anomalies are not noise—they are signals of operational failures.

### The Quarantine Pattern
To ensure absolute reproducibility and accountability, we strictly avoided destructive data operations like `df.dropna()`. Instead, we developed a vectorized quarantine mechanism. Any record violating data quality rules was routed to an isolated `rejection_manifest.csv` with an appended diagnostic column detailing the exact reason for failure. 

### Exorcising System Ghosts
Legacy Sales Force Automation (SFA) systems are notorious for generating "ghost" transactions. We applied strict business logic to sanitize the dataset:
- **Netting Returns:** Rather than deleting negative volumes, we correctly aggregated them to capture actual net consumption.
- **Filtering System Adjustments:** Zero-volume rows—hallmarks of automated system corrections—were strictly quarantined.
- **Deduplicating Uploads:** We utilized composite primary keys (`Outlet_ID`, `Year`, `Month`, `SKU_ID`, `Volume`) to detect and route retry-upload duplicates.

### Resolving Master Data Decay & "Lazy Reps"
Master data inherently decays; a shop labeled "Small" three years ago may now be an "Extra Large" transit hub. 
- **Dynamic Tiering:** We abandoned static SFA tags. We mathematically clustered outlets into empirical Tiers (1–4) based on their rolling 6-month transaction volume behavior, instantly correcting outdated `outlet_master` categorizations.
- **Lazy Rep Detection:** We identified behavioral anomalies where sales reps artificially dumped massive volumes into single SKUs to hit quotas, flagging these constrained outlets without discarding the critical underlying revenue data.

---

## Page 3: POI Data Acquisition & Spatial Signals (The Explorer)

To model human behavior accurately, we needed to look beyond the four walls of the outlet. We avoided the amateur approach of recursively looping brittle REST APIs (which often result in rate-limit bans and partial data) and built an enterprise-grade spatial engine.

### Overture Maps via DuckDB
We bypassed standard OpenStreetMap queries and integrated directly with the **Overture Maps Foundation's** cloud-native Parquet files hosted on AWS S3. 
By utilizing DuckDB's spatial extensions, we pushed bounding box filters directly into the cloud query. This allowed us to extract the entirety of Sri Lanka's relevant Point of Interest (POI) data in under 3 seconds. Crucially, Overture aggregates Meta's business data, enabling us to capture localized, unstructured Sri Lankan commercial nodes (such as mass tuition centers and regional sports grounds) that traditional mapping APIs completely miss.

### The Catchment Join & Spatio-Temporal Engineering
Using `GeoPandas`, we projected a precise 1,000-meter buffered catchment polygon around all 20,000 outlets. We mapped the POIs to these catchments to isolate environmental proxies (e.g., Youth/Education, Leisure, Athletic hubs).

However, static geography does not dictate impulse beverage sales—**timing does**. We created interactive Spatio-Temporal features (The "Secret Sauce") by crossing static space with dynamic time:
1. **The Athletic/Cricket Surge:** We multiplied Athletic/Grounds POIs by March/April timelines to successfully capture the massive hydration spikes driven by Sri Lanka's "Big Match" and weekend tap-rugby tournament culture.
2. **The Tuition Surge:** We multiplied Youth/Education catchments by the specific number of weekends in a given month, accurately modeling the surge of sugary beverage demand driven by weekend tuition classes.

---

## Page 4: Causal Base Logic & Demand Estimation (The Scientist)

Because the historical transaction data is right-censored ($Y_{observed} = \min(Y_{true\_demand}, C_{capacity\_limit})$), minimizing a standard loss function (like Mean Squared Error) would mathematically force the model to predict systemic stockouts. We required a methodology that unconstrained the data.

### The Censoring Flag
We mathematically isolated the constrained data. We calculated the **Coefficient of Variation (CV)** for each outlet's historical volumes. Outlets possessing a high Spatio-Temporal demand signal (e.g., strong Tuition Surge indicators) but exhibiting a CV near zero were mathematically flagged as flatlining against systemic limits (`Is_Censored = 1`).

### Route 1: The K-Means Empirical Baseline
Before deploying complex algorithms, we established a defensible, highly explainable baseline. We clustered the $\mathbb{R}^{38}$ standardized feature space into 50 behavioral peer groups using K-Means clustering. By identifying the "Stars" (uncensored shops) within each peer group, we calculated the empirical 90th percentile ceiling of unconstrained volume. We projected this peer-derived ceiling onto the censored shops to unlock an initial 721 liters of latent demand.

### Route 2: Quantile LightGBM (The Algorithmic Heavyweight)
For the final January 2026 inference, we deployed a **LightGBM Quantile Regressor**.
1. **The Asymmetric Loss:** The model was trained strictly on the uncensored historical data (`Is_Censored == 0`). We set the objective function to minimize the asymmetric pinball loss ($\alpha = 0.90$). Instead of mapping the conditional mean, the gradient boosting algorithm was forced to map the non-linear upper bounding hyperplane of the demand curve.
2. **The Future Grid Inference:** We generated a synthetic grid for January 2026, hardcoding specific future temporal constraints (e.g., exactly 9 weekend days, Jan-specific holidays, and seasonality metrics). The LightGBM model evaluated this future environment against the static spatial features to output the unconstrained 90th percentile potential.

**The Safety Floor:** To ensure absolute business stability and guard against algorithmic hallucination, the final prediction was bounded by reality: $\max(\text{LightGBM Jan 2026 Prediction}, \text{Historical Max Volume})$.

---

## Page 5: GenAI Transparency Log

Generative AI was leveraged throughout this hackathon strictly as an engineering accelerator and architectural sounding board. At no point was AI used to blindly generate unattended analysis; it was utilized to enhance and optimize our human-directed mathematical strategies.

**1. Architectural Brainstorming & Pipeline Optimization**
We utilized Large Language Models (LLMs) to spar over infrastructure trade-offs. After discussing the severe rate-limit bottlenecks of the Overpass API and OSMnx, the AI assisted in designing the highly optimized DuckDB + S3 cloud-native extraction pipeline, reducing POI acquisition time from 7 hours to 3 seconds.

**2. Boilerplate Generation & Vectorization**
We leveraged AI to aggressively refactor legacy code patterns. The AI was used to convert slow, procedural `iterrows()` data quality loops into highly optimized, fully vectorized `pandas` operations within the Silver layer, allowing the pipeline to process 2.3+ million rows in seconds.

**3. Mathematical Validation & Debugging**
The complex dual-model architecture required rigorous tuning. AI was used as an advanced debugger to correctly structure the asymmetric pinball loss parameters within the `lightgbm` framework. Furthermore, AI assisted in identifying categorical dtype conflicts during the transition between the Silver layer curation and the K-Means clustering algorithm, ensuring the mathematical integrity of the standardized $\mathbb{R}^{38}$ feature matrix.
