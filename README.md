# Data Storm 7.0 — Estimating Unconstrained Latent Demand

> **Team:** InsightAI
> **Objective:** Predict the absolute maximum latent monthly volume capacity for ~20,000 FMCG outlets in Sri Lanka for January 2026, mathematically unconstrained by historical supply-side bottlenecks.

## 🌟 The "InsightAI" Philosophy
Standard Time-Series models fail on this problem because they naively forecast historical supply chain bottlenecks. If a shop requested 500 Liters but the distributor only delivered 100 Liters, an ARIMA model predicts 100. 

Our pipeline bypasses this by utilizing a **Spatio-Temporal Causal Framework**. We built a Medallion Architecture (Bronze $\rightarrow$ Silver $\rightarrow$ Gold) that mathematically identifies censored outlets (flatlining volume), maps external Point-of-Interest (POI) drivers using Overture Maps, and uses **Quantile Regression ($\alpha=0.90$)** with Early Stopping to project true latent capacity.

---

## 🏗️ Architecture: The Lakehouse Pipeline

Our codebase strictly enforces a Lakehouse layer separation to ensure data hygiene, reproducibility, and analytical rigor.

### Directory Structure
```text
📦 Data-Storm-7.0-InsightAI
├── 📂 config/                  # Pipeline configurations (YAML)
├── 📂 data/
│   ├── 📂 bronze/              # Raw ingestion (zero transformations)
│   ├── 📂 silver/              # Cleaned Data + Quarantined DQ Rejections
│   └── 📂 gold/                # Feature-engineered Analytical Base Table (ABT)
├── 📂 src/
│   ├── 📂 bronze/              # CSV -> Parquet ETL scripts
│   ├── 📂 silver/              # Forensic Cleaning & Anomaly Detection (Ghost Exorcism)
│   ├── 📂 gold/                # Catchment Engineering & Collaborative Filtering
│   └── 📂 model/               # K-Means Peer Benchmarking & LightGBM Quantile Models
├── 📂 notebooks/               # Analytical Pipeline (EDA -> Proof of Unconstraining)
├── 📂 output/                  # Final Predictions CSV, Plots, and LaTeX Report
├── 📄 requirements.txt         # Environment dependencies
└── 📄 run_pipeline.py          # Master Orchestrator (Executes End-to-End)
```

---

## 🚀 How to Run the End-to-End Pipeline

We have automated the entire data cleaning, feature engineering, and modeling process into a single orchestrator script. 

### 1. Setup the Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Verify Data Availability
Ensure that the original `transactions_history_final.csv` and all other provided CSVs are located in the original `Refernce Resources/` or `data/raw/` directory as specified in `config/pipeline_config.yaml`.

### 3. Execute the Master Pipeline
```bash
# Run the complete pipeline (Bronze -> Silver -> Gold -> Model)
python run_pipeline.py
```
*You can also run specific stages individually:*
```bash
python run_pipeline.py --stage bronze
python run_pipeline.py --stage silver
python run_pipeline.py --stage gold
python run_pipeline.py --stage predict
```

---

## 🔬 The Analytical Pipeline: From EDA to Prediction

We generated five primary Jupyter Notebooks that document our entire analytical journey. **Judges are highly encouraged to review these notebooks in order:**

1. **`01_Data_Forensics.ipynb`**
   * **Purpose:** Proves our data hygiene process.
   * **Key Insight:** Visualizes the "Ghost Exorcism" where extreme SFA anomalies and System Upload Retries were quarantined. Demonstrates the detection of the "Master Data Decay" trap.
2. **`02_SpatioTemporal_EDA.ipynb`**
   * **Purpose:** Initial geographic exploration.
   * **Key Insight:** Explores how Overture Maps POIs correlate with historical volume before applying advanced algorithms.
3. **`03_The_Scientific_Proof.ipynb`**
   * **Purpose:** Strict Temporal Validation.
   * **Key Insight:** Proves we didn't use a random Train/Test split (which leaks future data). We used an Out-of-Time (OOT) holdout validation and measured the Quantile Pinball Loss.
4. **`04_Model_Interpretability.ipynb`**
   * **Purpose:** Proves our Spatio-Temporal interactions work.
   * **Key Insight:** Shows the LightGBM Information Gain charts, proving that engineered features like `Dynamic_Tier` and `Has_Youth_Catchment` drove the model, rather than overfitting on raw `Outlet_ID`s.
5. **`05_Final_Evaluation.ipynb`**
   * **Purpose:** Proves the unconstraining mathematical logic.
   * **Key Insight:** Displays the critical Scatter Plot where historical constrained actuals form a ceiling, but our LightGBM predictions "lift off" and float above the 45-degree line to capture latent demand.

---

## 🧠 Key Innovations & Causal Base Logic

### 1. Collaborative Filtering (Curing Portfolio Imbalance)
If a shop historically sells only Cola, but its geographic and tier-based peers sell Ginger Beer, the shop suffers from a structural Portfolio Imbalance. `src/gold/collaborative_filter.py` defines these Peer Groups, identifies "Core SKUs" (>50% peer penetration), and synthetically imputes the latent demand for missing portfolios.

### 2. Spatio-Temporal Catchments (Overture Maps)
We queried Overture Maps Foundation's S3 Parquet buckets via DuckDB bounding-boxes to extract precise coordinates for Universities, Parks, and Stadiums in Sri Lanka. We generated dynamic interaction features:
* `Tuition_Weekend_Surge`: (Youth POIs $\times$ Weekends)
* `Tourist_Peak_Multiplier`: (Leisure POIs $\times$ High Season)

### 3. LightGBM Quantile Regressor ($p90$) with Early Stopping
We trained a LightGBM regressor strictly on historical "Star" shops (Uncensored shops). To prevent overfitting, we utilized the final historical month as a validation set and triggered **Early Stopping**. We optimized the model on an upper-bound Quantile Loss Function ($\alpha=0.90$) to map the absolute maximum capacity for January 2026.

---

## 📊 Final Deliverables

1. **Predictions:** `output/insightai_predictions.csv` containing the final `[Outlet_ID, Maximum_Monthly_Liters]`.
2. **Technical Report:** `output/InsightAI_Final_Report.tex` (and compiled PDF in Overleaf).
3. **Codebase:** This completely reproducible, modular repository.

**Thank you for reviewing Team InsightAI's submission!**
