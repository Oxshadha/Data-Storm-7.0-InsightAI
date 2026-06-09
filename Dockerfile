# ============================================================
# InsightAI Spatial Intelligence Dashboard — Production Image
# ============================================================
# Multi-layer Docker build optimized for Streamlit + OR-Tools
# Target: AWS App Runner (1 vCPU / 2GB RAM)
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# ── System Dependencies ──────────────────────────────────────
# geopandas needs GDAL/GEOS/PROJ; ortools needs build-essential
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python Dependencies (cached layer) ──────────────────────
COPY requirements.txt .
RUN sed -i '/jupyter/d; /nbformat/d' requirements.txt && pip install --no-cache-dir -r requirements.txt

# ── Application Code ────────────────────────────────────────
COPY app.py .
COPY src/ src/
COPY config/ config/

# ── Pre-computed Data Files (runtime dependencies only) ──────
# These are the ONLY data files app.py reads at startup:
#   - data/gold/model_input.parquet      (13MB — ABT features)
#   - data/silver/outlet_coordinates_clean.parquet (0.5MB — geo coords)
#   - output/insightai_predictions.csv   (380KB — model predictions)
#   - output/insightai_budget_allocations.csv (108KB — MILP results)
#   - output/lgbm_feature_importances.csv (3KB — feature importances)
COPY data/gold/model_input.parquet data/gold/model_input.parquet
COPY data/silver/outlet_coordinates_clean.parquet data/silver/outlet_coordinates_clean.parquet
COPY output/insightai_predictions.csv output/insightai_predictions.csv
COPY output/insightai_budget_allocations.csv output/insightai_budget_allocations.csv
COPY output/lgbm_feature_importances.csv output/lgbm_feature_importances.csv

# ── Streamlit Configuration ─────────────────────────────────
RUN mkdir -p /root/.streamlit
RUN printf '[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
maxUploadSize = 5\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "dark"\n' > /root/.streamlit/config.toml

# ── Expose & Health Check ────────────────────────────────────
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ───────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
