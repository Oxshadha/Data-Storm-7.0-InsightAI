"""
InsightAI Outlet Spatial Intelligence Dashboard.
Streamlit application for retail analytics, purchase potential visualization,
and interactive trade spend budget allocation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.gold.llm_xai import explain_outlet
from src.gold.spend_optimizer import run_spend_optimizer

# ── Page Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightAI | Spatial Intelligence Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium SaaS Aesthetics ──────────────────────────────
st.markdown(
    """
    <style>
    /* Dark Premium Styling */
    .stApp {
        background: linear-gradient(135deg, #0e121a 0%, #151a26 100%);
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Card design with glow effect */
    .saas-card {
        background: rgba(22, 28, 45, 0.7);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .saas-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 25px rgba(59, 130, 246, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #3b82f6;
        margin-top: 5px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Header styling */
    .dashboard-header {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(to right, #3b82f6, #60a5fa, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .dashboard-subheader {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    
    /* Isolated Goldmine tag */
    .goldmine-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #000000;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        margin-top: 10px;
    }
    
    /* XAI block */
    .xai-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #f59e0b;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load and Merge Datasets ─────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_dashboard_data():
    # Load base model inputs for demographics & historical volumes
    abt = pd.read_parquet("data/gold/model_input.parquet")
    
    # Sort and take the latest record per outlet to get static metadata
    outlets = (
        abt.sort_values(["Year", "Month"])
        .groupby("Outlet_ID")
        .last()
        .reset_index()
    )
    
    # Load coordinates
    coords = pd.read_parquet("data/silver/outlet_coordinates_clean.parquet")
    outlets = outlets.merge(coords[["Outlet_ID", "Latitude", "Longitude"]], on="Outlet_ID", how="left")
    
    # Load predictions
    preds = pd.read_csv("output/insightai_predictions.csv")
    outlets = outlets.merge(preds, on="Outlet_ID", how="inner")
    
    # Clean up column subset to save memory
    keep_cols = [
        "Outlet_ID", "Distributor_ID", "Cooler_Count", "Avg_Monthly_Volume", 
        "Dynamic_Tier", "latent_opportunity_ratio", "is_isolated_goldmine",
        "competitive_saturation_index", "total_driver_gravity", "poi_driver_catchment",
        "poi_cannibal_risk", "Latitude", "Longitude", "Maximum_Monthly_Liters",
        "gravity_group_youth", "gravity_group_leisure", "gravity_group_health"
    ]
    outlets = outlets[[c for c in keep_cols if c in outlets.columns]]
    
    # Compute province based on distributor prefix
    # DIST_W_* -> Western, DIST_C_* -> Central, DIST_NW_* -> North-Western, DIST_S_* -> Southern
    def get_province(dist_id):
        if dist_id.startswith("DIST_W"):
            return "Western"
        elif dist_id.startswith("DIST_C"):
            return "Central"
        elif dist_id.startswith("DIST_NW"):
            return "North-Western"
        elif dist_id.startswith("DIST_S"):
            return "Southern"
        return "Other"
        
    outlets["Province"] = outlets["Distributor_ID"].apply(get_province)
    
    # Load budget allocations
    allocs = pd.read_csv("output/insightai_budget_allocations.csv")
    outlets = outlets.merge(allocs, on="Outlet_ID", how="left")
    outlets["Trade_Spend_Allocation"] = outlets["Trade_Spend_Allocation"].fillna(0.0)
    
    return outlets

try:
    df = load_dashboard_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}. Please ensure you run the pipeline stages first.")
    st.stop()

# ── Sidebar Filters ─────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=64)
st.sidebar.markdown("<h2 style='color:#e2e8f0;margin-top:0;'>InsightAI Control</h2>", unsafe_allow_html=True)

st.sidebar.subheader("Dataset Filters")
selected_province = st.sidebar.multiselect(
    "Province",
    options=sorted(df["Province"].unique()),
    default=sorted(df["Province"].unique()),
)

filtered_distributors = sorted(df[df["Province"].isin(selected_province)]["Distributor_ID"].unique())
selected_distributor = st.sidebar.multiselect(
    "Distributor ID",
    options=filtered_distributors,
    default=filtered_distributors,
)

# Apply filters
filtered_df = df[
    (df["Province"].isin(selected_province)) &
    (df["Distributor_ID"].isin(selected_distributor))
]

# Interactive Budget Optimization Control
st.sidebar.markdown("---")
st.sidebar.subheader("Spend Optimization Settings")
optim_budget = st.sidebar.slider("Total Budget (LKR)", min_value=1000000, max_value=10000000, value=5000000, step=500000)
run_reoptimization = st.sidebar.button("Re-run Optimizer", use_container_width=True)

if run_reoptimization:
    # Run optimizer dynamically with new budget
    with st.spinner("Re-solving spend allocation model..."):
        # We can implement a local quick optimization solver inside Streamlit to update the state immediately
        # For simplicity and speed, let's write the solver function directly here
        G = df["Maximum_Monthly_Liters"].values - df["Avg_Monthly_Volume"].values
        G = np.maximum(0.0, G)
        
        # Keep Western Province only
        is_wp = df["Province"] == "Western"
        
        max_ratio = df["latent_opportunity_ratio"].max()
        scaled_ratio = np.log1p(df["latent_opportunity_ratio"]) / np.log1p(max_ratio)
        alpha_base = 0.00005
        alpha = alpha_base * (1.0 + 2.0 * scaled_ratio) * (1.0 + 0.5 * df["Cooler_Count"].fillna(0))
        
        # Shadow price binary search
        low, high = 1e-12, float(np.max(G * alpha)) + 1e-5
        mid = (low + high) / 2.0
        
        for _ in range(50):
            mid = (low + high) / 2.0
            with np.errstate(divide="ignore", invalid="ignore"):
                val = np.log((G * alpha) / mid) / alpha
            spends = np.clip(val, 0.0, 50000.0)
            spends[np.isnan(spends) | np.isinf(spends)] = 0.0
            spends[G <= 0.0] = 0.0
            spends[~is_wp] = 0.0
            
            tot = spends.sum()
            if abs(tot - optim_budget) < 1.0:
                break
            elif tot > optim_budget:
                low = mid
            else:
                high = mid
                
        df["Trade_Spend_Allocation"] = np.round(spends, 2)
        filtered_df = df[
            (df["Province"].isin(selected_province)) &
            (df["Distributor_ID"].isin(selected_distributor))
        ]
        st.sidebar.success(f"Success! Allocated LKR {spends.sum():,.2f}")

# ── Main Dashboard Layout ───────────────────────────────────────────────
st.markdown("<div class='dashboard-header'>InsightAI Spatial Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='dashboard-subheader'>Medallion Lakehouse Gold Enrichment & Trade Spend Optimization Framework</div>", unsafe_allow_html=True)

# ── Key Performance Metrics Row ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_pot = filtered_df["Maximum_Monthly_Liters"].sum()
    st.markdown(
        f"""
        <div class="saas-card">
            <div class="metric-label">Total Projected Potential</div>
            <div class="metric-value">{total_pot:,.1f} L</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    total_hist = filtered_df["Avg_Monthly_Volume"].sum()
    st.markdown(
        f"""
        <div class="saas-card">
            <div class="metric-label">Historical Baseline</div>
            <div class="metric-value">{total_hist:,.1f} L</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    growth_gap = max(0.0, total_pot - total_hist)
    st.markdown(
        f"""
        <div class="saas-card">
            <div class="metric-label">Latent Opportunity Gap</div>
            <div class="metric-value">{growth_gap:,.1f} L</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    total_spend = filtered_df["Trade_Spend_Allocation"].sum()
    st.markdown(
        f"""
        <div class="saas-card">
            <div class="metric-label">Trade Spend Allocation</div>
            <div class="metric-value">LKR {total_spend:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Section 1: Spatial Map and Demographics ──────────────────────────────
st.subheader("🌐 Network Spatial Distribution & Local Markets")
m_col1, m_col2 = st.columns([2, 1])

with m_col1:
    # Build Map
    map_df = filtered_df.dropna(subset=["Latitude", "Longitude"]).sample(min(3000, len(filtered_df)), random_state=42)
    fig_map = px.scatter(
        map_df,
        x="Longitude",
        y="Latitude",
        color="Dynamic_Tier",
        size="Maximum_Monthly_Liters",
        hover_name="Outlet_ID",
        hover_data=["Distributor_ID", "Avg_Monthly_Volume", "Trade_Spend_Allocation"],
        color_discrete_map={"Tier-1": "#3b82f6", "Tier-2": "#60a5fa", "Tier-3": "#94a3b8", "Tier-4": "#475569"},
        title="Spatial Outlet Map (Sampled top 3,000 outlets for speed)",
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)

with m_col2:
    # Dynamic Tier Distribution
    tier_counts = filtered_df["Dynamic_Tier"].value_counts().reset_index()
    tier_counts.columns = ["Dynamic_Tier", "count"]
    fig_tier = px.pie(
        tier_counts,
        values="count",
        names="Dynamic_Tier",
        hole=0.4,
        color_discrete_sequence=["#3b82f6", "#60a5fa", "#a5b4fc", "#1e293b"],
        title="Outlet Breakdown by Dynamic Tier",
    )
    fig_tier.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_tier, use_container_width=True)

# ── Section 2: Outlet Detail & XAI Drilldown ────────────────────────────
st.markdown("---")
st.subheader("🔍 Single Outlet Drilldown & GenAI Explainability")

search_outlet = st.text_input("Enter Outlet ID to inspect (e.g. OUT_08605, OUT_04708, OUT_02116):", "OUT_08605")

if search_outlet not in df["Outlet_ID"].values:
    st.warning(f"Outlet ID {search_outlet} not found in the dataset. Showing OUT_08605 as fallback.")
    search_outlet = "OUT_08605"

outlet_row = df[df["Outlet_ID"] == search_outlet].iloc[0]

det1, det2 = st.columns([1, 1])

with det1:
    st.markdown(f"### Outlet Metrics: {search_outlet}")
    
    st.markdown(
        f"""
        <div class="saas-card">
            <strong>Distributor:</strong> {outlet_row['Distributor_ID']}<br>
            <strong>Province:</strong> {outlet_row['Province']}<br>
            <strong>Dynamic Tier:</strong> {outlet_row['Dynamic_Tier']}<br>
            <strong>Cooler Count:</strong> {int(outlet_row['Cooler_Count'])} units<br>
            <strong>Historical Average Volume:</strong> {outlet_row['Avg_Monthly_Volume']:.1f} Liters/month<br>
            <strong>January 2026 Potential:</strong> {outlet_row['Maximum_Monthly_Liters']:.1f} Liters/month<br>
            <strong>Allocated Trade Spend:</strong> LKR {outlet_row['Trade_Spend_Allocation']:,.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if int(outlet_row["is_isolated_goldmine"]) == 1:
        st.markdown("<span class='goldmine-badge'>★ ISOLATED GOLDMINE</span>", unsafe_allow_html=True)

with det2:
    st.markdown("### 🤖 Generative AI Explainability (XAI)")
    
    # Generate XAI Explanation
    with st.spinner("Generating business explanation..."):
        explanation = explain_outlet(dict(outlet_row))
        
    st.markdown(
        f"""
        <div class="xai-box">
            <strong>Model Decisional Insights:</strong><br>
            <p style="margin-top: 10px; font-style: italic; line-height: 1.6;">"{explanation}"</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Opportunity Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=outlet_row["latent_opportunity_ratio"],
        title={'text': "Latent Opportunity Ratio"},
        gauge={
            'axis': {'range': [None, 300]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(30, 41, 59, 0.5)"},
                {'range': [50, 150], 'color': "rgba(59, 130, 246, 0.2)"},
                {'range': [150, 300], 'color': "rgba(245, 158, 11, 0.2)"}
            ],
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 12px; padding: 20px;'>"
    "InsightAI Decision Engine &copy; 2026 | Built for Data Storm 7.0"
    "</div>",
    unsafe_allow_html=True,
)
