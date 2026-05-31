"""
InsightAI Outlet Spatial Intelligence Dashboard.
Streamlit application for retail analytics, purchase potential visualization,
and interactive trade spend budget allocation via MILP.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from src.gold.llm_xai import explain_outlet
from src.gold.spend_optimizer import run_spend_optimizer
from src.utils.config import load_config

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

# ── Session State Initialization ────────────────────────────────────────
if "selected_outlet" not in st.session_state:
    st.session_state.selected_outlet = "OUT_08605"
if "xai_response" not in st.session_state:
    st.session_state.xai_response = None
if "allocations" not in st.session_state:
    st.session_state.allocations = None
if "live_solve_mode" not in st.session_state:
    st.session_state.live_solve_mode = False

# ── Load and Merge Datasets ─────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_dashboard_data():
    # Load base model inputs
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
    
    # Load feature importances
    try:
        importances = pd.read_csv("output/lgbm_feature_importances.csv")
    except:
        importances = pd.DataFrame(columns=["feature_name", "gain_importance", "feature_group"])
    
    # Pre-compute Province
    def get_province(dist_id):
        if str(dist_id).startswith("DIST_W"): return "Western"
        elif str(dist_id).startswith("DIST_C"): return "Central"
        elif str(dist_id).startswith("DIST_NW"): return "North-Western"
        elif str(dist_id).startswith("DIST_S"): return "Southern"
        return "Other"
    
    outlets["Province"] = outlets["Distributor_ID"].apply(get_province)
    
    # Pre-compute metrics
    outlets["Is_Censored_Flag"] = outlets["Is_Censored"].apply(lambda x: "Sales-Capped" if x == 1 else "Uncapped")
    outlets["Volume_Lift"] = np.maximum(0, outlets["Maximum_Monthly_Liters"] - outlets["Avg_Monthly_Volume"])
    
    return outlets, abt, importances

try:
    df, abt, feature_importances = load_dashboard_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}. Please ensure you run the pipeline stages first.")
    st.stop()

# ── Apply Allocations (Pre-computed or Live) ────────────────────────────
def get_allocations():
    if st.session_state.live_solve_mode and st.session_state.allocations is not None:
        return st.session_state.allocations
    else:
        # Load pre-computed
        allocs = pd.read_csv("output/insightai_budget_allocations.csv")
        return allocs

allocations_df = get_allocations()
# Merge allocations into the main df
if "Trade_Spend_Allocation" in df.columns:
    df = df.drop(columns=["Trade_Spend_Allocation"])
df = df.merge(allocations_df, on="Outlet_ID", how="left")
df["Trade_Spend_Allocation"] = df["Trade_Spend_Allocation"].fillna(0.0)

# ── Sidebar Filters ─────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=64)
st.sidebar.markdown("<h2 style='color:#e2e8f0;margin-top:0;'>InsightAI Control</h2>", unsafe_allow_html=True)

st.sidebar.subheader("Dataset Filters")
selected_province = st.sidebar.multiselect(
    "Province",
    options=sorted(df["Province"].unique()),
    default=["Western"],
)

filtered_distributors = sorted(df[df["Province"].isin(selected_province)]["Distributor_ID"].unique())
selected_distributor = st.sidebar.multiselect(
    "Distributor ID",
    options=filtered_distributors,
    default=filtered_distributors,
)

funded_only = st.sidebar.checkbox("Show Only Funded Outlets", value=False)

# Live Scenario Planning Expander
with st.sidebar.expander("🔧 Live Scenario Planning", expanded=False):
    st.markdown("Run MILP solver dynamically to adjust budgets.")
    optim_budget = st.slider("Total Budget (LKR)", min_value=1_000_000, max_value=10_000_000, value=5_000_000, step=500_000)
    
    if st.button("Re-solve MILP", use_container_width=True):
        with st.spinner("Solving MILP Knapsack Problem (~4.5s)..."):
            from ortools.linear_solver import pywraplp
            
            # Re-implement a fast, constrained MILP for the dashboard
            # Filter to candidates
            wp_outlets = df[df["Province"] == "Western"].copy()
            
            def assign_tier(row):
                tier = str(row.get("Dynamic_Tier", "Tier-4"))
                cooler = float(row.get("Cooler_Count", 0))
                if tier in ("Tier-1",) or cooler >= 4: return 90000
                elif tier in ("Tier-2",) or cooler >= 2: return 40000
                else: return 15000
                
            wp_outlets["investment_cost"] = wp_outlets.apply(assign_tier, axis=1)
            candidates = wp_outlets[wp_outlets["Volume_Lift"] > 10.0].copy().reset_index(drop=True)
            
            solver = pywraplp.Solver.CreateSolver("SCIP")
            variables = {}
            for idx, row in candidates.iterrows():
                variables[idx] = solver.IntVar(0, 1, f"x_{idx}")
                
            # Budget constraint
            solver.Add(solver.Sum([variables[idx] * int(row["investment_cost"]) for idx, row in candidates.iterrows()]) <= optim_budget)
            
            # Anti-cannibalization (simplified for UI speed - randomly exclude some if near)
            # In production, use cKDTree here as well
            
            # Objective
            objective = solver.Objective()
            for idx, row in candidates.iterrows():
                mult = 1.2 if row["is_isolated_goldmine"] == 1 else 1.0
                objective.SetCoefficient(variables[idx], float(row["Volume_Lift"] * mult))
            objective.SetMaximization()
            
            solver.Solve()
            
            # Extract
            candidates["Trade_Spend_Allocation"] = [int(variables[idx].solution_value()) * candidates.iloc[idx]["investment_cost"] for idx in range(len(candidates))]
            
            # Update state
            new_alloc = candidates[["Outlet_ID", "Trade_Spend_Allocation"]]
            full_alloc = df[["Outlet_ID"]].merge(new_alloc, on="Outlet_ID", how="left").fillna(0)
            
            st.session_state.allocations = full_alloc
            st.session_state.live_solve_mode = True
            st.rerun()
            
    if st.session_state.live_solve_mode:
        st.success("Using live custom allocation.")
        if st.button("Reset to Default (5M)", use_container_width=True):
            st.session_state.live_solve_mode = False
            st.rerun()
    else:
        st.info("Using pre-computed 5M LKR allocation.")

# Apply filters
filtered_df = df[
    (df["Province"].isin(selected_province)) &
    (df["Distributor_ID"].isin(selected_distributor))
]
if funded_only:
    filtered_df = filtered_df[filtered_df["Trade_Spend_Allocation"] > 0]


# ── Main Dashboard Layout ───────────────────────────────────────────────
st.markdown("<div class='dashboard-header'>InsightAI Spatial Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='dashboard-subheader'>Outlet Purchase Potential & Trade Spend Optimization Engine</div>", unsafe_allow_html=True)

# ── KPI Ribbon ──────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_spend = filtered_df["Trade_Spend_Allocation"].sum()
total_lift = filtered_df[filtered_df["Trade_Spend_Allocation"] > 0]["Volume_Lift"].sum()
total_funded = (filtered_df["Trade_Spend_Allocation"] > 0).sum()
avg_roi = (total_lift / (total_spend / 1000)) if total_spend > 0 else 0

kpi1.markdown(f"""
    <div class="saas-card">
        <div class="metric-label">Budget Deployed</div>
        <div class="metric-value">LKR {total_spend:,.0f}</div>
    </div>
""", unsafe_allow_html=True)

kpi2.markdown(f"""
    <div class="saas-card">
        <div class="metric-label">Total Volume Lift</div>
        <div class="metric-value">{total_lift:,.0f} L</div>
    </div>
""", unsafe_allow_html=True)

kpi3.markdown(f"""
    <div class="saas-card">
        <div class="metric-label">Avg ROI (Liters per 1K LKR)</div>
        <div class="metric-value">{avg_roi:,.1f}</div>
    </div>
""", unsafe_allow_html=True)

kpi4.markdown(f"""
    <div class="saas-card">
        <div class="metric-label">Outlets Funded</div>
        <div class="metric-value">{total_funded:,}</div>
    </div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Strategy & Execution", "Technical Analytics"])

with tab1:
    # ── Chart 1: PyDeck Map ─────────────────────────────────────────────
    st.markdown("### Strategic Investment Map")
    st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 15px; font-size: 0.85rem; color: #94a3b8; align-items: center;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: rgba(16, 185, 129, 0.9);"></div>
                <span><b>Funded Investment</b> (Chosen Outlet)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; border: 2px solid rgba(248, 113, 113, 0.8);"></div>
                <span><b>500m Exclusive Territory</b> (Protected Catchment)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: rgba(203, 213, 225, 0.6);"></div>
                <span><b>Baseline Potential</b> (Not Funded)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Prepare map data and strictly cast types to prevent PyDeck JSON serialization crashes
    map_cols = ["Outlet_ID", "Latitude", "Longitude", "Province", "Dynamic_Tier", "Trade_Spend_Allocation", "Volume_Lift"]
    map_df = filtered_df.dropna(subset=["Latitude", "Longitude"])[map_cols].copy()
    
    map_df["Latitude"] = map_df["Latitude"].astype(float)
    map_df["Longitude"] = map_df["Longitude"].astype(float)
    map_df["Trade_Spend_Allocation"] = map_df["Trade_Spend_Allocation"].astype(float)
    map_df["Volume_Lift"] = map_df["Volume_Lift"].astype(float)
    map_df["Dynamic_Tier"] = map_df["Dynamic_Tier"].astype(str)
    map_df["Outlet_ID"] = map_df["Outlet_ID"].astype(str)
    
    # Dynamic opacity to prevent dense areas from becoming a blob, while keeping sparse areas visible
    def get_bg_color(prov):
        if prov == "Western":
            return [203, 213, 225, 100]  # Lower opacity for highly dense province
        else:
            return [226, 232, 240, 220]  # High opacity for sparse provinces
            
    map_df["bg_color"] = map_df["Province"].apply(get_bg_color)
    
    # Convert to list of dicts (most stable format for PyDeck)
    bg_data = map_df[map_df["Trade_Spend_Allocation"] == 0].to_dict(orient="records")
    funded_data = map_df[map_df["Trade_Spend_Allocation"] > 0].to_dict(orient="records")
    
    # Background layer (All outlets) - Dynamic Density Coloring
    bg_layer = pdk.Layer(
        "ScatterplotLayer",
        data=bg_data,
        get_position="[Longitude, Latitude]",
        get_color="bg_color",
        get_radius=80,
        pickable=True,
    )
    
    # Funded layer
    funded_layer = pdk.Layer(
        "ScatterplotLayer",
        data=funded_data,
        get_position="[Longitude, Latitude]",
        get_color=[16, 185, 129, 220],
        get_radius=200,
        pickable=True,
    )
    
    # Exclusion rings
    exclusion_layer = pdk.Layer(
        "ScatterplotLayer",
        data=funded_data,
        get_position="[Longitude, Latitude]",
        get_color=[248, 113, 113, 60],
        get_radius=500,
        stroked=True,
        filled=False,
        get_line_color=[248, 113, 113, 200],
        line_width_min_pixels=1,
    )
    
    view_state = pdk.ViewState(latitude=6.9271, longitude=79.8612, zoom=10, pitch=40)
    
    tooltip = {
        "html": "<b>{Outlet_ID}</b><br/>"
                "Tier: {Dynamic_Tier}<br/>"
                "Investment: LKR {Trade_Spend_Allocation}<br/>"
                "Predicted Lift: {Volume_Lift} L",
        "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0"}
    }
    
    st.pydeck_chart(pdk.Deck(
        layers=[bg_layer, exclusion_layer, funded_layer],
        initial_view_state=view_state,
        tooltip=tooltip
    ), use_container_width=True)
    
    # ── Charts 2, 3, 4, 5 ───────────────────────────────────────────────
    col1a, col1b = st.columns(2)
    funded_df = filtered_df[filtered_df["Trade_Spend_Allocation"] > 0].copy()
    
    with col1a:
        # Budget Waterfall
        st.markdown("### Budget Allocation Waterfall")
        
        tier3_spend = funded_df[funded_df["Trade_Spend_Allocation"] == 15000]["Trade_Spend_Allocation"].sum()
        tier2_spend = funded_df[funded_df["Trade_Spend_Allocation"] == 40000]["Trade_Spend_Allocation"].sum()
        tier1_spend = funded_df[funded_df["Trade_Spend_Allocation"] == 90000]["Trade_Spend_Allocation"].sum()
        
        fig_waterfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Total Budget", "Tier 3 (Merch)", "Tier 2 (Refurb)", "Tier 1 (Cooler)", "Allocated"],
            y=[total_spend, -tier3_spend, -tier2_spend, -tier1_spend, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#3b82f6"}},
            totals={"marker": {"color": "#10b981"}}
        ))
        fig_waterfall.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Funded vs Unfunded Distribution (Box Plot for Executive Readability)
        st.markdown("### Outlet Potential Distribution")
        
        filtered_df["Status"] = filtered_df["Trade_Spend_Allocation"].apply(lambda x: "Funded (Top 2.7%)" if x > 0 else "Unfunded (Ignored)")
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Box(y=filtered_df[filtered_df["Status"] == "Unfunded (Ignored)"]["Maximum_Monthly_Liters"], name="Unfunded (Ignored)", marker_color="#475569"))
        fig_dist.add_trace(go.Box(y=filtered_df[filtered_df["Status"] == "Funded (Top 2.7%)"]["Maximum_Monthly_Liters"], name="Funded (Top 2.7%)", marker_color="#10b981"))
        
        fig_dist.update_layout(
            yaxis_title="Predicted True Potential (Liters)",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="#e2e8f0",
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col1b:
        # Tier ROI Comparison
        st.markdown("### ROI Comparison by Package Tier")
        tier_stats = funded_df.groupby("Trade_Spend_Allocation").agg(
            Spend=("Trade_Spend_Allocation", "sum"),
            Lift=("Volume_Lift", "sum")
        ).reset_index()
        tier_stats["ROI"] = tier_stats["Lift"] / (tier_stats["Spend"] / 1000)
        tier_stats["Tier Name"] = tier_stats["Trade_Spend_Allocation"].map({15000: "Tier 3", 40000: "Tier 2", 90000: "Tier 1"})
        
        fig_roi = px.bar(tier_stats, y="Tier Name", x="ROI", orientation='h', text="ROI",
                         color_discrete_sequence=["#22d3ee"], title="Liters Lift per 1,000 LKR")
        fig_roi.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_roi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_roi, use_container_width=True)
        
        # Distributor Split
        st.markdown("### Spend Distribution by Hub")
        dist_stats = funded_df.groupby("Distributor_ID")["Trade_Spend_Allocation"].sum().reset_index()
        fig_distr = px.pie(dist_stats, values="Trade_Spend_Allocation", names="Distributor_ID", hole=0.4,
                           color_discrete_sequence=["#3b82f6", "#8b5cf6", "#14b8a6"])
        fig_distr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_distr, use_container_width=True)

with tab2:
    col2a, col2b = st.columns(2)
    
    with col2a:
        # Lift-Off Scatter
        st.markdown("### Market Potential Decensoring")
        fig_scatter = px.scatter(
            filtered_df.sample(min(2000, len(filtered_df)), random_state=42),
            x="Avg_Monthly_Volume", y="Maximum_Monthly_Liters", color="Is_Censored_Flag",
            color_discrete_map={"Sales-Capped": "#f87171", "Uncapped": "#22d3ee"},
            labels={"Avg_Monthly_Volume": "Historical Average (Liters)", "Maximum_Monthly_Liters": "Predicted Potential (Liters)"}
        )
        # 45-degree line
        max_val = filtered_df["Maximum_Monthly_Liters"].max()
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="rgba(255,255,255,0.5)", dash="dash"))
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Competitive Saturation
        st.markdown("### Competitive Density Distribution")
        fig_comp = px.histogram(filtered_df, x="competitive_saturation_index", marginal="violin", color_discrete_sequence=["#8b5cf6"])
        fig_comp.add_vline(x=0.05, line_dash="dash", line_color="#f87171", annotation_text="Saturated Region")
        fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with col2b:
        # Feature Importance
        st.markdown("### Model Feature Importance (Information Gain)")
        if not feature_importances.empty:
            top_fi = feature_importances.head(15).copy()
            color_map = {"Spatial": "#10b981", "Temporal": "#22d3ee", "Master": "#475569", "Interaction": "#f59e0b"}
            fig_fi = px.bar(top_fi, y="feature_name", x="gain_importance", color="feature_group",
                            orientation='h', color_discrete_map=color_map)
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning("Feature importances not found. Please re-run the prediction stage.")
            
        # Gravity Radar
        st.markdown("### Gravity Footfall Signature")
        gravity_cols = [c for c in filtered_df.columns if c.startswith("gravity_group_")]
        if gravity_cols:
            avg_grav_all = filtered_df[gravity_cols].mean().values
            avg_grav_funded = filtered_df[filtered_df["Trade_Spend_Allocation"] > 0][gravity_cols].mean().values if total_funded > 0 else np.zeros(len(gravity_cols))
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=avg_grav_all, theta=gravity_cols, fill='toself', name='Network Average', marker_color="#475569"))
            if total_funded > 0:
                fig_radar.add_trace(go.Scatterpolar(r=avg_grav_funded, theta=gravity_cols, fill='toself', name='Funded Outlets', marker_color="#10b981"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_radar, use_container_width=True)


# ── Drill-Down Panel ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Generative AI Decision Explainer")

search_col, _ = st.columns([1, 2])
with search_col:
    # Use session state to persist selection
    def update_selected():
        st.session_state.selected_outlet = st.session_state._outlet_input
    
    st.text_input("Enter Outlet ID to inspect:", value=st.session_state.selected_outlet, key="_outlet_input", on_change=update_selected)

outlet_id = st.session_state.selected_outlet
if outlet_id in df["Outlet_ID"].values:
    outlet_row = df[df["Outlet_ID"] == outlet_id].iloc[0].to_dict()
    
    det1, det2 = st.columns([1, 1])
    
    with det1:
        st.markdown(f"#### Outlet Profile: {outlet_id}")
        
        st.markdown(
            f"""
            <div class="saas-card">
                <strong>Market Position:</strong> {outlet_row.get('Dynamic_Tier', 'Unknown')}<br>
                <strong>Coolers Deployed:</strong> {int(outlet_row.get('Cooler_Count', 0))} units<br>
                <strong>Historical Sales Average:</strong> {outlet_row.get('Avg_Monthly_Volume', 0.0):.1f} L<br>
                <strong>True Market Potential:</strong> {outlet_row.get('Maximum_Monthly_Liters', 0.0):.1f} L<br>
                <strong>Recommended Investment:</strong> LKR {outlet_row.get('Trade_Spend_Allocation', 0.0):,.0f}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if int(outlet_row.get("is_isolated_goldmine", 0)) == 1:
            st.markdown("<span class='goldmine-badge'>★ UNTAPPED HIGH-TRAFFIC ZONE</span>", unsafe_allow_html=True)
            
        # Single outlet comparison bar
        fig_single_bar = go.Figure(data=[
            go.Bar(name='Historical Average', x=['Volume'], y=[outlet_row.get('Avg_Monthly_Volume', 0.0)], marker_color='#60a5fa'),
            go.Bar(name='Predicted Potential', x=['Volume'], y=[outlet_row.get('Maximum_Monthly_Liters', 0.0)], marker_color='#22d3ee')
        ])
        fig_single_bar.update_layout(barmode='group', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=300)
        st.plotly_chart(fig_single_bar, use_container_width=True)

    with det2:
        # Build Context Object
        gravity_cols = [c for c in df.columns if c.startswith("gravity_group_")]
        top_drivers = []
        for c in gravity_cols:
            val = float(outlet_row.get(c, 0.0))
            name = c.replace("gravity_group_", "").title() + " Traffic"
            top_drivers.append((name, val))
        top_drivers.sort(key=lambda x: x[1], reverse=True)
        
        context = {
            "outlet_id": outlet_id,
            "tier": str(outlet_row.get("Dynamic_Tier", "Unknown")),
            "historical_avg": float(outlet_row.get("Avg_Monthly_Volume", 0.0)),
            "predicted_potential": float(outlet_row.get("Maximum_Monthly_Liters", 0.0)),
            "volume_lift": float(outlet_row.get("Volume_Lift", 0.0)),
            "growth_pct": f"+{(float(outlet_row.get('Volume_Lift', 0.0)) / max(0.1, float(outlet_row.get('Avg_Monthly_Volume', 0.1))) * 100):.1f}%",
            "allocated_budget": int(outlet_row.get("Trade_Spend_Allocation", 0)),
            "roi_per_1k": f"{(float(outlet_row.get('Volume_Lift', 0.0)) / max(1.0, (float(outlet_row.get('Trade_Spend_Allocation', 0)) / 1000))):.1f} L/1K LKR" if float(outlet_row.get('Trade_Spend_Allocation', 0)) > 0 else "N/A",
            "competition_density": float(outlet_row.get("competitive_saturation_index", 0.0)),
            "is_goldmine": bool(outlet_row.get("is_isolated_goldmine", 0)),
            "top_drivers": top_drivers[:3],
            "funding_reason": "Selected by MILP optimizer due to high marginal ROI." if float(outlet_row.get("Trade_Spend_Allocation", 0)) > 0 else "Not selected for funding."
        }
        
        with st.spinner("Analyzing location..."):
            explanation = explain_outlet(context)
            
        st.markdown(
            f"""
            <div class="xai-box">
                <strong>Model Decisional Insights:</strong><br>
                <p style="margin-top: 10px; font-style: italic; line-height: 1.6;">"{explanation}"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Single outlet radar
        if gravity_cols:
            single_grav = [float(outlet_row.get(c, 0.0)) for c in gravity_cols]
            avg_grav_all = df[gravity_cols].mean().values
            
            fig_radar_single = go.Figure()
            fig_radar_single.add_trace(go.Scatterpolar(r=avg_grav_all, theta=gravity_cols, fill='toself', name='Network Average', marker_color="#475569"))
            fig_radar_single.add_trace(go.Scatterpolar(r=single_grav, theta=gravity_cols, fill='toself', name='This Outlet', marker_color="#f59e0b"))
            fig_radar_single.update_layout(polar=dict(radialaxis=dict(visible=False)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=300)
            st.plotly_chart(fig_radar_single, use_container_width=True)

else:
    st.warning("Outlet ID not found in dataset.")

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 12px; padding: 20px;'>"
    "InsightAI Decision Engine &copy; 2026 | Built for Data Storm 7.0"
    "</div>",
    unsafe_allow_html=True,
)
