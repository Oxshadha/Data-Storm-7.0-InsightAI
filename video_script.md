# Data Storm 7.0 - InsightAI Demonstration Video Script

**Target Duration:** 6 Minutes (approx. 850 words)
**Application:** InsightAI Spatial Intelligence Dashboard
**Objective:** A seamless, comprehensive walkthrough of the AI pipeline, business logic, math, and technical proof.

---

## 🎬 1. Introduction & The Problem Statement (0:00 - 0:45)
**Visual:** *Speaker on camera or title slide, transitioning to the main dashboard screen.*

**Script:**
"Hello judges, and welcome to Team InsightAI's submission for Data Storm 7.0. Today, we are presenting our Spatial Intelligence Engine. 

The core challenge in retail expansion is that historical sales data is notoriously flawed—it is capped by supply constraints. If an outlet requested 500 liters but was only supplied 100, traditional models only predict 100. 

To solve this, we built a Spatio-Temporal Framework using Overture Maps and Quantile Regression to uncover the *unconstrained* latent demand. We then feed that demand into a Mixed-Integer Linear Programming (MILP) optimizer to allocate our exact Trade Spend. This dashboard translates that complex mathematics into a dynamic business strategy."

---

## 📈 2. KPIs & The Underlying Math (0:45 - 1:15)
**Visual:** *Cursor highlights the top KPI ribbon.*

**Script:**
"At the top, our KPI ribbon provides the executive summary of our 5 Million LKR budget deployment. We don't just show volume; we translate it into financial impact. 

How does the math work? 
Our **Estimated Monthly Revenue** is calculated by taking the AI-predicted Volume Lift and multiplying it by an assumed standard unit price of 250 Rupees per liter. 
Our **Payback Period** is calculated by simply dividing the Total Budget Deployed by that Monthly Revenue. In this baseline scenario, our optimization strategy pays for itself in exactly 1.6 months, proving extreme capital efficiency."

---

## 🗺️ 3. The Map & Business Strategy (1:15 - 2:15)
**Visual:** *Cursor moves over the PyDeck map, then scrolls through the 6 Business Strategy charts.*

**Script:**
"The centerpiece of our execution is the Strategic Investment Map. 
The **gray dots** represent the unconstrained potential of the entire network. The **large green nodes** are the specific outlets our MILP algorithm selected for funding. The **red rings** around them represent a strict 500-meter exclusion radius, mathematically preventing self-cannibalization.

Scrolling down, we present exactly *how* the budget was optimized across 6 key strategic views:
1. **Budget Allocation Waterfall**: Breaks down the 5M LKR across our 15K, 40K, and 90K tiers.
2. **ROI Comparison by Package**: Proves that 15K and 40K tiers yield the highest marginal return per rupee.
3. **Strategic ROI**: This bar chart confirms our funded outlets achieve massively higher average volume lifts than unfunded ones.
4. **Outlet Potential Distribution**: The box plot visually proves the algorithm successfully shifted funding rightward into the highest-potential segment.
5. **Spend Distribution by Hub**: Shows exactly which distributor regions receive the capital injection.
6. **Targeted Expansion**: This pie chart highlights that we balance 85% of our budget on Strategic Urban Hubs while carving out 15% to capture high-margin Isolated Goldmines."
---

## 🎛️ 4. The Executive Stress-Test Engine (2:15 - 3:30)
**Visual:** *Cursor moves to the left sidebar. Clicks the 'Province' dropdown to demonstrate regional filtering. Then, adjusts the Budget slider to 3M and Demand Sensitivity to -10%. Clicks 'Run Simulation'.*

**Script:**
"Our sidebar allows regional managers to filter strategies down to specific Provinces or Distributors. But the true power of this dashboard is the **Executive Stress-Test Engine**.

Business environments change rapidly. What if headquarters slashes our budget from 5 Million to 3 Million? And what if a macro-economic shock drops consumer demand by 10%? 

By adjusting these sliders and clicking *Run Simulation*, the dashboard doesn't just filter data—it triggers a live, in-browser recalculation of the **MILP Knapsack** solver. 
Look at the KPI ribbon—it now displays red and green deltas, instantly quantifying our volume loss. Notice how the Map dynamically redraws, and the Waterfall chart shifts its strategy. The AI has instantly retreated to a defensive posture, abandoning high-risk 90K assets to protect our core 15K and 40K high-ROI monopolies."

---

## 🤖 5. Generative AI Drill-Down (3:30 - 4:45)
**Visual:** *Scrolls to the XAI Drill-Down panel. Highlights the Bullet Gauge and Radar Chart. Types an outlet ID for a 90K outlet, then a 15K outlet.*

**Script:**
"A major challenge with AI is adoption by non-technical people. A field sales rep doesn't understand what a 'MILP Knapsack solver' is—they just need to know *why* a shop got funded. Our XAI panel solves this.

When we select an outlet, the **Bullet Gauge** immediately compares its historical volume limit against its newly predicted True Potential, showing exactly how much growth we expect.

If we search for a **90K Tier Outlet**, look at the **Radar Chart**. We normalized the axes into Percentile Ranks (0 to 100). The dashed grey circle is the 50th percentile network median. This outlet's orange polygon spikes massively outside the grey circle in 'Transit' and 'Youth' demographics. 

If we switch to a **15K Tier Outlet**, the polygon shrinks inside the grey line—it has low absolute footfall. Why was it funded? Look at the Generative AI brief on the left, powered by Google Gemini. It explains that this outlet is an 'Isolated Goldmine'—it may have low traffic, but it holds a 100% monopoly over its local radius. Gemini automatically translates our complex MILP logic directly into a simple, human sales pitch."

---

## 🔬 6. Technical Analytics & Proof (4:45 - 6:00)
**Visual:** *Expands a 'Data Science Note' below a chart, then clicks the 'Technical Analytics' tab. Scrolls through the 4 charts.*

**Script:**
"You may have noticed the collapsible **'Data Science Notes'** and **'Interactive Tips'** beneath our charts. We designed this UI to serve two personas perfectly: executives get a clean, high-level view, while data science judges can expand those notes to review our mathematical methodology.

For ultimate technical proof, we built a dedicated Analytics tab:

1. **Market Potential Decensoring:** This scatter plot plots historical sales against our AI's predicted potential. Notice the 45-degree identity line. Every dot floating above that line is visual proof that our Quantile Regressor successfully broke through historical distributor quotas to unlock latent demand.
2. **Competitive Saturation:** This Empirical CDF plot proves our Cannibalization constraints. The red threshold line dynamically sits at the 85th percentile, mathematically proving exactly what portion of the network is hyper-competitive, which our MILP solver actively avoids.
3. **The Cannibalization Paradox:** The red OLS trendline on this scatter plot clearly slopes downward. It statistically proves that as Urban Density increases, potential volume drops due to hyper-competition. This mathematically justifies funding rural monopolies.
4. **Feature Importance:** Our LightGBM information gain chart confirms our entire philosophy—spatial catchment and temporal seasonality vastly outperform static master data.

In 6 minutes, we've taken raw, constrained data, uncovered true demand, optimized it mathematically, and deployed it into a dynamic, executive-grade engine. Thank you for your time."
