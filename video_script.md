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
Our **Estimated Monthly Revenue** is calculated by taking the AI-predicted Volume Lift and multiplying it by an average gross margin of 100 Rupees per liter. 
Our **Payback Period** is calculated by simply dividing the Total Budget Deployed by that Monthly Revenue. In this baseline scenario, our optimization strategy pays for itself in exactly 1.6 months, proving extreme capital efficiency."

---

## 🗺️ 3. The Map & Business Strategy (1:15 - 2:00)
**Visual:** *Cursor moves over the PyDeck map, then scrolls to the Waterfall and ROI charts.*

**Script:**
"The centerpiece of our execution is the Strategic Investment Map. 
The **gray dots** represent the unconstrained potential of the entire network. The **large green nodes** are the specific outlets our MILP algorithm selected for funding. Notice the **red rings** around them—these represent a strict 500-meter exclusion radius. Our algorithm mathematically prevents funding multiple outlets within the same catchment area, entirely eliminating self-cannibalization.

Scrolling down, we have our **Budget Allocation Waterfall**. This breaks down the 5 Million LKR across three strategic tiers: 15K POSM micro-investments for isolated regions, 40K visibility upgrades, and 90K core asset injections for high-traffic hubs. Beside it, the ROI Comparison chart proves that our algorithm heavily favors the 15K and 40K tiers because they yield the highest marginal return per rupee spent."

---

## 🎛️ 4. The Executive Stress-Test Engine (2:00 - 3:15)
**Visual:** *Cursor moves to the left sidebar. Adjusts the Budget slider to 3M and Demand Sensitivity to -10%. Clicks 'Run Simulation'.*

**Script:**
"Our sidebar allows regional managers to filter strategies by Province or specific Distributors. But the true power of this dashboard is the **Executive Stress-Test Engine**.

Business environments change rapidly. What if headquarters slashes our budget from 5 Million to 3 Million? And what if a macro-economic shock drops consumer demand by 10%? 

By adjusting these sliders and clicking *Run Simulation*, the dashboard doesn't just filter data—it triggers a live, in-browser recalculation of the MILP Knapsack solver. 
Look at the KPI ribbon—it now displays red and green deltas, instantly quantifying our volume loss. Notice how the Map dynamically redraws, and the Waterfall chart shifts its strategy. The AI has instantly retreated to a defensive posture, abandoning high-risk 90K assets to protect our core 15K and 40K high-ROI monopolies."

---

## 🤖 5. Generative AI Drill-Down (3:15 - 4:30)
**Visual:** *Scrolls to the XAI Drill-Down panel. Types an outlet ID for a 90K funded outlet, then searches for a 15K funded outlet.*

**Script:**
"For field sales managers, we need to explain *why* specific outlets get funding. Our XAI Drill-Down panel provides complete transparency.

If we search for a **90K Tier Outlet**, look at the Radar Chart. We normalized the axes into Percentile Ranks (0 to 100). The dashed grey circle is the 50th percentile network median. The orange polygon is this outlet. You can see massive spikes breaking outside the grey circle in 'Transit' and 'Youth' demographics. 

If we switch to a **15K Tier Outlet**, the polygon shrinks inside the grey line—it has low absolute footfall. Why was it funded? Read the Generative AI brief on the left, powered by Google Gemini. It reads the mathematical context and explains that this outlet is an 'Isolated Goldmine'—it may have low traffic, but it holds a 100% monopoly over its local radius, making it a high-ROI target. Gemini automatically translates our MILP logic into a human sales pitch."

---

## 🔬 6. Technical Analytics & Proof (4:30 - 6:00)
**Visual:** *Clicks the 'Technical Analytics' tab. Scrolls through the 4 charts.*

**Script:**
"Finally, for the judges evaluating our data science rigor, we present our Technical Analytics.

1. **Market Potential Decensoring:** This scatter plot plots historical sales against our AI's predicted potential. Notice the 45-degree identity line. Every dot floating above that line is visual proof that our Quantile Regressor successfully broke through historical distributor quotas to unlock latent demand.
2. **Competitive Saturation:** This Empirical CDF plot proves our Cannibalization constraints. The red threshold line dynamically sits at the 85th percentile, mathematically proving exactly what portion of the network is hyper-competitive, which our MILP solver actively avoids.
3. **The Cannibalization Paradox:** This scatter plot is our most counter-intuitive finding. The red OLS trendline clearly slopes downward. It statistically proves that as Spatial Driver Gravity (Urban Density) increases, potential volume actually drops due to hyper-competition. This justifies our strategy of funding rural monopolies.
4. **Feature Importance:** Our LightGBM information gain chart confirms our entire philosophy—spatial catchment and temporal seasonality vastly outperform static master data.

In 6 minutes, we've taken raw, constrained data, uncovered true demand, optimized it mathematically, and deployed it into a dynamic, executive-grade engine. Thank you for your time."
