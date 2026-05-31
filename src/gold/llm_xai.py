"""
Gold Layer — LLM Explainable AI (XAI) Module.

Generates human-readable, business-oriented narratives explaining model decisions
for individual retail outlets. Uses a hybrid design:
  1. Real LLM connection using google-generativeai (Gemini API) if GEMINI_API_KEY is configured.
  2. Professional, dynamic rule-based narrative engine as a fallback for offline local running.

Usage:
    from src.gold.llm_xai import explain_outlet
    explanation = explain_outlet(context)
"""

import os
from src.utils.logger import get_logger

logger = get_logger("gold.llm_xai")

# Try importing the Gemini API client
HAS_GEMINI = False
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _generate_fallback_narrative(context: dict) -> str:
    """
    Generates a highly-contextual, professional business narrative based on the pre-computed context object.
    Ensures the Streamlit app works out-of-the-box without API keys.
    """
    outlet_id = context.get("outlet_id", "Unknown Outlet")
    tier = context.get("tier", "Unknown")
    hist_avg = context.get("historical_avg", 0.0)
    pot = context.get("predicted_potential", 0.0)
    lift = context.get("volume_lift", 0.0)
    growth_pct = context.get("growth_pct", "+0.0%")
    budget = context.get("allocated_budget", 0)
    is_goldmine = context.get("is_goldmine", False)
    comp_density = context.get("competition_density", 0.0)
    top_drivers = context.get("top_drivers", [])
    
    cooler_count = context.get("cooler_count", 0)
    
    # 1. Why score
    p1 = f"**Why the model assigned this specific score:**<br>Outlet {outlet_id} operates as a {tier} retailer, which historically constrained its performance to a plateau of {hist_avg:,.1f} L/month. However, when evaluating the underlying spatial demographics and current deployment of {cooler_count} coolers, our Decensoring Model identified a significant suppressed demand. Removing artificial supply caps reveals a True Market Potential of **{pot:,.1f} L/month**, representing a massive **{growth_pct}** expansion opportunity that the current distribution network is missing."
    
    # 2. Local conditions and constraints
    if is_goldmine:
        env_text = "This location is uniquely positioned in an **Untapped High-Traffic Zone**. It enjoys a rare monopoly advantage with zero direct competitors within a 1km radius. This lack of competitive density means any trade marketing spend here will not be cannibalized by neighboring outlets, allowing you to capture 100% of the localized demand."
    elif comp_density > 0.05:
        env_text = "The outlet operates in a highly saturated competitive zone. The local environment is densely packed with rival retailers, meaning this investment is highly defensive. Trade marketing here is required not just to grow, but to aggressively protect and capture market share from competitors in the immediate vicinity."
    else:
        env_text = "The location holds a balanced spatial footprint with moderate competitive exposure. It is neither completely isolated nor hyper-competitive, representing a stable environment for consistent, targeted marketing deployments."
    p2 = f"<br><br>**How local conditions and constraints influenced the result:**<br>{env_text}"
    
    # 3. Factors increasing/decreasing prediction
    if top_drivers:
        driver_names = [f"{d[0]} (Score: {d[1]:.1f})" for d in top_drivers[:2] if d[1] > 0]
        if driver_names:
            driver_text = "The volume prediction is heavily driven upward by strong localized footfall, specifically from " + " and ".join(driver_names) + ". These spatial anchors guarantee a continuous influx of high-intent consumers."
        else:
            driver_text = "The volume is driven primarily by baseline demographic stability rather than specific high-impact points of interest."
    else:
        driver_text = "The volume is driven primarily by baseline demographic stability rather than specific high-impact points of interest."
        
    if budget > 0:
        roi = context.get("roi_per_1k", "0.0 L/1K LKR")
        rec_text = f"Consequently, the optimization engine has confidently allocated **LKR {budget:,}** to this outlet. This is projected to yield an exceptional ROI of **{roi}**. We recommend immediate execution of trade marketing activities to fully capture the **{lift:,.1f}L** latent volume lift."
    else:
        rec_text = "Given the current constraints and ROI thresholds, no additional budget has been allocated. We recommend maintaining standard distribution and re-evaluating in the next quarter."
        
    p3 = f"<br><br>**Which factors increased or decreased the prediction:**<br>{driver_text} {rec_text}"
    
    # 4. Confidence & Risk
    if budget > 0:
        risk_profile = "Low Risk / High Reward" if is_goldmine else "Moderate Risk / High Reward"
        risk_text = f"This volume prediction is statistically bound using a **p90 Quantile Confidence Interval**. This mathematically guarantees that we are 90% confident the outlet's true ceiling lies beneath {pot:,.1f} L/month. Given the robust data signals and local competition constraints, this allocation represents a **{risk_profile}** investment."
    else:
        risk_text = f"This volume prediction is statistically bound using a **p90 Quantile Confidence Interval**. Because the projected marginal ROI does not meet the optimizer's rigorous threshold, deploying capital here currently carries an **Unfavorable Risk Profile** compared to other network alternatives."
        
    p4 = f"<br><br>**Decision Confidence & Risk Assessment:**<br>{risk_text}"
    
    return p1 + p2 + p3 + p4


def explain_outlet(context: dict) -> str:
    """
    Generates a natural language explanation of the model's prediction for an outlet.
    
    Parameters
    ----------
    context : dict containing comprehensive business metrics, spatial drivers, and financial ROI.
    
    Returns
    -------
    str : Narrative explanation
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if HAS_GEMINI and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Format top drivers for prompt
            top_drivers_str = "\n".join([f"- {d[0]}: {d[1]:.2f}" for d in context.get("top_drivers", [])])
            
            # Construct a clear, structured prompt
            prompt = f"""
You are an AI Trade Marketing Director for a beverage company in Sri Lanka.
Analyze the provided OUTLET DATA PROFILE.
Generate a structured, 3-part business explanation for the Regional Sales Manager explaining WHY this outlet was assigned its specific prediction and budget.

OUTLET DATA PROFILE:
- Outlet ID: {context.get('outlet_id', 'Unknown')}
- Market Position: {context.get('tier', 'Unknown')}
- Cooler Count: {context.get('cooler_count', 0)}
- Historical Sales Average: {context.get('historical_avg', 0.0):.1f} Liters/month
- True Market Potential: {context.get('predicted_potential', 0.0):.1f} Liters/month
- Volume Lift: {context.get('volume_lift', 0.0):.1f} Liters/month ({context.get('growth_pct', '0%')})
- Recommended Investment: LKR {context.get('allocated_budget', 0):,}
- Expected ROI: {context.get('roi_per_1k', '0.0 L/1K LKR')}
- Untapped High-Traffic Zone (Goldmine): {'Yes' if context.get('is_goldmine', False) else 'No'}
- Competition Density: {context.get('competition_density', 0.0):.4f}

Top Footfall Drivers:
{top_drivers_str}

Decision Engine Reason: {context.get('funding_reason', 'N/A')}

RULES:
1. Do NOT use data science jargon (no 'SHAP', 'MILP', 'Decensoring').
2. You MUST structure your response into exactly four sections using bold HTML breaks or Markdown headers exactly as follows:
   **Why the model assigned this specific score:** (Explain the gap between historical caps/coolers and predicted potential).
   <br><br>
   **How local conditions and constraints influenced the result:** (Explain spatial mapping and competition intensity).
   <br><br>
   **Which factors increased or decreased the prediction:** (Explain which POI drivers pushed the score up/down, and conclude with the investment ROI).
   <br><br>
   **Decision Confidence & Risk Assessment:** (Explain that the prediction is statistically bound using a p90 Quantile Confidence Interval, meaning the AI is 90% confident the true ceiling is beneath the predicted potential. Assess if the investment is Low/Moderate/High risk based on competition and ROI).
3. Write in a fluid, professional, and detailed business narrative style. Provide 2-3 comprehensive sentences per section. Do NOT write brief, robotic one-liners.
4. Explicitly explain the business mechanics (e.g., how the specific footfall drivers physically bring in customers, or how the lack of competition guarantees market share).
5. Bold key metrics for visual emphasis.
"""
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to rule-based engine.")
            return _generate_fallback_narrative(context)
            
    else:
        # Offline or no API key
        return _generate_fallback_narrative(context)
