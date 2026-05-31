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
    
    # 1. Sales Prediction & Constraints
    p1 = f"**1. Sales Prediction & Constraints:**<br>Outlet {outlet_id} is a {tier} retailer currently constrained to a historical average of {hist_avg:,.1f} L/month. With {cooler_count} coolers deployed, the model predicts a True Market Potential of **{pot:,.1f} L/month**, representing a **{growth_pct}** expansion if supply caps are removed."
    
    # 2. Local Environment & Competition
    if is_goldmine:
        env_text = "This location operates in an **Untapped High-Traffic Zone** with zero direct competitors within a 1km radius, offering a rare monopoly advantage."
    elif comp_density > 0.05:
        env_text = "The outlet operates in a highly saturated competitive zone, meaning any investment here is highly defensive and requires aggressive market share capture."
    else:
        env_text = "The location holds a balanced spatial footprint with moderate competitive exposure."
    p2 = f"<br><br>**2. Local Environment & Competition:**<br>{env_text}"
    
    # 3. Key Model Drivers
    if top_drivers:
        driver_names = [f"{d[0]} (Score: {d[1]:.1f})" for d in top_drivers[:2] if d[1] > 0]
        if driver_names:
            driver_text = "Volume lift is heavily driven upward by strong local footfall from " + " and ".join(driver_names) + "."
        else:
            driver_text = "Volume is driven by standard demographic patterns rather than specific POI spikes."
    else:
        driver_text = "Volume is driven by standard demographic patterns."
        
    if budget > 0:
        roi = context.get("roi_per_1k", "0.0 L/1K LKR")
        rec_text = f"The optimizer allocated **LKR {budget:,}**, yielding a projected ROI of **{roi}**. Execute trade marketing to capture the **{lift:,.1f}L** volume lift."
    else:
        rec_text = "No additional budget allocated. Maintain standard distribution."
        
    p3 = f"<br><br>**3. Key Model Drivers & Recommendation:**<br>{driver_text} {rec_text}"
    
    return p1 + p2 + p3


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
2. You MUST structure your response into exactly three sections using bold HTML breaks or Markdown headers:
   **1. Sales Prediction & Constraints:** (Explain the gap between historical caps/coolers and predicted potential).
   <br><br>
   **2. Local Environment & Competition:** (Explain spatial mapping and competition intensity).
   <br><br>
   **3. Key Model Drivers & Recommendation:** (Explain which POI drivers pushed the score up/down, and conclude with the investment ROI).
3. Be confident, concise, and business-focused.
4. Bold key metrics for visual emphasis.
"""
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to rule-based engine.")
            return _generate_fallback_narrative(context)
            
    else:
        # Offline or no API key
        return _generate_fallback_narrative(context)
