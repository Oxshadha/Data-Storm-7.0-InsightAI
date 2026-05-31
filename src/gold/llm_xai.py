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
    
    # Paragraph 1: Potential and Tier
    p1 = (
        f"Outlet {outlet_id} is classified as a {tier} retailer with a True Market Potential of "
        f"{pot:,.1f} Liters/month, representing a {growth_pct} expansion over its historical baseline. "
    )
    
    # Paragraph 2: Drivers (Spatial and Competitive)
    drivers = []
    if is_goldmine:
        drivers.append("operates in an Untapped High-Traffic Zone with zero direct competitors within 1km")
    elif comp_density > 0.05:
        drivers.append("operates in a highly saturated competitive zone, suggesting that promotions must target market share capture")
    else:
        drivers.append("has a balanced spatial footprint with moderate competitive exposure")
        
    if top_drivers:
        driver_names = [f"{d[0]} (Score: {d[1]:.1f})" for d in top_drivers[:2] if d[1] > 0]
        if driver_names:
            drivers.append("benefits from strong local footfall driven by " + " and ".join(driver_names))
            
    if drivers:
        p2 = f"This potential is unlocked because the outlet " + ", and ".join(drivers) + ". "
    else:
        p2 = ""
        
    # Paragraph 3: Assets & Recommendations
    if budget > 0:
        roi = context.get("roi_per_1k", "0.0 L/1K LKR")
        p3 = f"Recommendation: The AI has allocated LKR {budget:,} yielding a projected ROI of {roi}. Execute trade marketing activities to capture the {lift:,.1f}L volume lift."
    else:
        p3 = "Recommendation: No additional budget allocated. Maintain standard distribution and monitor performance."
        
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
Write a 3-sentence explanation for the Regional Sales Manager explaining exactly 
WHY this specific outlet was chosen for budget allocation (or why it has high potential).

OUTLET DATA PROFILE:
- Outlet ID: {context.get('outlet_id', 'Unknown')}
- Market Position: {context.get('tier', 'Unknown')}
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
1. Do NOT use data science jargon (no 'SHAP', 'MILP', 'Decensoring', 'Gravity Model').
2. Synthesize three angles: (a) Financial ROI, (b) Spatial/footfall drivers, (c) Competition gap.
3. Be confident, concise, and business-focused.
4. If the outlet is an "Untapped High-Traffic Zone" (goldmine), emphasize the strategic opportunity.
5. End with a concrete action item based on the Recommended Investment.
"""
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to rule-based engine.")
            return _generate_fallback_narrative(context)
            
    else:
        # Offline or no API key
        return _generate_fallback_narrative(context)
