"""
Gold Layer — LLM Explainable AI (XAI) Module.

Generates human-readable, business-oriented narratives explaining model decisions
for individual retail outlets. Uses a hybrid design:
  1. Real LLM connection using google-generativeai (Gemini API) if GEMINI_API_KEY is configured.
  2. Professional, dynamic rule-based narrative engine as a fallback for offline local running.

Usage:
    from src.gold.llm_xai import explain_outlet
    explanation = explain_outlet(outlet_data)
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


def _generate_fallback_narrative(data: dict) -> str:
    """
    Generates a highly-contextual, professional business narrative based on outlet metrics.
    Ensures the Streamlit app works out-of-the-box without API keys.
    """
    outlet_id = data.get("Outlet_ID", "Unknown Outlet")
    potential = data.get("Maximum_Monthly_Liters", 0.0)
    avg_volume = data.get("Avg_Monthly_Volume", 0.0)
    growth_gap = potential - avg_volume
    growth_pct = (growth_gap / avg_volume * 100.0) if avg_volume > 0 else 0.0
    
    coolers = int(data.get("Cooler_Count", 0))
    tier = data.get("Dynamic_Tier", "Unknown")
    is_goldmine = int(data.get("is_isolated_goldmine", 0))
    ratio = data.get("latent_opportunity_ratio", 0.0)
    comp = data.get("competitive_saturation_index", 0.0)
    
    # Extract spatio-temporal gravity scores
    youth_grav = data.get("gravity_group_youth", 0.0)
    leisure_grav = data.get("gravity_group_leisure", 0.0)
    health_grav = data.get("gravity_group_health", 0.0)
    
    # ── Narrative Assembly ────────────────────────────────────────────────
    # Paragraph 1: Potential and Tier
    p1 = (
        f"Outlet {outlet_id} is classified as a {tier}-Tier retailer with a predicted maximum purchase potential of "
        f"{potential:,.1f} Liters/month, representing a {growth_pct:+.1f}% expansion over its historical average of {avg_volume:,.1f} Liters. "
    )
    
    # Paragraph 2: Drivers (Spatial and Competitive)
    drivers = []
    if is_goldmine:
        drivers.append("operates in a highly lucrative, isolated market with zero direct competitors within 1km")
    elif ratio > 10.0:
        drivers.append("benefits from a high latent opportunity ratio with strong traffic drivers and minimal competitive pressure")
    elif comp > 0.05:
        drivers.append("operates in a highly saturated competitive zone, suggesting that promotions must target market share capture")
    else:
        drivers.append("has a balanced spatial footprint with moderate competitive exposure")
        
    if youth_grav > 1.0:
        drivers.append("shows high sensitivity to student and youth traffic due to nearby schools and educational hubs")
    if leisure_grav > 0.5:
        drivers.append("enjoys tourist and weekend surges driven by proximity to local parks or beaches")
    if health_grav > 0.5:
        drivers.append("has consistent baseline footfall linked to nearby healthcare facilities")
        
    if drivers:
        p2 = f"This potential is unlocked because the outlet " + ", and ".join(drivers) + ". "
    else:
        p2 = ""
        
    # Paragraph 3: Assets & Recommendations
    if coolers == 0:
        p3 = f"Recommendation: Deploy at least 1 cooler unit immediately to capture the LKR {potential * 150:,.0f} potential value gap."
    elif coolers == 1 and potential > 300:
        p3 = "Recommendation: Upgrade to a double-door cooler to prevent out-of-stock events during peak weekends."
    else:
        p3 = "Recommendation: Allocate trade spend budget for weekend volume discounts to maximize velocity."
        
    return p1 + p2 + p3


def explain_outlet(data: dict) -> str:
    """
    Generates a natural language explanation of the model's prediction for an outlet.
    
    Parameters
    ----------
    data : dict containing all necessary model inputs and outputs for the outlet.
    
    Returns
    -------
    str : Narrative explanation
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if HAS_GEMINI and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Construct a clear, structured prompt
            prompt = f"""
You are the InsightAI Executive Assistant. Translate the following outlet metrics into a simple business narrative explaining why the outlet was assigned its purchase potential score.

Metrics for Outlet {data.get('Outlet_ID', 'Unknown')}:
- Predicted Potential Volume: {data.get('Maximum_Monthly_Liters', 0.0):.1f} Liters/month
- Historical Monthly Average: {data.get('Avg_Monthly_Volume', 0.0):.1f} Liters/month
- Market Position: {data.get('Dynamic_Tier', 'Unknown')} Tier retailer
- Coolers Deployed: {int(data.get('Cooler_Count', 0))} units
- Spatial Opportunity (Latent Opportunity Ratio): {data.get('latent_opportunity_ratio', 0.0):.2f}
- Competitive Saturation Index: {data.get('competitive_saturation_index', 0.0):.4f} (Isolated Goldmine: {int(data.get('is_isolated_goldmine', 0))})
- Gravity Signals: Youth/Education={data.get('gravity_group_youth', 0.0):.2f}, Leisure/Park={data.get('gravity_group_leisure', 0.0):.2f}, Health/Hospital={data.get('gravity_group_health', 0.0):.2f}

Write a professional explanation in exactly 3 business sentences:
1. First sentence: Describe the potential score and how it compares to history.
2. Second sentence: Explain the primary spatial/competitive reasons for this prediction.
3. Third sentence: Provide a concrete operational recommendation (e.g., cooler placement, trade promotions).
"""
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to rule-based engine.")
            return _generate_fallback_narrative(data)
            
    else:
        # Offline or no API key
        return _generate_fallback_narrative(data)
