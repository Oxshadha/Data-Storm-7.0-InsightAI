import unittest
import pandas as pd
import numpy as np
import plotly.express as px

def safe_plot_gravity(df):
    """
    Replicates the safe plotting logic used in app.py to render the gravity scatter plot.
    """
    if df.empty or len(df) < 2:
        fig = px.scatter(
            df,
            x="total_driver_gravity", y="Maximum_Monthly_Liters", color="Dynamic_Tier",
            color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"],
            labels={"total_driver_gravity": "Spatial Driver Gravity (Urban Density)", "Maximum_Monthly_Liters": "Predicted Potential (Liters)"},
            opacity=0.8
        )
    else:
        try:
            fig = px.scatter(
                df,
                x="total_driver_gravity", y="Maximum_Monthly_Liters", color="Dynamic_Tier",
                color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"],
                labels={"total_driver_gravity": "Spatial Driver Gravity (Urban Density)", "Maximum_Monthly_Liters": "Predicted Potential (Liters)"},
                opacity=0.8, trendline="ols", trendline_scope="overall", trendline_color_override="#f87171"
            )
        except Exception:
            fig = px.scatter(
                df,
                x="total_driver_gravity", y="Maximum_Monthly_Liters", color="Dynamic_Tier",
                color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"],
                labels={"total_driver_gravity": "Spatial Driver Gravity (Urban Density)", "Maximum_Monthly_Liters": "Predicted Potential (Liters)"},
                opacity=0.8
            )
    return fig

class TestDashboardCharts(unittest.TestCase):
    
    def test_empty_dataframe(self):
        """Test safe_plot_gravity on an empty DataFrame."""
        df = pd.DataFrame(columns=["total_driver_gravity", "Maximum_Monthly_Liters", "Dynamic_Tier"])
        try:
            fig = safe_plot_gravity(df)
            self.assertIsNotNone(fig)
            self.assertEqual(len(fig.data), 0)
        except Exception as e:
            self.fail(f"safe_plot_gravity crashed on empty DataFrame: {e}")

    def test_single_row_dataframe(self):
        """Test safe_plot_gravity on a single-row DataFrame (insufficient data for OLS trendline)."""
        df = pd.DataFrame([{
            "total_driver_gravity": 12.5,
            "Maximum_Monthly_Liters": 450.0,
            "Dynamic_Tier": "Tier 1"
        }])
        try:
            fig = safe_plot_gravity(df)
            self.assertIsNotNone(fig)
            self.assertTrue(len(fig.data) >= 1)
        except Exception as e:
            self.fail(f"safe_plot_gravity crashed on single-row DataFrame: {e}")

    def test_collinear_identical_x_values(self):
        """Test safe_plot_gravity on collinear/identical X values (fails to fit OLS trendline)."""
        df = pd.DataFrame([
            {"total_driver_gravity": 10.0, "Maximum_Monthly_Liters": 100.0, "Dynamic_Tier": "Tier 1"},
            {"total_driver_gravity": 10.0, "Maximum_Monthly_Liters": 200.0, "Dynamic_Tier": "Tier 1"}
        ])
        try:
            fig = safe_plot_gravity(df)
            self.assertIsNotNone(fig)
            self.assertTrue(len(fig.data) >= 1)
        except Exception as e:
            self.fail(f"safe_plot_gravity crashed on collinear X values: {e}")

    def test_all_nan_y_values(self):
        """Test safe_plot_gravity on a DataFrame with NaN Y values."""
        df = pd.DataFrame([
            {"total_driver_gravity": 10.0, "Maximum_Monthly_Liters": np.nan, "Dynamic_Tier": "Tier 1"},
            {"total_driver_gravity": 20.0, "Maximum_Monthly_Liters": np.nan, "Dynamic_Tier": "Tier 1"}
        ])
        try:
            fig = safe_plot_gravity(df)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"safe_plot_gravity crashed on NaN Y values: {e}")

    def test_normal_operation(self):
        """Test safe_plot_gravity on a normal multi-row DataFrame where OLS trendline is expected to succeed."""
        df = pd.DataFrame([
            {"total_driver_gravity": 10.0, "Maximum_Monthly_Liters": 100.0, "Dynamic_Tier": "Tier 1"},
            {"total_driver_gravity": 20.0, "Maximum_Monthly_Liters": 200.0, "Dynamic_Tier": "Tier 1"},
            {"total_driver_gravity": 30.0, "Maximum_Monthly_Liters": 150.0, "Dynamic_Tier": "Tier 2"}
        ])
        try:
            fig = safe_plot_gravity(df)
            self.assertIsNotNone(fig)
            # Normal operation should yield both the scatter points and the trendline trace
            self.assertTrue(len(fig.data) >= 2)
        except Exception as e:
            self.fail(f"safe_plot_gravity crashed on normal operational DataFrame: {e}")

if __name__ == "__main__":
    unittest.main()
