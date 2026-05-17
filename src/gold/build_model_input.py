"""
Gold Layer — Central Nervous System (ABT Generation).

Merges all cleaned datasets and engineered features into a single 
Analytical Base Table (ABT) for the prediction model.

Constructs the Sri Lankan Spatio-Temporal interactions:
  1. Tuition_Weekend_Surge
  2. Tourist_Peak_Multiplier
  3. Sports_Big_Match_Spike
  4. Park_Poya_Outing

Defines the `Is_Censored` target variable to handle "flatlining" demand constraints.

Usage:
    python -m src.gold.build_model_input
"""

import pandas as pd
import numpy as np
import calendar
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.logger import get_logger

logger = get_logger("gold.build_model_input")


def get_weekend_days(year: int, month: int) -> int:
    """Returns the number of weekend days (Saturday + Sunday) in a given month."""
    cal = calendar.monthcalendar(year, month)
    # Weekends are the last two days of the week (indices 5 and 6)
    return sum(1 for week in cal if week[5] != 0) + sum(1 for week in cal if week[6] != 0)


def build_model_input(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("Gold: Building Final Model Input (ABT)")

    paths = config["paths"]

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    logger.info("Loading Silver & CF tables...")
    trans_df = read_parquet(paths["gold"]["root"] + "/transactions_cf_enhanced.parquet")
    master_df = read_parquet(paths["silver"]["outlet_master"])
    season_df = read_parquet(paths["silver"]["distributor_seasonality"])
    holiday_df = read_parquet(paths["silver"]["holiday_list"])
    poi_df = read_parquet(paths["gold"]["root"] + "/outlet_poi_features.parquet")

    # ── 2. Initialize Base Grid (Outlet x Year x Month) ───────────────────────
    # The transactions dataframe already has Outlet, Year, Month, Distributor.
    # We will use this as our base grain.
    # Note: We group by to ensure uniqueness if multiple SKUs exist.
    # We aggregate Volume and Bill to total outlet level for the month.
    
    logger.info("Aggregating transactions to Outlet-Year-Month level...")
    base_grid = (
        trans_df.groupby(["Outlet_ID", "Distributor_ID", "Year", "Month"], observed=True)
        .agg(
            Total_Volume=("Volume_Liters", "sum"),
            Total_Bill=("Total_Bill_Value", "sum"),
            Returns_Count=("Is_Return", "sum"),
            Lazy_Rep_Flag=("Lazy_Rep_Flag", "max")
        )
        .reset_index()
    )

    logger.info("Computing RAW variance for Censored logic (before smoothing)...")
    raw_outlet_stats = (
        base_grid.groupby("Outlet_ID", observed=True)["Total_Volume"]
        .agg(["mean", "std"])
        .reset_index()
    )
    raw_outlet_stats["Volume_CV"] = raw_outlet_stats["std"] / raw_outlet_stats["mean"].replace(0, np.nan)
    raw_outlet_stats["Volume_CV"] = raw_outlet_stats["Volume_CV"].fillna(0)

    logger.info("Applying 3-month quarterly smoothing to correct Wholesale Sell-In spikes...")
    # Sort chronologically to ensure rolling works properly
    base_grid = base_grid.sort_values(["Outlet_ID", "Year", "Month"])
    
    # Apply rolling 3-month average to spread demand
    base_grid["Total_Volume"] = (
        base_grid.groupby("Outlet_ID")["Total_Volume"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    # ── 3. Merge Master Data & POIs ───────────────────────────────────────────
    logger.info("Merging Master and POI features...")
    # Master
    abt = base_grid.merge(master_df, on="Outlet_ID", how="left")
    
    # POIs
    abt = abt.merge(poi_df, on="Outlet_ID", how="left")
    
    # Define Catchment Boolean Flags based on POI categories
    # The columns in poi_df are prefixed with "poi_count_"
    
    def sum_poi(df, categories):
        cols = [f"poi_count_{c}" for c in categories if f"poi_count_{c}" in df.columns]
        if not cols: return pd.Series(0, index=df.index)
        return df[cols].sum(axis=1)

    # High Footfall Drivers (Catchments) - Numerical
    abt["poi_driver_catchment"] = sum_poi(abt, [
        "school", "education", "park", "beach", "hospital", 
        "transport_hub", "stadium", "gym", "leisure", "sports_center", "railway_station"
    ])
    abt["Has_High_Footfall_Catchment"] = (abt["poi_driver_catchment"] > 0).astype(int)
    
    # Competitive Cannibalization Risks (Supermarkets, Restaurants, Cafes) - Numerical
    abt["poi_cannibal_risk"] = sum_poi(abt, ["supermarket", "restaurant", "cafe", "convenience_store"])
    abt["Has_Cannibalization_Risk"] = (abt["poi_cannibal_risk"] > 0).astype(int)
    
    # Keep specific ones for interactions
    abt["Has_Youth_Catchment"] = (sum_poi(abt, ["school", "education"]) > 0).astype(int)
    abt["Has_Leisure_Catchment"] = (sum_poi(abt, ["park", "beach"]) > 0).astype(int)
    abt["Has_Health_Catchment"] = (sum_poi(abt, ["hospital"]) > 0).astype(int)
    abt["Has_Athletic_Catchment"] = (sum_poi(abt, ["stadium", "sports_centre", "pitch", "recreation_center"]) > 0).astype(int)

    # ── 4. Temporal Features ──────────────────────────────────────────────────
    logger.info("Processing Temporal Features (Holidays, Seasonality)...")
    
    # Seasonality
    abt = abt.merge(season_df, on=["Distributor_ID", "Year", "Month"], how="left")
    # If missing, impute with "Moderate" (neutral)
    abt["Seasonality_Index"] = abt["Seasonality_Index"].fillna("Moderate")
    
    # Holidays
    # Extract Year and Month from holiday Date
    holiday_df["Year"] = pd.to_datetime(holiday_df["Date"]).dt.year
    holiday_df["Month"] = pd.to_datetime(holiday_df["Date"]).dt.month
    monthly_holidays = (
        holiday_df.groupby(["Year", "Month"])
        .size()
        .reset_index(name="Holiday_Count")
    )
    abt = abt.merge(monthly_holidays, on=["Year", "Month"], how="left")
    abt["Holiday_Count"] = abt["Holiday_Count"].fillna(0).astype(int)
    
    # Weekends & Cultural Months
    abt["Number_of_Weekends"] = abt.apply(lambda row: get_weekend_days(row["Year"], row["Month"]), axis=1)
    abt["Is_Cultural_Month"] = abt["Month"].isin([3, 4]).astype(int)  # March & April
    
    # ── 5. The Secret Sauce: Spatio-Temporal Interactions ─────────────────────
    logger.info("Constructing Spatio-Temporal Interactions...")
    
    # 1. The Tuition/Weekend Surge
    # Logic: (Has High Youth/Education POIs) × (Number of Weekends in the Month)
    abt["Tuition_Weekend_Surge"] = abt["Has_Youth_Catchment"] * abt["Number_of_Weekends"]
    
    # 2. The Tourist Peak Multiplier
    # Logic: (Has Leisure/Tourist POIs) × (Is High Seasonality for that Province)
    # We define High Seasonality as Seasonality_Index == 'Favorable'
    abt["Is_High_Season"] = (abt["Seasonality_Index"] == "Favorable").astype(int)
    abt["Tourist_Peak_Multiplier"] = abt["Has_Leisure_Catchment"] * abt["Is_High_Season"]
    
    # 3. The Sports & "Big Match" Spike
    # Logic: (Has Athletic/Grounds POIs) × (Is March or April OR Number of Weekends)
    # We will use (Is_Cultural_Month * 1.5 + Number_of_Weekends) as the temporal multiplier
    abt["Sports_Big_Match_Spike"] = abt["Has_Athletic_Catchment"] * (abt["Is_Cultural_Month"] * 1.5 + abt["Number_of_Weekends"])
    
    # 4. The Health/Hospital Pulse
    # Logic: (Has Health POIs) x (Number of Weekends OR Holidays)
    abt["Health_Catchment_Spike"] = abt["Has_Health_Catchment"] * (abt["Number_of_Weekends"] + abt["Holiday_Count"])

    # ── 6. Censoring Signal Detection (Is_Censored) ───────────────────────────
    logger.info("Computing variance and flagging Censored (flatlining) outlets...")
    
    # Merge the raw CV computed before smoothing
    abt = abt.merge(raw_outlet_stats[["Outlet_ID", "Volume_CV"]], on="Outlet_ID", how="left")
    
    # A shop is censored if:
    # It has a strong Spatio-Temporal demand signal (e.g., Tuition Surge > 0)
    # AND its volume CV is very low (meaning it flatlined, hitting a system constraint limit)
    # AND it has non-zero volume
    
    # We look at the sum of interaction scores
    abt["Total_Interaction_Score"] = (
        abt["Tuition_Weekend_Surge"] + 
        abt["Tourist_Peak_Multiplier"] + 
        abt["Sports_Big_Match_Spike"] + 
        abt["Health_Catchment_Spike"]
    )
    
    cv_threshold = config.get("modeling", {}).get("censoring", {}).get("cv_threshold", 0.15)
    is_flatlined = (abt["Volume_CV"] < cv_threshold) & (abt["Total_Volume"] > 0)
    has_high_demand_signal = abt["Total_Interaction_Score"] > 0
    
    abt["Is_Censored"] = (is_flatlined & has_high_demand_signal).astype(int)
    
    censored_count = abt["Is_Censored"].sum()
    logger.warning(f"Detected {censored_count:,} monthly records as CENSORED (flatlining despite high demand).")

    # ── 7. Save ABT ───────────────────────────────────────────────────────────
    out_path = paths["gold"]["root"] + "/model_input.parquet"
    write_parquet(abt, out_path)
    
    logger.info(f"Final Model Input (ABT) shape: {abt.shape}")
    logger.info(f"Saved ABT to {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_model_input()
