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
    # Master (left join keeps all transaction outlets, even if missing from cleaned master)
    abt = base_grid.merge(master_df, on="Outlet_ID", how="left")
    
    # ── 3a. Impute orphaned outlets (quarantined from master due to null Outlet_Size) ──
    orphan_mask = abt["Avg_Monthly_Volume"].isna()
    n_orphan_outlets = abt.loc[orphan_mask, "Outlet_ID"].nunique()
    if n_orphan_outlets > 0:
        logger.info(f"Detected {n_orphan_outlets} outlets missing from cleaned master — imputing from transaction behavior...")
        
        # Compute Avg_Monthly_Volume from actual transaction records for orphaned outlets
        orphan_ids = abt.loc[orphan_mask, "Outlet_ID"].unique()
        orphan_avg = (
            abt.loc[abt["Outlet_ID"].isin(orphan_ids)]
            .groupby("Outlet_ID")["Total_Volume"]
            .mean()
            .rename("_imputed_avg")
        )
        abt = abt.merge(orphan_avg, on="Outlet_ID", how="left")
        abt.loc[orphan_mask, "Avg_Monthly_Volume"] = abt.loc[orphan_mask, "_imputed_avg"]
        abt.drop(columns=["_imputed_avg"], inplace=True)
        
        # Fill remaining master fields with safe defaults
        abt["Outlet_Size"] = abt["Outlet_Size"].fillna("Unknown")
        abt["Cooler_Count"] = abt["Cooler_Count"].fillna(0)
        abt["Outlet_Type"] = abt["Outlet_Type"].fillna("Unknown")
        abt["Dynamic_Tier"] = abt["Dynamic_Tier"].fillna("Tier-4")
        
        # Fallback: if Avg_Monthly_Volume is still NaN (brand new outlet with no history), set to 0
        abt["Avg_Monthly_Volume"] = abt["Avg_Monthly_Volume"].fillna(0.0)
        
        logger.info(f"  → Recovered {n_orphan_outlets} outlets with imputed profiles (Avg_Monthly_Volume from transactions)")
    
    # POIs (V2: now includes both flat counts AND gravity scores)
    abt = abt.merge(poi_df, on="Outlet_ID", how="left")
    
    # Define Catchment Boolean Flags based on POI categories
    # The columns in poi_df are prefixed with "poi_count_"
    
    def sum_poi(df, categories):
        cols = [f"poi_count_{c}" for c in categories if f"poi_count_{c}" in df.columns]
        if not cols: return pd.Series(0, index=df.index)
        return df[cols].sum(axis=1)

    # High Footfall Drivers (Catchments) - Numerical
    if "poi_driver_catchment" not in abt.columns:
        abt["poi_driver_catchment"] = sum_poi(abt, [
            "school", "education", "preschool", "college_university", "educational_services",
            "park", "beach", "playground", "national_park",
            "hospital", "bus_station", "train_station", "gym",
            "stadium_arena", "cricket_ground", "sports_and_recreation_venue",
            "buddhist_temple", "hindu_temple", "mosque", "church_cathedral", "catholic_church",
            "landmark_and_historical_building"
        ])
    abt["Has_High_Footfall_Catchment"] = (abt["poi_driver_catchment"] > 0).astype(int)
    
    # Competitive Cannibalization Risks - Numerical
    if "poi_cannibal_risk" not in abt.columns:
        abt["poi_cannibal_risk"] = sum_poi(abt, [
            "supermarket", "restaurant", "cafe", "convenience_store", "grocery_store",
            "hotel", "bakery", "accommodation", "resort", "bar", "liquor_store"
        ])
    abt["Has_Cannibalization_Risk"] = (abt["poi_cannibal_risk"] > 0).astype(int)
    
    # Keep specific ones for interactions (binary flags from flat counts)
    abt["Has_Youth_Catchment"] = (sum_poi(abt, ["school", "education", "preschool", "college_university", "educational_services"]) > 0).astype(int)
    abt["Has_Leisure_Catchment"] = (sum_poi(abt, ["park", "beach", "playground", "national_park"]) > 0).astype(int)
    abt["Has_Health_Catchment"] = (sum_poi(abt, ["hospital"]) > 0).astype(int)
    abt["Has_Athletic_Catchment"] = (sum_poi(abt, ["gym", "stadium_arena", "cricket_ground", "sports_club_and_league", "sports_and_recreation_venue"]) > 0).astype(int)
    abt["Has_Religious_Catchment"] = (sum_poi(abt, ["buddhist_temple", "hindu_temple", "mosque", "church_cathedral", "catholic_church"]) > 0).astype(int)
    abt["Has_Tourist_Catchment"] = (sum_poi(abt, ["landmark_and_historical_building"]) > 0).astype(int)

    # Fill NaN gravity scores for outlets with no nearby POIs
    gravity_cols = [c for c in abt.columns if c.startswith("gravity_")]
    for c in gravity_cols:
        abt[c] = abt[c].fillna(0.0)
    for c in ["competitive_saturation_index", "latent_opportunity_ratio", 
              "total_driver_gravity", "is_isolated_goldmine",
              "comp_saturation_retail", "comp_saturation_food",
              "competitor_count_flat"]:
        if c in abt.columns:
            abt[c] = abt[c].fillna(0)

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
    abt["Is_High_Season"] = (abt["Seasonality_Index"] == "Favorable").astype(int)
    
    # ── 5. Spatio-Temporal Interactions (V2: Binary + Continuous Gravity) ─────
    logger.info("Constructing Spatio-Temporal Interactions (V2: gravity-enhanced)...")
    
    # --- R1 Binary Interactions (backward-compatible) ---
    # 1. The Tuition/Weekend Surge (binary × int)
    abt["Tuition_Weekend_Surge"] = abt["Has_Youth_Catchment"] * abt["Number_of_Weekends"]
    # 2. The Tourist Peak Multiplier (binary × binary)
    abt["Tourist_Peak_Multiplier"] = abt["Has_Leisure_Catchment"] * abt["Is_High_Season"]
    # 3. The Sports & "Big Match" Spike
    abt["Sports_Big_Match_Spike"] = abt["Has_Athletic_Catchment"] * (abt["Is_Cultural_Month"] * 1.5 + abt["Number_of_Weekends"])
    # 4. The Health/Hospital Pulse
    abt["Health_Catchment_Spike"] = abt["Has_Health_Catchment"] * (abt["Number_of_Weekends"] + abt["Holiday_Count"])

    # --- V2 Continuous Gravity Interactions (NEW for Round 2) ---
    # These use distance-decay gravity scores instead of binary flags,
    # giving the model continuous signal strength rather than on/off switches.
    
    if "gravity_group_youth" in abt.columns:
        abt["Tuition_Weekend_Gravity"] = abt["gravity_group_youth"] * abt["Number_of_Weekends"]
        logger.info("  ✓ Tuition_Weekend_Gravity (continuous)")
    
    if "gravity_group_leisure" in abt.columns:
        abt["Tourist_Peak_Gravity"] = abt["gravity_group_leisure"] * abt["Is_High_Season"]
        logger.info("  ✓ Tourist_Peak_Gravity (continuous)")
    
    if "gravity_group_athletic" in abt.columns:
        abt["Sports_Match_Gravity"] = abt["gravity_group_athletic"] * (abt["Is_Cultural_Month"] * 1.5 + abt["Number_of_Weekends"])
        logger.info("  ✓ Sports_Match_Gravity (continuous)")
    
    if "gravity_group_health" in abt.columns:
        abt["Health_Pulse_Gravity"] = abt["gravity_group_health"] * (abt["Number_of_Weekends"] + abt["Holiday_Count"])
        logger.info("  ✓ Health_Pulse_Gravity (continuous)")
    
    # Competitive pressure interaction: high competition × low season = price war signal
    if "competitive_saturation_index" in abt.columns:
        abt["Competition_Season_Pressure"] = abt["competitive_saturation_index"] * (1 - abt["Is_High_Season"])
        logger.info("  ✓ Competition_Season_Pressure (competitive × off-season)")

    # ── 6. Censoring Signal Detection (Is_Censored) ───────────────────────────
    logger.info("Computing variance and flagging Censored (flatlining) outlets...")
    
    # Merge the raw CV computed before smoothing
    abt = abt.merge(raw_outlet_stats[["Outlet_ID", "Volume_CV"]], on="Outlet_ID", how="left")
    
    # A shop is censored if:
    # It has a strong Spatio-Temporal demand signal (e.g., Tuition Surge > 0)
    # AND its volume CV is very low (meaning it flatlined, hitting a system constraint limit)
    # AND it has non-zero volume
    
    # V2: Use total_driver_gravity (continuous) for more precise censoring detection
    abt["Total_Interaction_Score"] = (
        abt["Tuition_Weekend_Surge"] + 
        abt["Tourist_Peak_Multiplier"] + 
        abt["Sports_Big_Match_Spike"] + 
        abt["Health_Catchment_Spike"]
    )
    
    cv_threshold = config.get("modeling", {}).get("censoring", {}).get("cv_threshold", 0.15)
    is_flatlined = (abt["Volume_CV"] < cv_threshold) & (abt["Total_Volume"] > 0)
    
    # V2: Enhanced censoring — use gravity score OR binary interaction score
    if "total_driver_gravity" in abt.columns:
        has_high_demand_signal = (abt["Total_Interaction_Score"] > 0) | (abt["total_driver_gravity"] > 0.5)
    else:
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
