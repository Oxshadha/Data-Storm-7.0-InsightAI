"""
Silver Layer — Vectorized DQ Check Functions.

All checks operate on a DataFrame with a 'Rejection_Reason' column.
They TAG rows in-place by appending to Rejection_Reason (never deleting rows).
The quarantine split happens at the end in each clean_*.py script.

Key design rule:
    df.loc[mask, reason_col] += "Reason text; "
    → Multiple failures on one row stack up cleanly.

Usage:
    from src.silver.dq_checks import (
        add_rejection_column, check_nulls, check_duplicates,
        check_referential_integrity, check_value_range, check_format,
        check_zero_volumes, check_duplicate_retries, check_lazy_rep
    )
    df = add_rejection_column(df)
    df = check_nulls(df, ["Outlet_ID", "Volume_Liters"])
    df = check_zero_volumes(df)
    clean, quarantined = split_by_rejection(df)
"""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("silver.dq_checks")

REASON_COL = "Rejection_Reason"


# ── Setup ─────────────────────────────────────────────────────


def add_rejection_column(df: pd.DataFrame, reason_col: str = REASON_COL) -> pd.DataFrame:
    """Add an empty Rejection_Reason column to the DataFrame if not present."""
    if reason_col not in df.columns:
        df[reason_col] = ""
    return df


def split_by_rejection(
    df: pd.DataFrame, reason_col: str = REASON_COL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into clean and quarantined based on Rejection_Reason.

    Returns
    -------
    (clean_df, quarantined_df)
        clean_df: rows where Rejection_Reason is empty
        quarantined_df: rows where Rejection_Reason is filled
    """
    clean_mask = df[reason_col].str.strip() == ""
    clean_df = df[clean_mask].drop(columns=[reason_col]).reset_index(drop=True)
    quarantined_df = df[~clean_mask].reset_index(drop=True)
    return clean_df, quarantined_df


# ── Generic DQ Checks (Vectorized) ───────────────────────────


def check_nulls(
    df: pd.DataFrame,
    mandatory_columns: list[str],
    reason_col: str = REASON_COL,
) -> pd.DataFrame:
    """
    Flag rows where any mandatory field is null or empty string.
    Appends to Rejection_Reason without overwriting prior failures.
    """
    for col in mandatory_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame — skipping null check.")
            continue
        mask = df[col].isnull() | (df[col].astype(str).str.strip() == "")
        count = mask.sum()
        if count:
            logger.info(f"  NULL_CHECK '{col}': {count:,} violations")
            df.loc[mask, reason_col] += f"NULL:{col}; "

    return df


def check_duplicates(
    df: pd.DataFrame,
    key_columns: list[str],
    reason_col: str = REASON_COL,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Flag duplicate rows based on composite key. Keeps 'first' occurrence clean.
    Duplicate retries (keep=False on all) use check_duplicate_retries instead.
    """
    dupe_mask = df.duplicated(subset=key_columns, keep=keep)
    count = dupe_mask.sum()
    if count:
        logger.info(f"  DUPLICATE_CHECK {key_columns}: {count:,} duplicates flagged")
        df.loc[dupe_mask, reason_col] += f"DUPLICATE:{'+'.join(key_columns)}; "

    return df


def check_referential_integrity(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    fk_column: str,
    pk_column: str,
    ref_name: str = "reference",
    reason_col: str = REASON_COL,
) -> pd.DataFrame:
    """
    Flag rows where fk_column value does not exist in ref_df[pk_column].
    """
    valid_keys = set(ref_df[pk_column].dropna().unique())
    orphan_mask = ~df[fk_column].isin(valid_keys)
    count = orphan_mask.sum()
    if count:
        logger.info(
            f"  REF_INTEGRITY '{fk_column}' → {ref_name}.{pk_column}: "
            f"{count:,} orphans"
        )
        df.loc[orphan_mask, reason_col] += (
            f"REF_INTEGRITY:{fk_column}→{ref_name}.{pk_column}; "
        )

    return df


def check_value_range(
    df: pd.DataFrame,
    column: str,
    min_val: float,
    max_val: float,
    reason_col: str = REASON_COL,
) -> pd.DataFrame:
    """
    Flag rows where numeric column falls outside [min_val, max_val].
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found — skipping range check.")
        return df

    mask = (df[column] < min_val) | (df[column] > max_val)
    count = mask.sum()
    if count:
        logger.info(f"  RANGE_CHECK '{column}' [{min_val},{max_val}]: {count:,} violations")
        df.loc[mask, reason_col] += f"RANGE:{column}<{min_val}or>{max_val}; "

    return df


def check_format(
    df: pd.DataFrame,
    column: str,
    pattern: str,
    reason_col: str = REASON_COL,
) -> pd.DataFrame:
    """
    Flag rows where string column does not match regex pattern.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found — skipping format check.")
        return df

    non_null = df[column].notna()
    mismatch = non_null & ~df[column].astype(str).str.match(pattern, na=False)
    count = mismatch.sum()
    if count:
        logger.info(f"  FORMAT_CHECK '{column}' (pattern={pattern}): {count:,} violations")
        df.loc[mismatch, reason_col] += f"FORMAT:{column}!~{pattern}; "

    return df


def check_valid_values(
    df: pd.DataFrame,
    column: str,
    valid_values: list,
    reason_col: str = REASON_COL,
) -> pd.DataFrame:
    """
    Flag rows where a column contains values not in the allowed set.
    Used after corrections — anything remaining that's invalid gets quarantined.
    """
    if column not in df.columns:
        return df

    invalid_mask = df[column].notna() & ~df[column].isin(valid_values)
    count = invalid_mask.sum()
    if count:
        logger.info(f"  VALID_VALUES '{column}': {count:,} invalid values found")
        df.loc[invalid_mask, reason_col] += f"INVALID_VALUE:{column}; "

    return df


# ── Transaction-Specific Forensic Checks ─────────────────────


def check_zero_volumes(
    df: pd.DataFrame,
    volume_col: str = "Volume_Liters",
    reason_col: str = REASON_COL,
) -> pd.DataFrame:
    """
    Flag zero-volume rows as system adjustments / fee entries.
    These are quarantined — they corrupt pricing and average volume calculations.
    """
    mask = df[volume_col] == 0.0
    count = mask.sum()
    if count:
        logger.info(f"  ZERO_VOLUME: {count:,} rows with 0.0 {volume_col} (system fees)")
        df.loc[mask, reason_col] += "ZERO_VOLUME:system_adjustment_or_fee; "

    return df


def check_duplicate_retries(
    df: pd.DataFrame,
    key_columns: list[str],
    reason_col: str = REASON_COL,
) -> pd.DataFrame:
    """
    Detect duplicate retry rows caused by distributor system upload failures.
    These create EXACT twin rows for the same purchase event.

    Strategy: composite key on all meaningful transaction fields.
    Keep the FIRST occurrence, quarantine all subsequent duplicates.
    This is different from check_duplicates — we're specifically hunting
    system retries, not just key-level duplication.
    """
    dupe_mask = df.duplicated(subset=key_columns, keep="first")
    count = dupe_mask.sum()
    if count:
        logger.info(
            f"  DUPLICATE_RETRY: {count:,} system retry duplicates detected "
            f"(key: {key_columns})"
        )
        df.loc[dupe_mask, reason_col] += "DUPLICATE_RETRY:system_upload_failure; "

    return df


def tag_negative_volumes(
    df: pd.DataFrame,
    volume_col: str = "Volume_Liters",
    flag_col: str = "Is_Return",
) -> pd.DataFrame:
    """
    TAG negative volume rows as returns/reversals — do NOT quarantine them.
    They remain in the clean dataset and net out during aggregation via sum().

    Adds a boolean flag column 'Is_Return' for transparency.
    """
    df[flag_col] = df[volume_col] < 0
    return_count = df[flag_col].sum()
    if return_count:
        logger.info(
            f"  RETURN_TAG: {return_count:,} negative-volume rows tagged as "
            f"returns (NOT quarantined — will net out in aggregation)"
        )

    return df


def check_lazy_rep(
    df: pd.DataFrame,
    outlet_col: str = "Outlet_ID",
    sku_col: str = "SKU_ID",
    volume_col: str = "Volume_Liters",
    max_sku_count: int = 2,
    flag_col: str = "Lazy_Rep_Flag",
) -> pd.DataFrame:
    """
    Detect "Lazy Rep" outlets: massive volume logged under ≤ max_sku_count SKUs.

    A tiny village shop with 15L under 1 SKU = legitimate.
    A large shop with 1,000L under 1 SKU = Lazy Rep.

    Technique: flag outlets where unique SKU count ≤ max_sku_count
    AND total monthly volume > median volume of their outlet tier.

    The flag is added to the CLEAN data (not quarantined) for Gold layer handling.
    Gold layer will redistribute volume using collaborative filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level data (can have multiple rows per outlet/month).
    max_sku_count : int
        Threshold for "suspiciously low" SKU diversity. Default 2.
    """
    # Compute per-outlet stats
    outlet_stats = (
        df.groupby(outlet_col, observed=True)
        .agg(
            unique_skus=(sku_col, "nunique"),
            total_volume=(volume_col, "sum"),
        )
        .reset_index()
    )

    # Median volume across all outlets
    median_volume = outlet_stats["total_volume"].median()

    # Lazy Rep = low SKU diversity AND above-median volume
    lazy_rep_outlets = set(
        outlet_stats.loc[
            (outlet_stats["unique_skus"] <= max_sku_count)
            & (outlet_stats["total_volume"] > median_volume),
            outlet_col,
        ]
    )

    df[flag_col] = df[outlet_col].isin(lazy_rep_outlets)
    count = df[flag_col].sum()
    unique_outlets = len(lazy_rep_outlets)

    logger.info(
        f"  LAZY_REP: {unique_outlets} outlets flagged "
        f"(≤{max_sku_count} SKUs + above-median volume). "
        f"{count:,} transaction rows tagged with '{flag_col}'"
    )

    return df
