"""
Reusable Data Quality Checks — Parameterizable functions for the Silver layer.

Every check returns a DataFrame of FAILED records with a standardized schema:
    dataset | check_type | column | key_value | row_index | reason | original_value | timestamp

These are collected and sent to the quarantine system (see quarantine.py).

Usage:
    from src.silver.dq_checks import check_duplicates, check_nulls
    failures = check_duplicates(df, key_columns=["Outlet_ID", "Year", "Month"], dataset_name="transactions")
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("silver.dq_checks")


# ── Standardized Rejection Record Schema ─────────────────────


def _rejection_record(
    dataset: str,
    check_type: str,
    column: str,
    key_value: str,
    row_index: int,
    reason: str,
    original_value: str,
) -> dict:
    """Create a single standardized rejection record."""
    return {
        "dataset": dataset,
        "check_type": check_type,
        "column": column,
        "key_value": str(key_value),
        "row_index": int(row_index),
        "reason": reason,
        "original_value": str(original_value),
        "timestamp": datetime.now().isoformat(),
    }


def _build_rejection_df(records: list[dict]) -> pd.DataFrame:
    """Convert a list of rejection dicts to a standardized DataFrame."""
    if not records:
        return pd.DataFrame(
            columns=[
                "dataset", "check_type", "column", "key_value",
                "row_index", "reason", "original_value", "timestamp"
            ]
        )
    return pd.DataFrame(records)


# ── DQ Check Functions ───────────────────────────────────────


def check_duplicates(
    df: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
    id_column: str = "Outlet_ID",
) -> pd.DataFrame:
    """
    Detect duplicate records based on a configurable composite key.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    key_columns : list[str]
        Columns forming the composite key.
    dataset_name : str
        Name of the dataset (for rejection records).
    id_column : str
        Column to use as the key_value identifier.

    Returns
    -------
    pd.DataFrame
        Rejection records for duplicates found.
    """
    dupes = df[df.duplicated(subset=key_columns, keep="first")]
    logger.info(
        f"[{dataset_name}] Duplicate check on {key_columns}: "
        f"{len(dupes):,} duplicates found out of {len(df):,} rows"
    )

    records = []
    for idx, row in dupes.iterrows():
        records.append(_rejection_record(
            dataset=dataset_name,
            check_type="DUPLICATE",
            column="|".join(key_columns),
            key_value=row.get(id_column, ""),
            row_index=idx,
            reason=f"Duplicate on composite key: {key_columns}",
            original_value=str({col: row[col] for col in key_columns}),
        ))

    return _build_rejection_df(records)


def check_nulls(
    df: pd.DataFrame,
    mandatory_columns: list[str],
    dataset_name: str,
    id_column: str = "Outlet_ID",
) -> pd.DataFrame:
    """
    Flag records where mandatory fields contain null or empty values.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    mandatory_columns : list[str]
        Columns that must not be null.
    dataset_name : str
        Name of the dataset.
    id_column : str
        Column to use as the key_value identifier.

    Returns
    -------
    pd.DataFrame
        Rejection records for null violations.
    """
    records = []
    for col in mandatory_columns:
        null_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        null_rows = df[null_mask]

        logger.info(
            f"[{dataset_name}] Null check on '{col}': "
            f"{len(null_rows):,} nulls found"
        )

        for idx, row in null_rows.iterrows():
            records.append(_rejection_record(
                dataset=dataset_name,
                check_type="NULL_VALUE",
                column=col,
                key_value=row.get(id_column, ""),
                row_index=idx,
                reason=f"Mandatory column '{col}' is null or empty",
                original_value=str(row[col]),
            ))

    return _build_rejection_df(records)


def check_referential_integrity(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    fk_column: str,
    pk_column: str,
    dataset_name: str,
    ref_dataset_name: str = "reference",
    id_column: str = "Outlet_ID",
) -> pd.DataFrame:
    """
    Validate that foreign key values exist in a reference dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the foreign key.
    ref_df : pd.DataFrame
        Reference dataset containing the primary key.
    fk_column : str
        Foreign key column in df.
    pk_column : str
        Primary key column in ref_df.
    dataset_name : str
        Name of the source dataset.
    ref_dataset_name : str
        Name of the reference dataset.
    id_column : str
        Column to use as the key_value identifier.

    Returns
    -------
    pd.DataFrame
        Rejection records for referential integrity violations.
    """
    valid_keys = set(ref_df[pk_column].dropna().unique())
    orphan_mask = ~df[fk_column].isin(valid_keys)
    orphans = df[orphan_mask]

    logger.info(
        f"[{dataset_name}] Referential integrity {fk_column} → "
        f"{ref_dataset_name}.{pk_column}: {len(orphans):,} orphans found"
    )

    records = []
    for idx, row in orphans.iterrows():
        records.append(_rejection_record(
            dataset=dataset_name,
            check_type="REFERENTIAL_INTEGRITY",
            column=fk_column,
            key_value=row.get(id_column, ""),
            row_index=idx,
            reason=(
                f"Value '{row[fk_column]}' in '{fk_column}' not found in "
                f"{ref_dataset_name}.{pk_column}"
            ),
            original_value=str(row[fk_column]),
        ))

    return _build_rejection_df(records)


def check_value_range(
    df: pd.DataFrame,
    column: str,
    min_val: float,
    max_val: float,
    dataset_name: str,
    id_column: str = "Outlet_ID",
) -> pd.DataFrame:
    """
    Assert numeric fields fall within expected min/max boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    column : str
        Numeric column to check.
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.
    dataset_name : str
        Name of the dataset.
    id_column : str
        Column to use as the key_value identifier.

    Returns
    -------
    pd.DataFrame
        Rejection records for out-of-range values.
    """
    out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]

    logger.info(
        f"[{dataset_name}] Range check on '{column}' "
        f"[{min_val}, {max_val}]: {len(out_of_range):,} violations"
    )

    records = []
    for idx, row in out_of_range.iterrows():
        records.append(_rejection_record(
            dataset=dataset_name,
            check_type="VALUE_RANGE",
            column=column,
            key_value=row.get(id_column, ""),
            row_index=idx,
            reason=(
                f"Value {row[column]} outside allowed range "
                f"[{min_val}, {max_val}]"
            ),
            original_value=str(row[column]),
        ))

    return _build_rejection_df(records)


def check_format(
    df: pd.DataFrame,
    column: str,
    pattern: str,
    dataset_name: str,
    id_column: str = "Outlet_ID",
) -> pd.DataFrame:
    """
    Validate fields conform to an expected regex format.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    column : str
        Column to validate.
    pattern : str
        Regex pattern the values should match (e.g., r"^OUT_\\d{5}$").
    dataset_name : str
        Name of the dataset.
    id_column : str
        Column to use as the key_value identifier.

    Returns
    -------
    pd.DataFrame
        Rejection records for format violations.
    """
    non_null = df[df[column].notna()].copy()
    match_mask = non_null[column].astype(str).str.match(pattern)
    violations = non_null[~match_mask]

    logger.info(
        f"[{dataset_name}] Format check on '{column}' "
        f"(pattern: {pattern}): {len(violations):,} violations"
    )

    records = []
    for idx, row in violations.iterrows():
        records.append(_rejection_record(
            dataset=dataset_name,
            check_type="FORMAT_VIOLATION",
            column=column,
            key_value=row.get(id_column, ""),
            row_index=idx,
            reason=f"Value '{row[column]}' does not match pattern '{pattern}'",
            original_value=str(row[column]),
        ))

    return _build_rejection_df(records)


# ── Transaction-Specific Forensic Checks ─────────────────────


def check_negative_volumes(
    df: pd.DataFrame,
    volume_column: str = "Volume_Liters",
    dataset_name: str = "transactions",
    id_column: str = "Outlet_ID",
) -> pd.DataFrame:
    """
    Detect negative volume rows — these are returns/reversals, NOT errors.
    Tag them for forensic analysis rather than dropping.

    Returns
    -------
    pd.DataFrame
        Rejection records tagged as NEGATIVE_VOLUME (quarantined for analysis).
    """
    negatives = df[df[volume_column] < 0]

    logger.info(
        f"[{dataset_name}] Negative volume check: "
        f"{len(negatives):,} rows with negative {volume_column}"
    )

    records = []
    for idx, row in negatives.iterrows():
        records.append(_rejection_record(
            dataset=dataset_name,
            check_type="NEGATIVE_VOLUME",
            column=volume_column,
            key_value=row.get(id_column, ""),
            row_index=idx,
            reason=(
                f"Negative volume ({row[volume_column]:.2f} L) — "
                f"likely return/reversal. Tagged for aggregation, not dropped."
            ),
            original_value=str(row[volume_column]),
        ))

    return _build_rejection_df(records)


def check_zero_volumes(
    df: pd.DataFrame,
    volume_column: str = "Volume_Liters",
    dataset_name: str = "transactions",
    id_column: str = "Outlet_ID",
) -> pd.DataFrame:
    """
    Detect zero-volume rows — system adjustments or fee entries.

    Returns
    -------
    pd.DataFrame
        Rejection records tagged as ZERO_VOLUME.
    """
    zeros = df[df[volume_column] == 0.0]

    logger.info(
        f"[{dataset_name}] Zero volume check: "
        f"{len(zeros):,} rows with exactly 0.0 {volume_column}"
    )

    records = []
    for idx, row in zeros.iterrows():
        records.append(_rejection_record(
            dataset=dataset_name,
            check_type="ZERO_VOLUME",
            column=volume_column,
            key_value=row.get(id_column, ""),
            row_index=idx,
            reason="Zero volume — likely system adjustment or fee entry",
            original_value=str(row[volume_column]),
        ))

    return _build_rejection_df(records)
