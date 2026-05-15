"""
I/O Helpers — Standardized Parquet and CSV read/write operations.

Usage:
    from src.utils.io import read_parquet, write_parquet, read_csv_raw
    df = read_csv_raw("Refernce Resources/transactions_history_final.csv")
    write_parquet(df, "data/bronze/transactions_history.parquet")
"""

from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("utils.io")


def read_csv_raw(relative_path: str, **kwargs) -> pd.DataFrame:
    """
    Read a raw CSV file from a path relative to project root.
    No transformations — preserves the data exactly as-is.

    Parameters
    ----------
    relative_path : str
        Path relative to project root.
    **kwargs
        Additional kwargs passed to pd.read_csv().

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    path = PROJECT_ROOT / relative_path
    logger.info(f"Reading raw CSV: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_csv(path, **kwargs)
    logger.info(f"  → {len(df):,} rows × {len(df.columns)} columns")
    return df


def read_parquet(relative_path: str, **kwargs) -> pd.DataFrame:
    """
    Read a Parquet file from a path relative to project root.

    Parameters
    ----------
    relative_path : str
        Path relative to project root.
    **kwargs
        Additional kwargs passed to pd.read_parquet().

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    path = PROJECT_ROOT / relative_path
    logger.info(f"Reading Parquet: {path.name}")
    df = pd.read_parquet(path, **kwargs)
    logger.info(f"  → {len(df):,} rows × {len(df.columns)} columns")
    return df


def write_parquet(df: pd.DataFrame, relative_path: str, **kwargs) -> Path:
    """
    Write a DataFrame to Parquet at a path relative to project root.
    Creates parent directories if they don't exist.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    relative_path : str
        Path relative to project root.
    **kwargs
        Additional kwargs passed to df.to_parquet().

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    logger.info(f"Wrote Parquet: {path.name} → {len(df):,} rows ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def write_csv(df: pd.DataFrame, relative_path: str, **kwargs) -> Path:
    """
    Write a DataFrame to CSV at a path relative to project root.
    Creates parent directories if they don't exist.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    relative_path : str
        Path relative to project root.
    **kwargs
        Additional kwargs passed to df.to_csv().

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info(f"Wrote CSV: {path.name} → {len(df):,} rows")
    return path
