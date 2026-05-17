"""
Silver Layer — Quarantine System (Rewritten).

Receives a tagged DataFrame (with Rejection_Reason column filled).
Splits into clean vs quarantined, writes both to disk.

Usage:
    from src.silver.quarantine import write_quarantine_outputs
    write_quarantine_outputs(df, dataset_name="transactions", config=cfg)
"""

import pandas as pd
from pathlib import Path
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("silver.quarantine")

REASON_COL = "Rejection_Reason"


def write_quarantine_outputs(
    df: pd.DataFrame,
    dataset_name: str,
    clean_path: str,
    config: dict | None = None,
) -> tuple[int, int]:
    """
    Split a tagged DataFrame and write both clean and quarantined outputs.

    Parameters
    ----------
    df : pd.DataFrame
        Tagged DataFrame with Rejection_Reason column.
    dataset_name : str
        Name used for quarantine filenames (e.g., "transactions").
    clean_path : str
        Relative path for the clean Silver Parquet output.
    config : dict, optional
        Pipeline config.

    Returns
    -------
    (n_clean, n_quarantined) row counts
    """
    if config is None:
        config = load_config()

    rejected_dir = resolve_path(config["paths"]["silver"]["rejected_dir"])
    rejected_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_path(config["paths"]["silver"]["rejection_manifest"])

    bronze_count = len(df)

    # ── Split ────────────────────────────────────────────────
    clean_mask = df[REASON_COL].str.strip() == ""
    clean_df = df[clean_mask].drop(columns=[REASON_COL]).reset_index(drop=True)
    quarantined_df = df[~clean_mask].reset_index(drop=True)

    n_clean = len(clean_df)
    n_quarantined = len(quarantined_df)

    logger.info(
        f"[{dataset_name}] Split: "
        f"{n_clean:,} clean | {n_quarantined:,} quarantined "
        f"(Bronze total: {bronze_count:,})"
    )

    # Verify no silent data loss
    assert n_clean + n_quarantined == bronze_count, (
        f"DATA LOSS: {n_clean} + {n_quarantined} ≠ {bronze_count}"
    )

    # ── Write clean Silver output ────────────────────────────
    clean_out = resolve_path(clean_path)
    clean_out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(clean_out, index=False)
    logger.info(f"  Clean → {clean_out.name} ({clean_out.stat().st_size / 1e6:.2f} MB)")

    # ── Write quarantined rows (full rows + reason) ──────────
    if not quarantined_df.empty:
        quarantine_parquet = rejected_dir / f"{dataset_name}_rejected.parquet"
        quarantine_csv = rejected_dir / f"{dataset_name}_rejected.csv"

        quarantined_df.to_parquet(quarantine_parquet, index=False)
        quarantined_df.to_csv(quarantine_csv, index=False)

        logger.info(
            f"  Quarantined → {quarantine_csv.name} "
            f"({len(quarantined_df):,} rows)"
        )

        # Log rejection reason breakdown
        reasons = _parse_reasons(quarantined_df[REASON_COL])
        for reason, count in reasons.items():
            logger.info(f"    [{reason}]: {count:,} rows")

        # Update global manifest
        _append_to_manifest(quarantined_df, dataset_name, manifest_path)

    return n_clean, n_quarantined


def _parse_reasons(reason_series: pd.Series) -> dict:
    """Parse stacked rejection reasons into a summary count dict."""
    counts = {}
    for cell in reason_series:
        for part in str(cell).split(";"):
            tag = part.strip().split(":")[0].strip()
            if tag:
                counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _append_to_manifest(
    quarantined_df: pd.DataFrame,
    dataset_name: str,
    manifest_path: Path,
) -> None:
    """Append summary stats to the global rejection manifest CSV."""
    from datetime import datetime

    reasons = _parse_reasons(quarantined_df[REASON_COL])
    rows = [
        {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "check_type": reason,
            "row_count": count,
        }
        for reason, count in reasons.items()
    ]
    summary_df = pd.DataFrame(rows)

    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

    summary_df.to_csv(manifest_path, index=False)
