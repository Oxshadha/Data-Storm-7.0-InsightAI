"""
Quarantine System — Rejected records handler.

Records that fail DQ checks are NEVER silently dropped.
They are stored in the rejected records store with documented failure reasons.
"""

from pathlib import Path
import pandas as pd
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("silver.quarantine")


class QuarantineManager:
    """Manages the rejected records store."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self._rejected_dir = resolve_path(config["paths"]["silver"]["rejected_dir"])
        self._manifest_path = resolve_path(config["paths"]["silver"]["rejection_manifest"])
        self._rejections: list[pd.DataFrame] = []
        self._rejected_dir.mkdir(parents=True, exist_ok=True)

    def add_rejections(self, rejection_df: pd.DataFrame) -> None:
        """Add a batch of rejection records from a DQ check."""
        if rejection_df.empty:
            return
        self._rejections.append(rejection_df)
        logger.info(f"Quarantined {len(rejection_df):,} records")

    def get_all_rejections(self) -> pd.DataFrame:
        """Get all collected rejection records as a single DataFrame."""
        if not self._rejections:
            return pd.DataFrame()
        return pd.concat(self._rejections, ignore_index=True)

    def flush(self) -> None:
        """Write all rejections to disk (per-dataset parquets + manifest CSV)."""
        all_rejections = self.get_all_rejections()
        if all_rejections.empty:
            logger.info("No rejections to flush.")
            return

        for dataset_name, group in all_rejections.groupby("dataset"):
            path = self._rejected_dir / f"{dataset_name}_rejected.parquet"
            group.to_parquet(path, index=False)
            logger.info(f"Wrote {len(group):,} rejected → {path.name}")

        all_rejections.to_csv(self._manifest_path, index=False)
        logger.info(f"Manifest: {len(all_rejections):,} total → {self._manifest_path.name}")

    @property
    def total_rejections(self) -> int:
        return sum(len(df) for df in self._rejections)
