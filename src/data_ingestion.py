"""
src/data_ingestion.py
─────────────────────────────────────────────────────────────────────────────
DataIngestionService — manages dynamic data intake for retraining.

Companies have multiple data batches arriving over time.  This module:
  • Validates incoming CSVs against the expected schema
  • Stores each batch in data/pool/ with a unique timestamped filename
  • Maintains an ingestion log (JSON) for full audit trail
  • Merges all pooled data on demand for retraining

Usage:
    service = DataIngestionService()
    result  = service.ingest(df, source_label="Q1-2026-export")
    merged  = service.get_training_data()
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import (
    DATA_POOL_DIR,
    INGESTION_LOG_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# Minimum required columns (a subset — we validate what the preprocessor needs)
_REQUIRED_COLUMNS = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
}


class DataIngestionService:
    """
    Manages the data pool for incremental training.

    Each ingested CSV is stored as a separate file in data/pool/ so the
    lineage of every training row is traceable.
    """

    def __init__(self, pool_dir: Path = DATA_POOL_DIR) -> None:
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = INGESTION_LOG_PATH

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(
        self,
        df: pd.DataFrame,
        source_label: str = "upload",
    ) -> Dict:
        """
        Validate and store a new data batch.

        Parameters
        ----------
        df : DataFrame with customer records (must include required columns)
        source_label : human-readable label for audit trail

        Returns
        -------
        {
            "status": "success" | "error",
            "filename": str,
            "rows": int,
            "columns": int,
            "missing_columns": [...],  # only on error
            "timestamp": str,
        }
        """
        # Schema validation
        incoming_cols = set(df.columns)
        missing = _REQUIRED_COLUMNS - incoming_cols
        if missing:
            logger.warning("Ingestion rejected — missing columns: %s", sorted(missing))
            return {
                "status": "error",
                "message": f"Missing required columns: {sorted(missing)}",
                "missing_columns": sorted(missing),
            }

        # Generate unique filename
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{ts}_{source_label}.csv"
        filepath = self.pool_dir / filename

        # Persist
        df.to_csv(filepath, index=False)

        # Update ingestion log
        entry = {
            "filename": filename,
            "source": source_label,
            "rows": len(df),
            "columns": df.shape[1],
            "has_target": TARGET_COLUMN in df.columns,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._append_log(entry)

        logger.info(
            "Ingested %d rows from '%s' → %s",
            len(df),
            source_label,
            filepath,
        )
        return {"status": "success", **entry}

    # ── Retrieve pooled data ──────────────────────────────────────────────────

    def get_training_data(self, include_raw: bool = True) -> pd.DataFrame:
        """
        Merge all pooled CSVs (+ optionally the original raw dataset)
        into a single DataFrame for retraining.
        """
        frames: List[pd.DataFrame] = []

        # Optionally include the original dataset
        if include_raw and RAW_DATA_PATH.exists():
            frames.append(pd.read_csv(RAW_DATA_PATH))

        # Add all pooled batches
        for csv_path in sorted(self.pool_dir.glob("*.csv")):
            frames.append(pd.read_csv(csv_path))

        if not frames:
            raise FileNotFoundError("No data available — ingest at least one dataset.")

        merged = pd.concat(frames, ignore_index=True)
        logger.info(
            "Merged training data: %d rows × %d cols from %d source(s)",
            merged.shape[0],
            merged.shape[1],
            len(frames),
        )
        return merged

    # ── Pool statistics ───────────────────────────────────────────────────────

    def get_pool_stats(self) -> Dict:
        """Return summary statistics about the current data pool."""
        batches = sorted(self.pool_dir.glob("*.csv"))
        total_rows = 0
        for b in batches:
            try:
                total_rows += sum(1 for _ in open(b)) - 1  # exclude header
            except Exception:
                pass

        raw_rows = 0
        if RAW_DATA_PATH.exists():
            try:
                raw_rows = sum(1 for _ in open(RAW_DATA_PATH)) - 1
            except Exception:
                pass

        return {
            "pool_batches": len(batches),
            "pool_rows": total_rows,
            "raw_dataset_rows": raw_rows,
            "total_available_rows": total_rows + raw_rows,
            "batch_files": [b.name for b in batches],
        }

    # ── Ingestion log ─────────────────────────────────────────────────────────

    def get_ingestion_log(self) -> List[Dict]:
        """Return the full ingestion history."""
        if not self._log_path.exists():
            return []
        with open(self._log_path) as f:
            return json.load(f)

    def _append_log(self, entry: Dict) -> None:
        log = self.get_ingestion_log()
        log.append(entry)
        with open(self._log_path, "w") as f:
            json.dump(log, f, indent=2)
