"""
src/monitoring.py
─────────────────────────────────────────────────────────────────────────────
DataDriftDetector & ModelPerformanceMonitor

Detects distribution shifts between training data and incoming inference data,
and tracks model output score distributions over time.

Drift detection uses Population Stability Index (PSI):
  PSI < 0.10  → No significant change
  PSI 0.10–0.20 → Moderate drift — monitor closely
  PSI > 0.20  → Significant drift — consider retraining

─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import MONITORING, PROCESSED_X_TRAIN_PATH

logger = logging.getLogger(__name__)

_PSI_BINS = 10  # Number of bins for PSI calculation


# ── PSI Utilities ─────────────────────────────────────────────────────────────

def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = _PSI_BINS,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    Parameters
    ----------
    reference : 1D array of the reference distribution (training data)
    current   : 1D array of the current distribution (inference data)

    Returns
    -------
    PSI value (float). Higher → more drift.
    """
    eps = 1e-6

    # Use reference quantiles as bin edges
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.nanpercentile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts / (len(reference) + eps)) + eps
    cur_pct = (cur_counts / (len(current) + eps)) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def _drift_label(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.20:
        return "moderate_drift"
    return "significant_drift"


# ── DataDriftDetector ─────────────────────────────────────────────────────────


class DataDriftDetector:
    """
    Compares incoming inference data against the training distribution.

    Usage
    -----
    detector = DataDriftDetector()
    report = detector.detect(new_df)
    """

    def __init__(
        self,
        reference_path: Path = PROCESSED_X_TRAIN_PATH,
        psi_threshold: float = MONITORING["drift_psi_threshold"],
    ) -> None:
        self.psi_threshold = psi_threshold
        self._reference: Optional[pd.DataFrame] = None

        if Path(reference_path).exists():
            self._reference = pd.read_csv(reference_path)
            logger.info(
                "DataDriftDetector: reference data loaded | shape=%s",
                self._reference.shape,
            )
        else:
            logger.warning(
                "DataDriftDetector: reference data not found at %s", reference_path
            )

    def detect(self, current_df: pd.DataFrame) -> Dict:
        """
        Compute PSI for every numeric column in current_df vs. reference.

        Returns
        -------
        {
          "timestamp": str,
          "overall_drift_detected": bool,
          "columns": {
            "feature_name": {"psi": float, "status": str},
            ...
          },
          "drifted_columns": [str, ...]
        }
        """
        if self._reference is None:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_drift_detected": False,
                "error": "Reference data not available.",
                "columns": {},
                "drifted_columns": [],
            }

        common_cols = [
            c for c in current_df.columns
            if c in self._reference.columns
            and pd.api.types.is_numeric_dtype(current_df[c])
        ]

        col_results = {}
        drifted = []

        for col in common_cols:
            ref_vals = self._reference[col].dropna().values
            cur_vals = current_df[col].dropna().values
            if len(cur_vals) < 5:
                continue
            psi = _compute_psi(ref_vals, cur_vals)
            status = _drift_label(psi)
            col_results[col] = {"psi": round(psi, 6), "status": status}
            if psi >= self.psi_threshold:
                drifted.append(col)

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_drift_detected": len(drifted) > 0,
            "drifted_columns": drifted,
            "columns": col_results,
        }

        if drifted:
            logger.warning(
                "Data drift detected in %d columns: %s", len(drifted), drifted
            )
        else:
            logger.info("No significant data drift detected.")

        return report


# ── ModelPerformanceMonitor ───────────────────────────────────────────────────


class ModelPerformanceMonitor:
    """
    Tracks prediction score distributions over time.

    Emits an alert when the mean predicted churn probability deviates
    significantly from the training baseline (possible model degradation).
    """

    def __init__(
        self,
        alert_threshold: float = MONITORING["churn_score_alert_threshold"],
    ) -> None:
        self.alert_threshold = alert_threshold
        self._history: List[Dict] = []

    def record(
        self,
        predictions: np.ndarray,
        label: str = "batch",
    ) -> Dict:
        """
        Record a batch of churn probability predictions.

        Parameters
        ----------
        predictions : 1D array of churn probabilities [0, 1]
        label       : descriptive label for this batch

        Returns
        -------
        Summary statistics dict
        """
        proba = np.asarray(predictions)
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "label": label,
            "n": int(len(proba)),
            "mean": float(np.mean(proba)),
            "std": float(np.std(proba)),
            "p25": float(np.percentile(proba, 25)),
            "p50": float(np.percentile(proba, 50)),
            "p75": float(np.percentile(proba, 75)),
            "pct_high_risk": float((proba >= 0.70).mean()),
            "alert": bool(np.mean(proba) >= self.alert_threshold),
        }
        self._history.append(stats)

        if stats["alert"]:
            logger.warning(
                "PERFORMANCE ALERT | mean_churn_prob=%.3f exceeds threshold=%.2f | label=%s",
                stats["mean"],
                self.alert_threshold,
                label,
            )
        else:
            logger.info(
                "Prediction stats | n=%d mean=%.3f std=%.3f pct_high_risk=%.3f",
                stats["n"],
                stats["mean"],
                stats["std"],
                stats["pct_high_risk"],
            )

        return stats

    def summary(self) -> List[Dict]:
        """Return all recorded snapshots."""
        return self._history

    def export(self, path: Path) -> None:
        """Export history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2)
        logger.info("Monitor history exported → %s", path)
