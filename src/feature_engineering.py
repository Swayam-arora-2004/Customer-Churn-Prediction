"""
src/feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
FeatureEngineer — derives high-value features from preprocessed data.

Derived features (all configurable via src/config.py):
  • avg_monthly_charges   : TotalCharges / (tenure + 1)
  • service_count         : number of active add-on services
  • has_premium_services  : 1 if OnlineSecurity OR TechSupport active
  • tenure_group          : binned tenure → [new, growing, loyal, champion]

These features improve model performance and SHAP interpretability.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import List

import pandas as pd

from src.config import (
    FEATURE_ENGINEERING,
    TENURE_BINS,
    TENURE_LABELS,
)

logger = logging.getLogger(__name__)


# Columns that represent binary add-on services (post-encoding, values are 0 or 1)
_SERVICE_COLUMNS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class FeatureEngineer:
    """
    Derives high-signal features from the encoded (pre-scaled) DataFrame.

    Usage
    -----
    Call `.transform(df)` on the encoded dataset **before** scaling, so that
    raw numeric magnitudes are available for ratio calculations.

    Notes
    -----
    - Must be applied **after** `DataPreprocessor.encode()` but **before**
      `DataPreprocessor.scale()` to preserve numeric ranges.
    - New columns are appended; existing columns are never mutated.
    """

    def __init__(
        self,
        create_avg_monthly_charges: bool = FEATURE_ENGINEERING[
            "create_avg_monthly_charges"
        ],
        create_service_count: bool = FEATURE_ENGINEERING["create_service_count"],
        create_has_premium_services: bool = FEATURE_ENGINEERING[
            "create_has_premium_services"
        ],
        create_tenure_group: bool = FEATURE_ENGINEERING["create_tenure_group"],
    ) -> None:
        self.create_avg_monthly_charges = create_avg_monthly_charges
        self.create_service_count = create_service_count
        self.create_has_premium_services = create_has_premium_services
        self.create_tenure_group = create_tenure_group
        self.new_feature_names_: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all enabled feature engineering steps.

        Parameters
        ----------
        df : encoded (but not yet scaled) DataFrame

        Returns
        -------
        DataFrame with additional derived feature columns appended.
        """
        df = df.copy()
        added: List[str] = []

        if self.create_avg_monthly_charges and self._cols_exist(
            df, ["TotalCharges", "tenure"]
        ):
            df["avg_monthly_charges"] = df["TotalCharges"] / (df["tenure"] + 1)
            added.append("avg_monthly_charges")

        if self.create_service_count:
            service_cols = [c for c in _SERVICE_COLUMNS if c in df.columns]
            if service_cols:
                df["service_count"] = df[service_cols].sum(axis=1)
                added.append("service_count")

        if self.create_has_premium_services:
            premium_cols = [
                c for c in ["OnlineSecurity", "TechSupport"] if c in df.columns
            ]
            if premium_cols:
                df["has_premium_services"] = df[premium_cols].any(axis=1).astype(int)
                added.append("has_premium_services")

        if self.create_tenure_group and "tenure" in df.columns:
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=TENURE_BINS,
                labels=TENURE_LABELS,
                right=False,
            )
            # One-hot encode tenure_group
            dummies = pd.get_dummies(
                df["tenure_group"], prefix="tenure_group", dtype=int
            )
            df = pd.concat([df.drop(columns=["tenure_group"]), dummies], axis=1)
            added.extend(dummies.columns.tolist())

        self.new_feature_names_ = added
        logger.info("Feature engineering added %d features: %s", len(added), added)
        return df

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _cols_exist(df: pd.DataFrame, cols: List[str]) -> bool:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(
                "Feature engineering skipped — columns not found: %s", missing
            )
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"FeatureEngineer("
            f"avg_monthly={self.create_avg_monthly_charges}, "
            f"service_count={self.create_service_count}, "
            f"premium={self.create_has_premium_services}, "
            f"tenure_group={self.create_tenure_group})"
        )
