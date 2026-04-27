"""
src/data_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
DataPreprocessor — sklearn-compatible transformer for the Telco Churn dataset.

Responsibilities:
  • Load raw CSV
  • Clean data (TotalCharges blank → 0.0, drop customerID)
  • Encode categorical & binary features
  • Scale numeric features
  • Stratified train/test split
  • Persist processed splits to data/processed/

Usage:
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split()
    preprocessor.save(config.PREPROCESSOR_PATH)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_X_TRAIN_PATH,
    PROCESSED_X_TEST_PATH,
    PROCESSED_Y_TRAIN_PATH,
    PROCESSED_Y_TEST_PATH,
    FEATURE_NAMES_PATH,
    PREPROCESSOR_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    BINARY_COLUMNS,
    BINARY_SERVICE_COLUMNS,
    CATEGORICAL_COLUMNS,
    PREPROCESSING,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    End-to-end data preprocessing pipeline for the Telco Churn dataset.

    Follows sklearn's fit/transform pattern so it can be embedded in
    Pipeline objects and reused consistently at inference time.

    Attributes
    ----------
    scaler : fitted sklearn scaler (StandardScaler by default)
    feature_names_ : list[str] — ordered feature names after encoding
    is_fitted : bool
    """

    _SCALER_MAP = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(
        self,
        scaler_type: str = PREPROCESSING["scaler"],
        test_size: float = PREPROCESSING["test_size"],
        random_state: int = PREPROCESSING["random_state"],
        stratify: bool = PREPROCESSING["stratify"],
    ) -> None:
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        scaler_cls = self._SCALER_MAP.get(scaler_type, StandardScaler)
        self.scaler = scaler_cls()
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

        logger.info(
            "DataPreprocessor initialised | scaler=%s test_size=%.2f seed=%d",
            scaler_type,
            test_size,
            random_state,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_raw(self, path: Path = RAW_DATA_PATH) -> pd.DataFrame:
        """Load raw CSV from disk."""
        logger.info("Loading raw data from %s", path)
        df = pd.read_csv(path)
        logger.info("Raw data loaded | shape=%s", df.shape)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning steps to raw DataFrame.

        Steps
        -----
        1. Drop the ID column (no predictive value)
        2. Fix TotalCharges: blank strings → 0.0, cast to float
        3. Drop duplicate rows
        4. Reset index
        """
        logger.info("Cleaning data …")
        df = df.copy()

        # 1. Drop identifier
        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])
            logger.debug("Dropped column '%s'", ID_COLUMN)

        # 2. Fix TotalCharges (stored as object with blank strings)
        if "TotalCharges" in df.columns:
            before = df.shape[0]
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            blank_count = df["TotalCharges"].isna().sum()
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
            logger.info(
                "TotalCharges: fixed %d blank/non-numeric values → 0.0", blank_count
            )
            assert df.shape[0] == before, "Row count changed during TotalCharges fix"

        # 3. Drop exact duplicates
        dupes = df.duplicated().sum()
        if dupes:
            df = df.drop_duplicates()
            logger.warning("Dropped %d duplicate rows", dupes)

        # 4. Reset index
        df = df.reset_index(drop=True)

        logger.info("Cleaning complete | shape=%s", df.shape)
        return df

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode all categorical features.

        Encoding strategy
        -----------------
        • Binary Yes/No columns            → {Yes: 1, No: 0}
        • Binary service columns           → simplify 'No X service' → 'No', then {Yes:1, No:0}
        • gender                           → {Male: 1, Female: 0}
        • SeniorCitizen                    → already 0/1 int, cast to int
        • Multi-class (InternetService,    → one-hot encoding (drop_first=False to keep
          Contract, PaymentMethod)            all categories visible for SHAP)
        • Churn (target)                   → {Yes: 1, No: 0}
        """
        logger.info("Encoding features …")
        df = df.copy()

        binary_map = PREPROCESSING["binary_map"]
        service_no_map = PREPROCESSING["service_no_map"]

        # Binary Yes/No columns (includes Churn)
        for col in BINARY_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map(binary_map).astype(int)

        # Binary service columns (with 'No X service' variants)
        for col in BINARY_SERVICE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].replace(service_no_map).map(binary_map).astype(int)

        # gender → binary
        if "gender" in df.columns:
            df["gender"] = (df["gender"] == "Male").astype(int)

        # SeniorCitizen → ensure int
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

        # Multi-class one-hot encoding
        ohe_columns = [
            c for c in ["InternetService", "Contract", "PaymentMethod"]
            if c in df.columns
        ]
        if ohe_columns:
            df = pd.get_dummies(df, columns=ohe_columns, drop_first=False, dtype=int)
            logger.debug("One-hot encoded: %s", ohe_columns)

        logger.info("Encoding complete | shape=%s", df.shape)
        return df

    def scale(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Scale numeric columns using the configured scaler.

        Parameters
        ----------
        df   : encoded DataFrame
        fit  : if True, fit the scaler before transforming (training set).
               if False, only transform (validation/test/inference set).
        """
        logger.info("Scaling numeric features (fit=%s) …", fit)
        df = df.copy()

        # Only scale columns that exist in this df
        numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise RuntimeError(
                    "Scaler has not been fitted. Call scale(fit=True) on training data first."
                )
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        logger.info("Scaling complete")
        return df

    def fit_transform_split(
        self,
        raw_path: Path = RAW_DATA_PATH,
        save: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Full pipeline: load → clean → encode → split → scale → (optionally save).

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        # 1. Load
        df = self.load_raw(raw_path)

        # 2. Clean
        df = self.clean(df)

        # 3. Encode
        df = self.encode(df)

        # 4. Split features / target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # 5. Train / test split
        stratify_col = y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col,
        )
        logger.info(
            "Split | train=%d test=%d | churn_rate_train=%.3f churn_rate_test=%.3f",
            len(X_train),
            len(X_test),
            y_train.mean(),
            y_test.mean(),
        )

        # 6. Scale (fit on train, apply to test)
        X_train = self.scale(X_train, fit=True)
        X_test = self.scale(X_test, fit=False)

        # 7. Store feature names
        self.feature_names_ = X_train.columns.tolist()

        # 8. Save processed data
        if save:
            self._save_splits(X_train, X_test, y_train, y_test)
            self._save_feature_names()

        return X_train, X_test, y_train, y_test

    def transform(
        self,
        df: pd.DataFrame,
        include_id: bool = False,
    ) -> pd.DataFrame:
        """
        Transform a single-row or multi-row inference DataFrame.

        Parameters
        ----------
        df         : raw input DataFrame (as received by the API)
        include_id : if True, preserve the ID column for downstream tracing

        Returns
        -------
        Encoded + scaled DataFrame aligned to training feature names
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before calling transform(). "
                "Load a saved preprocessor or run fit_transform_split() first."
            )

        customer_ids: Optional[pd.Series] = None
        df = df.copy()

        if include_id and ID_COLUMN in df.columns:
            customer_ids = df[ID_COLUMN].copy()

        df = self.clean(df)
        df = self.encode(df)

        # Drop target if it slipped through
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        df = self.scale(df, fit=False)

        # Align columns to training schema (add missing as 0, drop extras)
        df = df.reindex(columns=self.feature_names_, fill_value=0)

        if include_id and customer_ids is not None:
            df.insert(0, ID_COLUMN, customer_ids.values)

        return df

    def save(self, path: Path = PREPROCESSOR_PATH) -> None:
        """Persist the fitted preprocessor to disk using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted; nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Preprocessor saved → %s", path)

    @classmethod
    def load(cls, path: Path = PREPROCESSOR_PATH) -> "DataPreprocessor":
        """Load a previously fitted preprocessor from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {path}")
        preprocessor: DataPreprocessor = joblib.load(path)
        logger.info("Preprocessor loaded ← %s", path)
        return preprocessor

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _save_splits(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Persist processed train/test splits to CSV."""
        X_train.to_csv(PROCESSED_X_TRAIN_PATH, index=False)
        X_test.to_csv(PROCESSED_X_TEST_PATH, index=False)
        y_train.to_csv(PROCESSED_Y_TRAIN_PATH, index=False)
        y_test.to_csv(PROCESSED_Y_TEST_PATH, index=False)
        logger.info(
            "Processed data saved → %s", PROCESSED_X_TRAIN_PATH.parent
        )

    def _save_feature_names(self) -> None:
        """Persist feature names as JSON for downstream alignment checks."""
        FEATURE_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURE_NAMES_PATH, "w") as f:
            json.dump(self.feature_names_, f, indent=2)
        logger.debug("Feature names saved → %s", FEATURE_NAMES_PATH)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DataPreprocessor("
            f"scaler={self.scaler_type!r}, "
            f"test_size={self.test_size}, "
            f"fitted={self.is_fitted}, "
            f"features={len(self.feature_names_)})"
        )


# ── Module-level entry point ──────────────────────────────────────────────────

def run_preprocessing(
    raw_path: Path = RAW_DATA_PATH,
    save_preprocessor: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function: run the full preprocessing pipeline and save all artifacts.

    Usage
    -----
    >>> from src.data_preprocessing import run_preprocessing
    >>> X_train, X_test, y_train, y_test = run_preprocessing()
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(
        raw_path=raw_path, save=True
    )

    if save_preprocessor:
        preprocessor.save()

    logger.info("Preprocessing pipeline completed successfully.")
    logger.info(
        "  X_train: %s  |  X_test: %s  |  Features: %d",
        X_train.shape,
        X_test.shape,
        len(preprocessor.feature_names_),
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
