"""
tests/test_data_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/data_preprocessing.py and src/feature_engineering.py.

Run with:
    pytest tests/test_data_preprocessing.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_raw_df() -> pd.DataFrame:
    """
    Minimal synthetic raw dataframe matching the Telco Churn schema.
    Includes edge cases: blank TotalCharges, a duplicate row, both churn values.
    """
    return pd.DataFrame(
        {
            "customerID": ["A001", "A002", "A003", "A003"],  # A003 duplicated
            "gender": ["Male", "Female", "Male", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "No", "No"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "tenure": [1, 34, 0, 0],  # tenure=0 → blank TotalCharges
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service", "No internet service"],
            "TechSupport": ["No", "No", "No internet service", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 53.85],
            "TotalCharges": ["29.85", "1889.5", " ", " "],  # blank TotalCharges
            "Churn": ["No", "No", "Yes", "Yes"],
        }
    )


@pytest.fixture
def preprocessor():
    """Return a fresh DataPreprocessor instance."""
    from src.data_preprocessing import DataPreprocessor
    return DataPreprocessor(test_size=0.5, random_state=42)


@pytest.fixture
def cleaned_df(preprocessor, minimal_raw_df):
    """Return a cleaned (not yet encoded) copy of the minimal dataframe."""
    return preprocessor.clean(minimal_raw_df)


@pytest.fixture
def encoded_df(preprocessor, minimal_raw_df):
    """Return a cleaned + encoded copy of the minimal dataframe."""
    cleaned = preprocessor.clean(minimal_raw_df)
    return preprocessor.encode(cleaned)


# ── DataPreprocessor.clean() ──────────────────────────────────────────────────


class TestClean:
    def test_drops_customer_id(self, cleaned_df):
        assert "customerID" not in cleaned_df.columns

    def test_total_charges_is_float(self, cleaned_df):
        assert cleaned_df["TotalCharges"].dtype == float

    def test_blank_total_charges_becomes_zero(self, cleaned_df):
        """Customers with blank TotalCharges (tenure=0) should get 0.0."""
        assert (cleaned_df["TotalCharges"] == 0.0).any()

    def test_no_missing_values(self, cleaned_df):
        assert cleaned_df.isnull().sum().sum() == 0

    def test_duplicates_removed(self, minimal_raw_df, preprocessor):
        """The fixture has one exact duplicate row — it should be dropped."""
        cleaned = preprocessor.clean(minimal_raw_df)
        assert cleaned.shape[0] == 3  # 4 rows - 1 duplicate

    def test_row_count_preserved_no_dupes(self, preprocessor):
        """When there are no duplicates, all rows are preserved."""
        df = pd.DataFrame(
            {
                "customerID": ["X1"],
                "TotalCharges": ["100.50"],
                "Churn": ["No"],
            }
        )
        cleaned = preprocessor.clean(df)
        assert cleaned.shape[0] == 1


# ── DataPreprocessor.encode() ────────────────────────────────────────────────


class TestEncode:
    def test_churn_is_binary(self, encoded_df):
        assert set(encoded_df["Churn"].unique()).issubset({0, 1})

    def test_partner_is_binary(self, encoded_df):
        assert set(encoded_df["Partner"].unique()).issubset({0, 1})

    def test_gender_is_binary(self, encoded_df):
        assert set(encoded_df["gender"].unique()).issubset({0, 1})

    def test_multiple_lines_no_phone_service_mapped_to_zero(self, encoded_df):
        """'No phone service' should map to 0 (same as No)."""
        assert 0 in encoded_df["MultipleLines"].values

    def test_one_hot_internet_service(self, encoded_df):
        """InternetService should be one-hot encoded."""
        ohe_cols = [c for c in encoded_df.columns if c.startswith("InternetService_")]
        assert len(ohe_cols) >= 2

    def test_one_hot_contract(self, encoded_df):
        ohe_cols = [c for c in encoded_df.columns if c.startswith("Contract_")]
        assert len(ohe_cols) >= 2

    def test_no_string_columns_remain(self, encoded_df):
        string_cols = encoded_df.select_dtypes(include="object").columns.tolist()
        assert string_cols == [], f"String columns remain: {string_cols}"

    def test_original_columns_replaced_by_ohe(self, encoded_df):
        assert "InternetService" not in encoded_df.columns
        assert "Contract" not in encoded_df.columns
        assert "PaymentMethod" not in encoded_df.columns


# ── DataPreprocessor.scale() ─────────────────────────────────────────────────


class TestScale:
    def test_scale_fit_reduces_mean_to_near_zero(self, preprocessor, encoded_df):
        X = encoded_df.drop(columns=["Churn"])
        scaled = preprocessor.scale(X, fit=True)
        # StandardScaler: mean of numeric cols ≈ 0
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        numeric_cols = [c for c in numeric_cols if c in scaled.columns]
        for col in numeric_cols:
            assert abs(scaled[col].mean()) < 1.0  # mean near 0 after standard scaling

    def test_scale_transform_fails_without_fit(self, preprocessor, encoded_df):
        from src.data_preprocessing import DataPreprocessor
        fresh = DataPreprocessor()
        X = encoded_df.drop(columns=["Churn"])
        with pytest.raises(RuntimeError, match="Scaler has not been fitted"):
            fresh.scale(X, fit=False)

    def test_scale_no_new_columns(self, preprocessor, encoded_df):
        X = encoded_df.drop(columns=["Churn"])
        n_cols_before = X.shape[1]
        scaled = preprocessor.scale(X, fit=True)
        assert scaled.shape[1] == n_cols_before


# ── DataPreprocessor save / load ──────────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, preprocessor, encoded_df):
        X = encoded_df.drop(columns=["Churn"])
        preprocessor.scale(X, fit=True)  # fit the scaler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preprocessor.pkl"
            preprocessor.save(path)
            assert path.exists()

            from src.data_preprocessing import DataPreprocessor
            loaded = DataPreprocessor.load(path)
            assert loaded.is_fitted
            assert loaded.scaler_type == preprocessor.scaler_type

    def test_save_fails_when_not_fitted(self, preprocessor):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preprocessor.pkl"
            with pytest.raises(RuntimeError, match="not fitted"):
                preprocessor.save(path)

    def test_load_fails_bad_path(self):
        from src.data_preprocessing import DataPreprocessor
        with pytest.raises(FileNotFoundError):
            DataPreprocessor.load(Path("/nonexistent/preprocessor.pkl"))


# ── DataPreprocessor.transform() (inference path) ────────────────────────────


class TestTransform:
    def test_transform_aligns_to_training_features(
        self, preprocessor, minimal_raw_df
    ):
        """Inference transform must produce exactly the same columns as training."""
        from src.data_preprocessing import DataPreprocessor
        # We need a larger dataset to do a real split — use duplicated minimal data
        big_df = pd.concat([minimal_raw_df] * 20, ignore_index=True)
        # Assign unique customerIDs
        big_df["customerID"] = [f"C{i:03d}" for i in range(len(big_df))]
        big_df = big_df.drop_duplicates(subset=["customerID"])

        with tempfile.TemporaryDirectory() as tmpdir:
            import src.config as cfg
            # Temporarily override paths
            orig_x_train = cfg.PROCESSED_X_TRAIN_PATH
            orig_x_test = cfg.PROCESSED_X_TEST_PATH
            orig_y_train = cfg.PROCESSED_Y_TRAIN_PATH
            orig_y_test = cfg.PROCESSED_Y_TEST_PATH

            cfg.PROCESSED_X_TRAIN_PATH = Path(tmpdir) / "X_train.csv"
            cfg.PROCESSED_X_TEST_PATH = Path(tmpdir) / "X_test.csv"
            cfg.PROCESSED_Y_TRAIN_PATH = Path(tmpdir) / "y_train.csv"
            cfg.PROCESSED_Y_TEST_PATH = Path(tmpdir) / "y_test.csv"

            import io, os
            raw_csv = io.StringIO(big_df.to_csv(index=False))
            raw_path = Path(tmpdir) / "raw.csv"
            big_df.to_csv(raw_path, index=False)

            p = DataPreprocessor(test_size=0.3, random_state=0)
            X_train, _, _, _ = p.fit_transform_split(raw_path=raw_path, save=True)

            # Inference on a single row
            single_row = minimal_raw_df.iloc[[0]].copy()
            transformed = p.transform(single_row)
            assert list(transformed.columns) == p.feature_names_

            # Restore
            cfg.PROCESSED_X_TRAIN_PATH = orig_x_train
            cfg.PROCESSED_X_TEST_PATH = orig_x_test
            cfg.PROCESSED_Y_TRAIN_PATH = orig_y_train
            cfg.PROCESSED_Y_TEST_PATH = orig_y_test


# ── FeatureEngineer ───────────────────────────────────────────────────────────


class TestFeatureEngineer:
    """Tests for derived feature creation."""

    @pytest.fixture
    def encoded_no_churn(self, preprocessor, minimal_raw_df):
        cleaned = preprocessor.clean(minimal_raw_df)
        encoded = preprocessor.encode(cleaned)
        return encoded.drop(columns=["Churn"])

    def test_avg_monthly_charges_created(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.transform(encoded_no_churn)
        assert "avg_monthly_charges" in result.columns

    def test_avg_monthly_charges_formula(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.transform(encoded_no_churn)
        expected = encoded_no_churn["TotalCharges"] / (encoded_no_churn["tenure"] + 1)
        pd.testing.assert_series_equal(
            result["avg_monthly_charges"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_service_count_non_negative(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.transform(encoded_no_churn)
        assert (result["service_count"] >= 0).all()

    def test_service_count_max_7(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.transform(encoded_no_churn)
        assert (result["service_count"] <= 7).all()

    def test_has_premium_services_is_binary(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.transform(encoded_no_churn)
        assert set(result["has_premium_services"].unique()).issubset({0, 1})

    def test_tenure_group_columns_created(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.transform(encoded_no_churn)
        tenure_group_cols = [c for c in result.columns if c.startswith("tenure_group_")]
        assert len(tenure_group_cols) == 4  # new, growing, loyal, champion

    def test_no_string_columns_after_fe(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.transform(encoded_no_churn)
        string_cols = result.select_dtypes(include="object").columns.tolist()
        assert string_cols == []

    def test_original_columns_preserved(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        result = fe.transform(encoded_no_churn)
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in encoded_no_churn.columns:
                assert col in result.columns

    def test_disabled_features_not_created(self, encoded_no_churn):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(
            create_avg_monthly_charges=False,
            create_service_count=False,
            create_has_premium_services=False,
            create_tenure_group=False,
        )
        result = fe.transform(encoded_no_churn)
        assert "avg_monthly_charges" not in result.columns
        assert "service_count" not in result.columns
        assert "has_premium_services" not in result.columns
        assert not any(c.startswith("tenure_group_") for c in result.columns)
