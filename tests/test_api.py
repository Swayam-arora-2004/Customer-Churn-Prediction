"""
tests/test_api.py
─────────────────────────────────────────────────────────────────────────────
API integration tests using Flask test client.

Tests run against `create_app()` with mocked model artefacts so no
trained model files are required on disk.
─────────────────────────────────────────────────────────────────────────────
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.app import create_app


# ── Sample payload ────────────────────────────────────────────────────────────

VALID_FEATURES = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
}

VALID_PREDICT_PAYLOAD = {
    "customer_id": "TEST-001",
    "features": VALID_FEATURES,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model():
    """Mock sklearn-compatible model."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65]])
    return model


@pytest.fixture
def mock_preprocessor():
    """Mock DataPreprocessor that returns a single-row DataFrame."""
    preprocessor = MagicMock()
    preprocessor.is_fitted = True
    preprocessor.feature_names_ = [f"f{i}" for i in range(10)]
    preprocessor.transform.return_value = pd.DataFrame(
        np.zeros((1, 10)), columns=[f"f{i}" for i in range(10)]
    )
    return preprocessor


@pytest.fixture
def mock_metadata():
    return {
        "model_name": "xgboost",
        "model_class": "XGBClassifier",
        "feature_count": 10,
        "test_roc_auc": 0.88,
        "test_f1": 0.73,
        "test_precision": 0.69,
        "test_recall": 0.78,
        "trained_at": "2026-04-27T00:00:00",
    }


@pytest.fixture
def client(mock_model, mock_preprocessor, mock_metadata, tmp_path):
    """Flask test client with all model artefacts mocked."""
    with (
        patch("app.app.ModelTrainer.load_best_model", return_value=mock_model),
        patch("app.app.DataPreprocessor.load", return_value=mock_preprocessor),
        patch("app.app.ModelTrainer.load_metadata", return_value=mock_metadata),
        patch("app.app.SHAPExplainer", return_value=None),
        patch("app.app.AUDIT_DB_PATH", tmp_path / "test.db"),
    ):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


# ── /v1/health ────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200

    def test_health_status_is_success(self, client):
        data = resp_json(client.get("/v1/health"))
        assert data["status"] == "success"

    def test_health_contains_model_name(self, client):
        data = resp_json(client.get("/v1/health"))
        assert data["data"]["model_name"] == "xgboost"

    def test_health_contains_uptime(self, client):
        data = resp_json(client.get("/v1/health"))
        assert "uptime_seconds" in data["data"]

    def test_health_roc_auc_value(self, client):
        data = resp_json(client.get("/v1/health"))
        assert data["data"]["test_roc_auc"] == 0.88


# ── /v1/metrics ───────────────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_returns_200(self, client):
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_required_fields(self, client):
        data = resp_json(client.get("/v1/metrics"))["data"]
        for field in ["test_roc_auc", "test_f1", "test_precision", "test_recall"]:
            assert field in data


# ── /v1/predict ───────────────────────────────────────────────────────────────


class TestPredict:
    def test_valid_predict_returns_200(self, client):
        resp = client.post(
            "/v1/predict",
            json=VALID_PREDICT_PAYLOAD,
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_predict_returns_probability(self, client):
        data = resp_json(client.post("/v1/predict", json=VALID_PREDICT_PAYLOAD))["data"]
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_returns_risk_segment(self, client):
        data = resp_json(client.post("/v1/predict", json=VALID_PREDICT_PAYLOAD))["data"]
        assert data["risk_segment"] in ("High Risk", "Medium Risk", "Low Risk")

    def test_predict_probability_matches_mock(self, client):
        data = resp_json(client.post("/v1/predict", json=VALID_PREDICT_PAYLOAD))["data"]
        # mock returns prob=0.65
        assert abs(data["churn_probability"] - 0.65) < 0.01

    def test_predict_high_prob_high_risk(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.20, 0.80]])
        data = resp_json(client.post("/v1/predict", json=VALID_PREDICT_PAYLOAD))["data"]
        assert data["risk_segment"] == "High Risk"
        assert data["will_churn"] is True

    def test_predict_low_prob_low_risk(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.90, 0.10]])
        data = resp_json(client.post("/v1/predict", json=VALID_PREDICT_PAYLOAD))["data"]
        assert data["risk_segment"] == "Low Risk"
        assert data["will_churn"] is False

    def test_predict_missing_features_422(self, client):
        resp = client.post("/v1/predict", json={"customer_id": "X"})
        assert resp.status_code == 422

    def test_predict_invalid_gender_422(self, client):
        bad_payload = dict(VALID_PREDICT_PAYLOAD)
        bad_features = dict(VALID_FEATURES, gender="Unknown")
        bad_payload["features"] = bad_features
        resp = client.post("/v1/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_predict_negative_tenure_422(self, client):
        bad_payload = dict(VALID_PREDICT_PAYLOAD)
        bad_features = dict(VALID_FEATURES, tenure=-5)
        bad_payload["features"] = bad_features
        resp = client.post("/v1/predict", json=bad_payload)
        assert resp.status_code == 422


# ── /v1/predict/batch ────────────────────────────────────────────────────────


class TestPredictBatch:
    def test_batch_predict_returns_200(self, client):
        resp = client.post(
            "/v1/predict/batch",
            json={"customers": [VALID_PREDICT_PAYLOAD]},
        )
        assert resp.status_code == 200

    def test_batch_predict_returns_totals(self, client):
        data = resp_json(
            client.post(
                "/v1/predict/batch",
                json={"customers": [VALID_PREDICT_PAYLOAD, VALID_PREDICT_PAYLOAD]},
            )
        )["data"]
        assert data["total"] == 2
        assert "high_risk" in data
        assert "predictions" in data

    def test_batch_empty_list_422(self, client):
        resp = client.post("/v1/predict/batch", json={"customers": []})
        assert resp.status_code == 422


# ── /v1/recommend ────────────────────────────────────────────────────────────


class TestRecommend:
    def test_recommend_returns_200(self, client):
        resp = client.post("/v1/recommend", json=VALID_PREDICT_PAYLOAD)
        assert resp.status_code == 200

    def test_recommend_has_recommendations_list(self, client):
        data = resp_json(client.post("/v1/recommend", json=VALID_PREDICT_PAYLOAD))[
            "data"
        ]
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    def test_recommend_has_segment(self, client):
        data = resp_json(client.post("/v1/recommend", json=VALID_PREDICT_PAYLOAD))[
            "data"
        ]
        assert data["customer_segment"] in ("High Risk", "Medium Risk", "Low Risk")

    def test_recommend_has_retention_lift(self, client):
        data = resp_json(client.post("/v1/recommend", json=VALID_PREDICT_PAYLOAD))[
            "data"
        ]
        assert 0.0 <= data["estimated_retention_lift"] <= 1.0


# ── 404 / 405 handlers ────────────────────────────────────────────────────────


class TestErrorHandlers:
    def test_unknown_route_404(self, client):
        resp = client.get("/v1/nonexistent")
        assert resp.status_code == 404

    def test_wrong_method_405(self, client):
        resp = client.get("/v1/predict")
        assert resp.status_code == 405


# ── Utility ───────────────────────────────────────────────────────────────────


def resp_json(resp) -> dict:
    return json.loads(resp.data)
