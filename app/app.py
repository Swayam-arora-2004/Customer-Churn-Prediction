"""
app/app.py
─────────────────────────────────────────────────────────────────────────────
Flask REST API — Customer Churn Prediction + Prevention System

Endpoints:
  GET  /v1/health                  → Model health & metadata
  GET  /v1/metrics                 → Model performance metrics
  POST /v1/predict                 → Single customer churn prediction
  POST /v1/predict/batch           → Batch prediction (up to 1000 rows)
  GET  /v1/explain/<customer_id>   → SHAP explanation (from recent predictions)
  POST /v1/explain                 → SHAP explanation for inline features
  POST /v1/recommend               → Retention recommendations for inline features

Run:
  python app/app.py
  gunicorn -w 4 -b 0.0.0.0:5000 "app.app:create_app()"
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional

# ── Path fix ──────────────────────────────────────────────────────────────────
# When run as `python app/app.py`, Python sets sys.path[0] = 'app/'
# so `from app.xxx` and `from src.xxx` fail.
# Inserting the project root fixes both invocation styles:
#   python app/app.py          ← script mode (needs this fix)
#   python -m app.app          ← module mode (root already on path)
#   gunicorn "app.app:create_app()" ← production (root already on path)
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import pandas as pd
from flask import Flask, g, jsonify, request
from pydantic import ValidationError

from app.logger import get_logger
from app.schemas import (
    APIResponse,
    BatchPredictRequest,
    PredictRequest,
)
from src.config import API, AUDIT_DB_PATH, BEST_MODEL_PATH, PREPROCESSOR_PATH
from src.data_preprocessing import DataPreprocessor
from src.explainability import SHAPExplainer
from src.model_training import ModelTrainer
from src.prevention import RetentionEngine

logger = get_logger("churn_api")

_START_TIME = time.time()


# ── Application Factory ───────────────────────────────────────────────────────


def create_app() -> Flask:
    """Application factory — creates and configures the Flask app."""
    app = Flask(__name__)

    # ── Load model artefacts ──────────────────────────────────────────────────
    try:
        model = ModelTrainer.load_best_model(BEST_MODEL_PATH)
        preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
        metadata = ModelTrainer.load_metadata()
        logger.info(
            "Model loaded | name=%s AUC=%.4f",
            metadata.get("model_name", "unknown"),
            metadata.get("test_roc_auc", 0),
        )
    except FileNotFoundError as exc:
        logger.error("Model artefacts not found: %s", exc)
        model = None
        preprocessor = None
        metadata = {}

    # Load training data for SHAP background (once at startup)
    shap_explainer: Optional[SHAPExplainer] = None
    if model is not None and preprocessor is not None:
        try:
            from src.config import PROCESSED_X_TRAIN_PATH
            _bg = pd.read_csv(PROCESSED_X_TRAIN_PATH)
            n_bg = min(100, len(_bg))
            X_train_bg = _bg.sample(n_bg, random_state=42)
            shap_explainer = SHAPExplainer(model, X_train_bg)
            logger.info("SHAP explainer initialised.")
        except Exception as exc:
            logger.warning("SHAP explainer could not be initialised: %s", exc)

    retention_engine = RetentionEngine()

    # ── Audit DB ──────────────────────────────────────────────────────────────
    _init_db()

    # ── Request middleware ────────────────────────────────────────────────────

    @app.before_request
    def _before():
        g.start_time = time.time()
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    @app.after_request
    def _after(response):
        duration_ms = round((time.time() - g.start_time) * 1000, 2)
        logger.info(
            "request | method=%s path=%s status=%d duration_ms=%.2f request_id=%s",
            request.method,
            request.path,
            response.status_code,
            duration_ms,
            g.request_id,
        )
        response.headers["X-Request-ID"] = g.request_id
        response.headers["X-Response-Time-ms"] = str(duration_ms)
        return response

    # ── Helper: standardised response ────────────────────────────────────────

    def _ok(data: Any):
        return jsonify({"status": "success", "data": data,
                        "request_id": g.request_id}), 200

    def _err(message: str, code: int = 400):
        logger.warning("API error | %s", message)
        return jsonify({"status": "error", "error": message,
                        "request_id": g.request_id}), code

    def _model_required():
        if model is None or preprocessor is None:
            return _err("Model artefacts not available. Run training pipeline first.", 503)
        return None

    # ── Helper: parse & transform features ───────────────────────────────────

    def _parse_predict_request(payload: Dict) -> tuple:
        """Validate, preprocess, and return (customer_id, X_transformed)."""
        req = PredictRequest(**payload)
        customer_id = req.customer_id
        raw_df = pd.DataFrame([req.features.model_dump()])
        X = preprocessor.transform(raw_df)
        return customer_id, X

    # ═══════════════════════════════════════════════════════════════════════════
    # Routes
    # ═══════════════════════════════════════════════════════════════════════════

    # ── GET /v1/health ────────────────────────────────────────────────────────
    @app.route(f"/{API['version']}/health", methods=["GET"])
    def health():
        if model is None:
            return _err("Model not loaded", 503)
        return _ok({
            "status": "healthy",
            "model_name": metadata.get("model_name", "unknown"),
            "model_class": metadata.get("model_class", "unknown"),
            "feature_count": metadata.get("feature_count", 0),
            "test_roc_auc": metadata.get("test_roc_auc", 0),
            "test_f1": metadata.get("test_f1", 0),
            "trained_at": metadata.get("trained_at", "unknown"),
            "uptime_seconds": round(time.time() - _START_TIME, 2),
            "version": API["version"],
        })

    # ── GET /v1/metrics ───────────────────────────────────────────────────────
    @app.route(f"/{API['version']}/metrics", methods=["GET"])
    def metrics():
        if not metadata:
            return _err("No model metadata available.", 503)
        return _ok({
            "model_name": metadata.get("model_name"),
            "test_roc_auc": metadata.get("test_roc_auc"),
            "test_f1": metadata.get("test_f1"),
            "test_precision": metadata.get("test_precision"),
            "test_recall": metadata.get("test_recall"),
            "trained_at": metadata.get("trained_at"),
        })

    # ── POST /v1/predict ──────────────────────────────────────────────────────
    @app.route(f"/{API['version']}/predict", methods=["POST"])
    def predict():
        guard = _model_required()
        if guard:
            return guard

        try:
            customer_id, X = _parse_predict_request(request.get_json(force=True) or {})
        except ValidationError as exc:
            return _err(str(exc), 422)
        except Exception as exc:
            return _err(f"Request parsing failed: {exc}", 400)

        prob = float(model.predict_proba(X)[0, 1])
        will_churn = prob >= 0.5
        segment = _segment_label(prob)
        confidence = _confidence_label(prob)

        result = {
            "customer_id": customer_id,
            "churn_probability": round(prob, 4),
            "will_churn": will_churn,
            "risk_segment": segment,
            "confidence": confidence,
        }

        _log_prediction(customer_id, prob, will_churn, request.get_json(force=True))
        return _ok(result)

    # ── POST /v1/predict/batch ────────────────────────────────────────────────
    @app.route(f"/{API['version']}/predict/batch", methods=["POST"])
    def predict_batch():
        guard = _model_required()
        if guard:
            return guard

        try:
            batch_req = BatchPredictRequest(**(request.get_json(force=True) or {}))
        except ValidationError as exc:
            return _err(str(exc), 422)

        results = []
        for item in batch_req.customers:
            raw_df = pd.DataFrame([item.features.model_dump()])
            X = preprocessor.transform(raw_df)
            prob = float(model.predict_proba(X)[0, 1])
            will_churn = prob >= 0.5
            results.append({
                "customer_id": item.customer_id,
                "churn_probability": round(prob, 4),
                "will_churn": will_churn,
                "risk_segment": _segment_label(prob),
                "confidence": _confidence_label(prob),
            })
            _log_prediction(item.customer_id, prob, will_churn, item.features.model_dump())

        return _ok({
            "total": len(results),
            "high_risk": sum(1 for r in results if r["risk_segment"] == "High Risk"),
            "medium_risk": sum(1 for r in results if r["risk_segment"] == "Medium Risk"),
            "low_risk": sum(1 for r in results if r["risk_segment"] == "Low Risk"),
            "predictions": results,
        })

    # ── POST /v1/explain ──────────────────────────────────────────────────────
    @app.route(f"/{API['version']}/explain", methods=["POST"])
    def explain():
        guard = _model_required()
        if guard:
            return guard

        if shap_explainer is None:
            return _err("SHAP explainer not available.", 503)

        try:
            customer_id, X = _parse_predict_request(request.get_json(force=True) or {})
        except ValidationError as exc:
            return _err(str(exc), 422)
        except Exception as exc:
            return _err(f"Request parsing failed: {exc}", 400)

        explanation = shap_explainer.explain_instance(X)
        explanation["customer_id"] = customer_id
        return _ok(explanation)

    # ── POST /v1/recommend ────────────────────────────────────────────────────
    @app.route(f"/{API['version']}/recommend", methods=["POST"])
    def recommend():
        guard = _model_required()
        if guard:
            return guard

        try:
            customer_id, X = _parse_predict_request(request.get_json(force=True) or {})
        except ValidationError as exc:
            return _err(str(exc), 422)
        except Exception as exc:
            return _err(f"Request parsing failed: {exc}", 400)

        prob = float(model.predict_proba(X)[0, 1])

        # Get SHAP drivers if explainer available
        shap_drivers = []
        if shap_explainer:
            try:
                exp = shap_explainer.explain_instance(X)
                shap_drivers = exp.get("top_drivers", [])
            except Exception:
                pass

        # Feature dict for rule matching
        features_dict = X.iloc[0].to_dict()
        result = retention_engine.recommend(features_dict, prob, shap_drivers)
        result["customer_id"] = customer_id
        return _ok(result)

    # ── 404 / 405 handlers ────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"status": "error", "error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(e):
        logger.error("Internal server error: %s", e)
        return jsonify({"status": "error", "error": "Internal server error"}), 500

    return app


# ── Helpers ───────────────────────────────────────────────────────────────────


def _segment_label(prob: float) -> str:
    if prob >= 0.70:
        return "High Risk"
    if prob >= 0.30:
        return "Medium Risk"
    return "Low Risk"


def _confidence_label(prob: float) -> str:
    distance = abs(prob - 0.5)
    if distance >= 0.30:
        return "high"
    if distance >= 0.15:
        return "medium"
    return "low"


# ── Audit DB ──────────────────────────────────────────────────────────────────


def _init_db() -> None:
    """Create the predictions audit table if it doesn't exist."""
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUDIT_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                customer_id TEXT,
                churn_probability REAL,
                will_churn INTEGER,
                timestamp TEXT,
                input_json TEXT
            )
        """)
        conn.commit()


def _log_prediction(
    customer_id: Optional[str],
    prob: float,
    will_churn: bool,
    input_payload: Any,
) -> None:
    """Persist a prediction record to the SQLite audit log."""
    try:
        with sqlite3.connect(AUDIT_DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO predictions (id, customer_id, churn_probability, will_churn, timestamp, input_json)
                VALUES (?, ?, ?, ?, datetime('now'), ?)
                """,
                (
                    str(uuid.uuid4()),
                    customer_id,
                    round(prob, 6),
                    int(will_churn),
                    json.dumps(input_payload) if input_payload else "{}",
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Audit log failed: %s", exc)


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=API["host"],
        port=API["port"],
        debug=API["debug"],
    )
