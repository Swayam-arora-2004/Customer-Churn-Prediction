"""
src/config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Customer Churn Prediction + Prevention System.

All paths, column definitions, hyperparameters, and thresholds live here.
Import from this module everywhere — never hard-code paths or magic numbers.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Directory Paths ───────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist at import time
for _dir in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── File Paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH = RAW_DATA_DIR / "raw_dataset.csv"

PROCESSED_X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
PROCESSED_X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
PROCESSED_Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
PROCESSED_Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

BEST_MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "best_model.pkl")))
PREPROCESSOR_PATH = Path(
    os.getenv("PREPROCESSOR_PATH", str(MODELS_DIR / "preprocessor.pkl"))
)
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

AUDIT_DB_PATH = ROOT_DIR / "logs" / "predictions.db"

# ── Column Definitions ────────────────────────────────────────────────────────
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_COLUMNS = [
    # Yes / No columns
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

BINARY_SERVICE_COLUMNS = [
    # Yes / No / No internet service / No phone service
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",   # stored as 0/1 int but treated categorically
    "InternetService",
    "Contract",
    "PaymentMethod",
]

# All feature columns after preprocessing (order matters for SHAP)
RAW_FEATURE_COLUMNS = (
    NUMERIC_COLUMNS
    + BINARY_COLUMNS[:-1]          # exclude target
    + BINARY_SERVICE_COLUMNS
    + CATEGORICAL_COLUMNS
)

# ── Preprocessing Config ──────────────────────────────────────────────────────
PREPROCESSING = {
    "test_size": 0.20,
    "random_state": 42,
    "stratify": True,
    # Map for binary Yes/No columns
    "binary_map": {"Yes": 1, "No": 0},
    # Simplify No internet/phone service → No (0)
    "service_no_map": {"No internet service": "No", "No phone service": "No"},
    "scaler": "standard",  # "standard" | "minmax" | "robust"
}

# ── Feature Engineering Config ────────────────────────────────────────────────
FEATURE_ENGINEERING = {
    "create_avg_monthly_charges": True,   # TotalCharges / (tenure + 1)
    "create_service_count": True,         # Count of active services
    "create_has_premium_services": True,  # OnlineSecurity OR TechSupport
    "create_tenure_group": True,          # Binned tenure: new/growing/loyal/champion
}

TENURE_BINS = [0, 12, 24, 48, float("inf")]
TENURE_LABELS = ["new", "growing", "loyal", "champion"]

# ── Model Training Config ─────────────────────────────────────────────────────
TRAINING = {
    "random_state": 42,
    "cv_folds": 5,
    "scoring_metric": "roc_auc",
    "use_smote": True,
    "smote_random_state": 42,
    "class_weight": "balanced",   # fallback if SMOTE disabled
}

MODEL_HYPERPARAMS = {
    "logistic_regression": {
        "C": 0.1,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "objective": "binary:logistic",   # Must be explicit for sklearn ≥1.6 tag system
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3,    # ~3:1 non-churn:churn ratio
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    },
    "catboost": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 0,
    },
}

# ── MLflow Config ─────────────────────────────────────────────────────────────
MLFLOW = {
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "mlruns"),
    "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction"),
    "run_tags": {
        "project": "customer-churn-prediction",
        "team": "ml-engineering",
        "dataset": "telco-churn",
    },
}

# ── Evaluation Config ─────────────────────────────────────────────────────────
EVALUATION = {
    "classification_threshold": 0.50,
    "target_metrics": {
        "roc_auc": 0.85,
        "f1": 0.70,
        "precision": 0.65,
        "recall": 0.75,
    },
}

# ── Explainability Config ─────────────────────────────────────────────────────
EXPLAINABILITY = {
    "top_n_features": 10,      # Show top N SHAP features per prediction
    "background_samples": 100, # Background samples for SHAP kernel explainer
}

# ── Prevention / Retention Engine Config ─────────────────────────────────────
PREVENTION = {
    "risk_thresholds": {
        "high": 0.70,    # >= 70% → High Risk
        "medium": 0.30,  # 30–70% → Medium Risk
                         # < 30%  → Low Risk
    },
    "max_recommendations": 3,  # Max retention actions to return per customer
}

# ── API Config ────────────────────────────────────────────────────────────────
API = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": os.getenv("API_DEBUG", "false").lower() == "true",
    "version": os.getenv("API_VERSION", "v1"),
    "max_batch_size": 1000,    # Max rows in a single batch prediction
}

# ── Logging Config ────────────────────────────────────────────────────────────
LOGGING = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": os.getenv("LOG_FORMAT", "json"),  # "json" | "text"
    "log_file": str(LOGS_DIR / "app.log"),
}

# ── Monitoring Config ─────────────────────────────────────────────────────────
MONITORING = {
    "drift_psi_threshold": float(
        os.getenv("DRIFT_PSI_THRESHOLD", 0.2)
    ),
    "churn_score_alert_threshold": float(
        os.getenv("CHURN_SCORE_ALERT_THRESHOLD", 0.70)
    ),
    "reference_data_path": str(PROCESSED_X_TRAIN_PATH),
}
