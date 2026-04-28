<div align="center">

# 📊 Customer Churn Prediction & Prevention Platform

[![CI](https://github.com/Swayam-arora-2004/Customer-Churn-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Swayam-arora-2004/Customer-Churn-Prediction/actions)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-006400?logo=xgboost)
![Flask](https://img.shields.io/badge/API-Flask-000000?logo=flask)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

**Production-grade, automated ML platform for predicting, explaining, and preventing customer churn.**

[Live Demo](https://customer-churn-prediction-sa.streamlit.app/) · [API Docs](#-rest-api) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start)

</div>

---

## 🧭 What Makes This Production-Grade

This is **not** a static notebook trained on one CSV. It's an automated ML platform with:

| Capability | Implementation |
|---|---|
| **Dynamic Data Ingestion** | Upload new customer data anytime → validated, pooled, audit-trailed |
| **Automated Retraining** | One-click retrain on accumulated data; auto-promotes if better |
| **Model Versioning** | Full registry with version history, promote, rollback |
| **Data Drift Detection** | PSI-based monitoring — alerts when incoming data diverges from training |
| **Explainable AI** | Per-customer SHAP explanations (why *this* customer will churn) |
| **Actionable Prevention** | 12 domain-driven retention actions ranked by SHAP relevance |
| **REST API** | 7 production endpoints with Pydantic validation and audit logging |
| **CI/CD Pipeline** | GitHub Actions: lint → test (96 tests) → Docker build |
| **Containerized Deployment** | Docker Compose (API + Dashboard + MLflow) |

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                         │
│                                                                              │
│  📥 Data Ingestion Service                                                   │
│  ├── Schema validation (19 required columns)                                │
│  ├── Batch storage (data/pool/)                                             │
│  ├── Audit trail (logs/ingestion_log.json)                                  │
│  └── Merge-on-demand for retraining                                         │
│                                                                              │
│  📡 Data Drift Detector (PSI)                                               │
│  ├── Per-feature PSI computation                                            │
│  ├── stable / moderate_drift / significant_drift                            │
│  └── Auto-alert on threshold breach                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                          ML PIPELINE                                        │
│                                                                              │
│  🔧 DataPreprocessor → FeatureEngineer → ModelTrainer                       │
│  ├── 4 algorithms: LogisticRegression, RandomForest, XGBoost, CatBoost     │
│  ├── Stratified 5-fold CV + SMOTE for class imbalance                      │
│  ├── MLflow experiment tracking                                             │
│  └── Auto-select best model by ROC-AUC                                     │
│                                                                              │
│  📋 Model Registry                                                          │
│  ├── Version control (v1, v2, v3, …)                                       │
│  ├── Metrics + dataset stats per version                                    │
│  ├── promote() → copies to production path                                  │
│  └── rollback() → reverts to previous version                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                          SERVING LAYER                                      │
│                                                                              │
│  🌐 Flask REST API (7 endpoints)                                            │
│  ├── /v1/predict, /v1/predict/batch                                         │
│  ├── /v1/explain, /v1/recommend                                             │
│  ├── /v1/health, /v1/metrics                                                │
│  └── SQLite audit log                                                       │
│                                                                              │
│  📊 Streamlit Dashboard (6 pages)                                           │
│  ├── Single Prediction + SHAP + Retention Plan                              │
│  ├── Batch Analysis (unlimited rows, paginated)                             │
│  ├── Risk Segments (portfolio view)                                         │
│  ├── Model Insights (ROC, PR, SHAP summary)                                │
│  ├── System Monitor (health, audit log)                                     │
│  └── Data Management (ingest, retrain, registry, drift)                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🤖 Machine Learning Pipeline
- **4 algorithms** compared: Logistic Regression, Random Forest, XGBoost, CatBoost
- **Stratified K-Fold CV** (5 folds) with SMOTE for class imbalance
- **MLflow experiment tracking** — all runs, metrics, and artefacts logged
- **Target metrics**: ROC-AUC ≥ 0.85, F1 ≥ 0.70

### 📥 Dynamic Data Ingestion
- Upload new customer CSVs via dashboard or API
- Schema validation against 19 required columns
- Each batch stored with timestamp in `data/pool/` for traceability
- Full audit trail in `logs/ingestion_log.json`

### 🔄 Automated Retraining + Model Registry
- **One-click retraining** on all accumulated data
- **Auto-promote**: new model replaces production only if it beats the current one
- **Version history**: every trained model tracked with metrics + dataset stats
- **Rollback**: revert to any previous model version instantly

### 🧠 Explainability (SHAP)
- `TreeExplainer` for tree models, `LinearExplainer` for LR
- Per-customer JSON-serializable explanation
- Top feature drivers with direction (increases/decreases churn)

### 🛡️ Retention Engine
- **12 domain-driven retention actions** across 5 categories
- **SHAP-driver boosting** — actions aligned with top churn drivers get priority
- **Risk segmentation**: High (≥70%), Medium (30–70%), Low (<30%)
- **Estimated retention lift** calculation

### 📡 Data Drift Monitoring
- **Population Stability Index (PSI)** per feature
- Three-tier alerting: Stable / Moderate Drift / Significant Drift
- Visual drift dashboard with per-feature PSI chart

### 🌐 REST API (Flask)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Model metadata & API uptime |
| `/v1/metrics` | GET | Model performance metrics |
| `/v1/predict` | POST | Single customer prediction |
| `/v1/predict/batch` | POST | Batch prediction |
| `/v1/explain` | POST | SHAP explanation for a customer |
| `/v1/recommend` | POST | Retention action recommendations |

### 📊 Streamlit Dashboard (6 Pages)

| Page | Description |
|------|-------------|
| **🔍 Prediction** | Form-based prediction + SHAP explanation + retention plan |
| **📂 Batch Analysis** | Upload CSV → bulk predictions (no row limit) with pagination |
| **🎯 Risk Segments** | Portfolio view with segment breakdown and high-risk table |
| **🧠 Model Insights** | ROC/PR curves, confusion matrix, feature importance |
| **🖥 System Monitor** | API health, model metadata, audit log |
| **📡 Data Management** | Ingest → Retrain → Registry → Drift Detection |

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
├── src/                             # Core ML modules
│   ├── config.py                    # Central configuration (paths, hyperparams)
│   ├── data_ingestion.py            # DataIngestionService (dynamic data intake)
│   ├── data_preprocessing.py        # DataPreprocessor (sklearn-compatible)
│   ├── feature_engineering.py       # FeatureEngineer (derived features)
│   ├── model_training.py            # ModelTrainer + MLflow integration
│   ├── model_registry.py            # ModelRegistry (version, promote, rollback)
│   ├── evaluation.py                # ModelEvaluator + plot generation
│   ├── explainability.py            # SHAPExplainer (tree + linear + kernel)
│   ├── prevention.py                # RetentionEngine (12 action rules)
│   └── monitoring.py                # DataDriftDetector + ModelPerformanceMonitor
├── app/                             # Serving layer
│   ├── app.py                       # Flask REST API (7 endpoints)
│   ├── shared.py                    # Shared Streamlit cached resources
│   ├── schemas.py                   # Pydantic v2 request/response models
│   └── logger.py                    # Structured JSON/text logging
├── pages/                           # Streamlit multipage dashboard
│   ├── 1_🔍_Prediction.py
│   ├── 2_📂_Batch_Analysis.py
│   ├── 3_🎯_Risk_Segments.py
│   ├── 4_🧠_Model_Insights.py
│   ├── 5_🖥_System_Monitor.py
│   └── 6_📡_Data_Management.py
├── tests/                           # Test suite (96 tests, 56% coverage)
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   ├── test_prevention.py
│   └── test_api.py
├── notebooks/                       # Exploratory analysis
├── data/
│   ├── raw/                         # Original Telco dataset
│   └── pool/                        # Ingested data batches
├── models/
│   ├── best_model.pkl               # Active production model
│   ├── preprocessor.pkl             # Fitted preprocessor
│   └── registry/                    # Model version history
├── .github/workflows/ci.yml         # CI pipeline (lint → test → docker)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development + CI dependencies
├── streamlit_app.py                 # Streamlit Cloud entry point
└── pyproject.toml
```

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/Swayam-arora-2004/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env
```

### 2. Train the Pipeline

```bash
python -m src.data_preprocessing    # Preprocess + feature engineering
python -m src.model_training        # Train all models (logs to MLflow)
```

### 3. Start the Dashboard

```bash
streamlit run streamlit_app.py
```

### 4. Start the API

```bash
python app/app.py
# Production:
gunicorn -w 4 -b 0.0.0.0:5000 "app.app:create_app()"
```

### 5. Run Tests

```bash
pytest tests/ -v --cov=src --cov=app
```

### 6. Docker (All Services)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:5000 |
| Dashboard | http://localhost:8501 |
| MLflow | http://localhost:5001 |

---

## 🔌 API Usage

```bash
# Single prediction
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-001",
    "features": {
      "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
      "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
      "MultipleLines": "No", "InternetService": "DSL",
      "OnlineSecurity": "No", "OnlineBackup": "Yes",
      "DeviceProtection": "No", "TechSupport": "No",
      "StreamingTV": "No", "StreamingMovies": "No",
      "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 65.0, "TotalCharges": 780.0
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "customer_id": "CUST-001",
    "churn_probability": 0.7234,
    "will_churn": true,
    "risk_segment": "High Risk",
    "confidence": "high"
  }
}
```

---

## 📈 Model Performance

| Metric | Target | Description |
|--------|--------|-------------|
| ROC-AUC | ≥ 0.85 | Discriminative power |
| F1 Score | ≥ 0.70 | Balance of precision and recall |
| Precision | ≥ 0.65 | Correct positive predictions |
| Recall | ≥ 0.75 | Correctly identified churners |

---

## 📊 Dataset

**Telco Customer Churn** — IBM Sample Data
- 7,043 customers × 21 features
- ~26% churn rate (handled with SMOTE)
- Features: demographics, account info, services, billing
- Extensible via Data Ingestion Service (new batches can be added anytime)

---

## 🧪 Testing

96 tests across 4 test modules:
- `test_data_preprocessing.py` — Cleaning, encoding, scaling, feature engineering
- `test_model_training.py` — Training, predictions, SHAP, model persistence
- `test_prevention.py` — Segmentation, recommendations, retention lift
- `test_api.py` — All API endpoints, error handling, validation

```bash
pytest tests/ -v --cov=src --cov=app --cov-report=term-missing
# Coverage: 56%+ (threshold: 40%)
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | scikit-learn, XGBoost, CatBoost |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | Flask + Pydantic v2 |
| Dashboard | Streamlit (multipage) |
| Testing | pytest + pytest-cov |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Data Validation | Custom schema validation |
| Drift Detection | Population Stability Index (PSI) |

---

## 📄 License

MIT © 2026 Swayam Arora
