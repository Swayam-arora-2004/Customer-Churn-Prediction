# Customer Churn Prediction & Prevention System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/API-Flask-lightgrey?logo=flask)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Container-Docker-blue?logo=docker)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-green?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green)

**Production-grade, end-to-end ML system for predicting and preventing customer churn.**

</div>

---

## Overview

This is a FAANG-ready Customer Churn Prediction + Prevention System built on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers × 21 features).

The system goes beyond prediction — it **explains** *why* a customer is at risk and **recommends** targeted retention actions to prevent churn.

---

## Architecture

```
Raw Data → DataPreprocessor → FeatureEngineer → ModelTrainer (4 algorithms + MLflow)
                                                       ↓
                                              Best Model (.pkl)
                                                       ↓
                              ┌────────────────────────┼────────────────────────┐
                              ▼                        ▼                        ▼
                       Flask REST API          SHAPExplainer           RetentionEngine
                       /predict, /explain      SHAP waterfall          12 action rules
                       /recommend, /health     JSON-serializable       SHAP-boosted ranking
                              │                        │                        │
                              └────────────────────────┴────────────────────────┘
                                                       ↓
                                          Streamlit Dashboard (5 pages)
                                                       ↓
                                    DataDriftDetector + ModelPerformanceMonitor
```

---

## Features

### 🤖 Machine Learning Pipeline
- **4 algorithms**: Logistic Regression (baseline), Random Forest, XGBoost, CatBoost
- **Stratified K-Fold CV** (5 folds) with SMOTE for class imbalance
- **MLflow experiment tracking** — all runs logged automatically
- **Target metrics**: ROC-AUC ≥ 0.85, F1 ≥ 0.70

### 🧠 Explainability (SHAP)
- `TreeExplainer` for tree models, `LinearExplainer` for LR
- Per-customer waterfall explanation (JSON-serializable)
- Global SHAP summary beeswarm plot

### 🛡️ Retention Engine
- **12 domain-driven retention actions** across 5 categories (pricing, contract, service, support, engagement)
- **Risk segmentation**: High (≥70%), Medium (30–70%), Low (<30%)
- **SHAP-driver boosting** — actions aligned with top drivers get a relevance boost
- **Estimated retention lift** calculation

### 🌐 REST API (Flask)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Model metadata & API uptime |
| `/v1/metrics` | GET | Model performance metrics |
| `/v1/predict` | POST | Single customer prediction |
| `/v1/predict/batch` | POST | Batch prediction (up to 1,000 rows) |
| `/v1/explain` | POST | SHAP explanation for a customer |
| `/v1/recommend` | POST | Retention action recommendations |

### 📊 Streamlit Dashboard (5 Pages)
1. **Single Prediction** — Form-based prediction + SHAP waterfall + retention plan
2. **Batch Analysis** — CSV upload → bulk predictions + download
3. **Risk Segments** — Portfolio view (pie, box plots, high-risk table)
4. **Model Insights** — ROC/PR curves, confusion matrix, feature importance, SHAP summary
5. **System Monitor** — API health check, metadata, audit log

### 🐳 Production Infrastructure
- **Docker Compose** — API + Dashboard + MLflow server
- **GitHub Actions CI** — lint → test → Docker build
- **SQLite audit log** — every prediction persisted
- **PSI-based drift detection** — alerts on data distribution shift

---

## Project Structure

```
Customer-Churn-Prediction/
├── src/
│   ├── config.py                # All paths, column defs, hyperparameters
│   ├── data_preprocessing.py    # DataPreprocessor (sklearn-compatible)
│   ├── feature_engineering.py   # FeatureEngineer (derived features)
│   ├── model_training.py        # ModelTrainer + MLflow
│   ├── evaluation.py            # ModelEvaluator + plots
│   ├── explainability.py        # SHAPExplainer
│   ├── prevention.py            # RetentionEngine
│   └── monitoring.py            # DataDriftDetector + ModelPerformanceMonitor
├── app/
│   ├── app.py                   # Flask REST API
│   ├── dashboard.py             # Streamlit Dashboard
│   ├── schemas.py               # Pydantic v2 models
│   └── logger.py                # Structured JSON/text logging
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   ├── test_prevention.py
│   └── test_api.py
├── notebooks/
│   ├── 01_initial_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_and_insights.ipynb
├── data/raw/raw_dataset.csv
├── models/                      # Trained model artefacts
├── figures/                     # Evaluation plots
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

### 1. Setup

```bash
# Clone repo
git clone https://github.com/Swayam-arora-2004/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Create virtual environment
python -m venv churn-prediction
source churn-prediction/bin/activate   # On Windows: churn-prediction\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Copy env template
cp .env.example .env
```

### 2. Run the Pipeline

```bash
# Step 1: Preprocess data
python -m src.data_preprocessing

# Step 2: Train models (logs to MLflow)
python -m src.model_training

# Step 3: Start MLflow UI (optional)
mlflow ui --port 5001
```

### 3. Start the API

```bash
python app/app.py
# or with gunicorn:
gunicorn -w 4 -b 0.0.0.0:5000 "app.app:create_app()"
```

### 4. Start the Dashboard

```bash
streamlit run app/dashboard.py
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

## API Usage Example

```bash
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-001",
    "features": {
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
      "TotalCharges": 780.0
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

## Model Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| ROC-AUC | ≥ 0.85 | Overall discriminative power |
| F1 Score | ≥ 0.70 | Balance of precision and recall for churn class |
| Precision | ≥ 0.65 | Correct positive predictions |
| Recall | ≥ 0.75 | Correctly identified churners |

---

## Dataset

**Telco Customer Churn** — IBM Sample Data
- 7,043 customers × 21 features
- ~26% churn rate (class imbalance handled with SMOTE)
- Features: demographics, account info, phone/internet services, billing

---

## License

MIT © 2026 Swayam Arora
