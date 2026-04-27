# 🚀 Customer Churn Prediction & Prevention Engine
**Comprehensive System Architecture & Engineering Report**

> This document is designed to serve as a comprehensive technical deep-dive into the architectural decisions, machine learning methodologies, and software engineering practices implemented in this project. It is structured to help you explain the system at a "FAANG / Staff Engineer" level during technical interviews.

---

## 📑 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [What Was Built (The Goal)](#2-what-was-built-the-goal)
3. [System Architecture (How It Works)](#3-system-architecture-how-it-works)
4. [Data Engineering & Preprocessing (Why We Did It)](#4-data-engineering--preprocessing-why-we-did-it)
5. [Machine Learning Engine (The Intelligence)](#5-machine-learning-engine-the-intelligence)
6. [Explainability (SHAP) & Actionability (Retention Rules)](#6-explainability-shap--actionability-retention-rules)
7. [Backend & API Design (The Microservice)](#7-backend--api-design-the-microservice)
8. [DevOps, CI/CD, & Testing (The Guardrails)](#8-devops-cicd--testing-the-guardrails)
9. [Future Work (What Else We Could Have Done)](#9-future-work-what-else-we-could-have-done)
10. [Interview Cheat Sheet (Common Questions)](#10-interview-cheat-sheet-common-questions)

---

## 1. Executive Summary
Most data science projects end in a Jupyter Notebook. This project was specifically architected to demonstrate what happens next: taking a raw predictive model and industrializing it into a highly scalable, observable, and fully automated microservice. 

We built an **End-to-End Customer Churn Prediction & Prevention System**. It does not simply spit out a "probability of churning." It utilizes XGBoost to compute the probability, SHAP to explain exactly *why* that specific user is at risk on a feature level, and a Python-based Rule Engine to assign targeted retention actions to save the customer. The entire pipeline is wrapped in a Flask container validated by Pydantic, monitored by SQLite and Population Stability Indexes (PSI), and strictly gated by GitHub Actions CI/CD workflows and Pre-Commit hooks.

## 2. What Was Built (The Goal)
**The Problem**: A telecommunications company is losing revenue because customers are canceling their subscriptions (churning). By the time the customer clicks "Cancel," it is too late.
**The Solution**: Identify customers *before* they cancel, understand their pain points, and preemptively offer them incentives.

**Key Deliverables:**
- **Modular Data Pipeline**: Object-Oriented pipeline with strict `fit/transform` methods to eliminate Train-Serve Skew.
- **Ensemble Model**: XGBoost algorithm tracking experiments strictly via MLflow.
- **RESTful API**: A Flask gateway running via `gunicorn` that exposes real-time, batch, explanation, and recommendation endpoints.
- **Interactive UI**: A Streamlit dashboard connecting all APIs natively into a business-friendly interface.
- **Industrial DevOps**: Complete Dockerization, 96 Pytest integration tests, Makefile automation, and strict Flake8/Black linting.

---

## 3. System Architecture (How It Works)

### High-Level Design (HLD)

```mermaid
graph TD
    UI[Streamlit Dashboard] -->|HTTP POST| API(Flask API Gateway)
    UI -.->|Direct inference for UI| PreProc[Data Preprocessor]
    
    subgraph Microservice [Docker Container: API]
        API -->|Validates Input| Pydantic schemas
        Pydantic schemas --> PreProc
        PreProc --> Model[XGBoost Classifier]
        Model --> SHAP[SHAP Explainer]
        Model --> RuleGen[Retention Engine]
        API --> Audit[SQLite Audit DB]
        API --> Metrics[Data Drift / PSI Monitor]
    end
    
    subgraph Model Registry
        MLflow[MLflow Server] --> Model
    end
```

**Why This Architecture?**
In massive companies, API boundaries allow completely separate teams to work efficiently. The ML team can retrain the model and push it to `MLflow` without touching the `Flask API` code. The frontend team can rebuild the `Streamlit` dashboard without knowing how the `XGBoost` model parses arrays.

---

## 4. Data Engineering & Preprocessing (Why We Did It)

### 4.1 The Object-Oriented Approach
In amateur projects, data cleaning is done iteratively in Pandas notebooks (`df = df.fillna()`). In production, this leads to **Train-Serve Skew**—where the data the model is trained on looks statistically different from the data the API receives in production.

We built `src/data_preprocessing.py` as an Object-Oriented `DataPreprocessor` class. 

**What I Did:**
- Created a robust pipeline that executes: `clean()` → `encode()` → `scale()`.
- Saved the pipeline mathematically into a serialized pickle (`preprocessor.pkl`).

**Why I Did It:**
By saving the *state* of the data (e.g., the exact mean and variance used in the `StandardScaler`, or the exact column mappings used during One-Hot Encoding), any incoming API request mapping a single customer passes through the exact same vector space. If a customer is submitted via the API with a categorical feature the model has never seen, the pipeline gracefully ignores or maps it, preventing a 500 Internal Server error.

### 4.2 Feature Engineering Extensibility
We created `src/feature_engineering.py` (generating features like `avg_monthly_charges` or `tenure_groups`). While not all were currently ingested by the model, building an extensible `FeatureEngineer` class demonstrates an understanding that models decay over time and require constant feature iteration.

---

## 5. Machine Learning Engine (The Intelligence)

### 5.1 Model Selection
**What I Did:** Used XGBoost (`XGBClassifier`) optimized for the `ROC-AUC` scoring metric.
**Why I Did It:** Random Forest is robust, and Logistic Regression is highly interpretable. However, *Gradient Boosted Trees* (XGBoost) iteratively learn from the mistakes of previous trees, allowing them to capture highly non-linear relationships in tabular data (which telco data heavily is) outperforming almost all other non-neural network architectures. 

### 5.2 Handling Imbalance via SMOTE
**What I Did:** I implemented `SMOTE` (Synthetic Minority Over-sampling Technique).
**Why I Did It:** Churn datasets are heavily imbalanced (e.g., 80% stay, 20% leave). If an algorithm simply predicts "Stay" every time, it scores 80% accuracy but is completely useless. SMOTE mathematically synthesizes new data points for the minority "Churn" class based on k-nearest neighbors. *Note: We intentionally turned off SMOTE in the final best model because XGBoost's native `scale_pos_weight` and threshold manipulation handled the imbalance gracefully without fabricating data.*

### 5.3 Experiment Tracking
**What I Did:** Integrated natively with **MLflow**.
**Why I Did It:** If a model degrades in production, you must trace exactly what hyperparameters and datasets generated it. MLflow acts as the "Git for Machine Learning," logging all runs, AUC curves, F1 scores, and persisting the binary artifacts.

---

## 6. Explainability (SHAP) & Actionability (Retention Rules)

A high-accuracy black-box model is useless to a customer success team. Providing a prediction of `78% Churn Probability` prompts the question: *"Why? And what should I do about it?"*

### 6.1 Game Theory Explainability (SHAP)
**What I Did:** Integrated the `shap` library (SHapley Additive exPlanations).
**Why I Did It:** SHAP treats every feature as a player in a cooperative game and measures the marginal impact of that feature on the final prediction. By parsing the SHAP array dynamically into the API, we can tell the frontend exactly *which* feature drove the score up (e.g., `Contract_Month-to-month` pushed probability UP by 12%).

### 6.2 The Retention Engine
**What I Did:** Programmed an algorithmic rules engine (`src/prevention.py`).
**Why I Did It:** ML Models predict; software acts. The retention engine ingests the final probability, the customer's raw features, and the SHAP drivers, cross-referencing them against business logic to map automated tasks. For example:
- If `prob > 0.70` AND `SeniorCitizen == 1` → Output Action: *Dedicated Human Phone Call*.
- If `prob > 0.50` AND `MonthlyCharges > 90` → Output Action: *Offer 20% Loyalty Discount*.
This bridges the gap between Data Science and Business Value.

---

## 7. Backend & API Design (The Microservice)

### 7.1 WSGI and API Routing (Flask)
**What I Did:** Built a Flask app containing routes like `POST /predict/batch`, `POST /recommend`, and `GET /health`.
**Why I Did It:** Standardizing the model behind an HTTP layer. To achieve FAANG-scale concurrency, the `Dockerfile` runs the API via `Gunicorn` with 4 dedicated worker threads rather than Flask's single-threaded dev server.

### 7.2 Strict Contracts via Pydantic
**What I Did:** Enforced strict typing on incoming requests using `Pydantic` schemas (`app/schemas.py`).
**Why I Did It:** If the frontend sends `"MonthlyCharges": "One Hundred"`, the Python environment will crash since it expects a float. Pydantic catches this at the API boundary, resulting in a clean `422 Unprocessable Entity` HTTP error telling the client *exactly* which field is malformed. 

### 7.3 Observability (The Audit Trail)
**What I Did:** I designed an asynchronous SQLite Database interceptor (`_log_prediction`).
**Why I Did It:** "Shadow mode logging." Every prediction made is quietly pushed to an SQLite database. Six months from now, if the business asks, *"Why did our retention budget skyrocket?"*, engineers can query the SQLite database and see exactly what inputs generated what outputs.

---

## 8. DevOps, CI/CD, & Testing (The Guardrails)

### 8.1 Testing (Pytest Suite)
**What I Did:** Engineered 96 unit and integration test assertions spanning the entire architecture.
**Why I Did It:** At FAANG, code without tests is rejected instantly. I tested boundary classes, API responses, ML pipeline row-count alignment, and assertion logic. We implemented `pytest-cov` to ensure the codebase strictly hits a mathematically rigid 40%+ test coverage mark (currently operating above 56%).

### 8.2 GitHub Actions (CI Automation)
**What I Did:** I created `.github/workflows/ci.yml`.
**Why I Did It:** Continuous Integration. When an engineer pushes to the `main` branch:
1. `Flake8` and `Black` check the code style.
2. `Pytest` boots a headless docker container and confirms nothing was broken.
3. Codecov forces a fail state if overall code coverage drops.
4. Docker validates it can build a clean image locally.
This ensures the `main` branch is **always** deployable.

### 8.3 Developer Ergonomics (Makefile & Pre-Commit)
**What I Did:** Injected `.pre-commit-config.yaml` and a root `Makefile`.
**Why I Did It:** 
- **Makefile**: Developers just type `make ci-validate` instead of remembering 4 different massive terminal commands. It abstracts complexity.
- **Pre-commit**: We force developers locally to auto-format their code with `Black` *before* Git even allows them to push to the central repo. It blocks garbage code from leaving user machines.

---

## 9. Future Work (What Else We Could Have Done)
To push this architecture from "Staff Engineer" level to an enterprise cloud ecosystem processing 1 Million events per second, I would have historically modified:

1. **Message Queues (Kafka / RabbitMQ)**
   Right now, the API is synchronous. If we receive a batch of 100,000 users at once, HTTP will timeout. I would decouple the inference by placing incoming requests into an Apache Kafka event stream. A pool of worker nodes (Celery) would consume the queue, predict, and write the output asynchronously to the database.
2. **Cloud Database / Feature Store**
   Transitioning the `.csv` and SQLite data structures into a Snowflake / AWS Redshift data warehouse, and utilizing a latency-optimized Feature Store (like Redis or Hopsworks) so the UI doesn't have to pass 20 features across the web; it just passes the `customer_id` and the backend grabs the latest 20 features from Redis.
3. **Automated Continuous Training (CT)**
   Hooking up the `src/monitoring.py` (Population Stability Index). If drift exceeds 0.20 on a cron job, trigger a GitHub action or Airflow DAG to automatically pull new data, run SMOTE-XGBoost, save the `.pkl`, and swap it in production.
4. **Transitioning to FastAPI**
   Replacing Flask with FastAPI allows native asynchronous (`async/await`) execution out of the box, offering slightly better CPU saturation under heavy load and auto-generating OpenAPI/Swagger documentation.

---

## 10. Interview Cheat Sheet (Common Questions)

**Q: Why not use a Deep Learning model?**
> "Deep Learning excels at unstructured data (images, text, audio). For strictly tabular business metrics where data points like 'Tenure' and 'Monthly Charge' dictate outcomes, ensemble tree models like XGBoost almost always outperform Neural Networks, require dramatically less compute to train, and crucially—interface perfectly with Tree-based SHAP explainers for business optics."

**Q: Tell me about a bug you ran into regarding how data is scored.**
> "We encountered 'Train-Serve Data Skew'. The API was failing because the training data dropped row duplicates via `.clean()`, but when batch API predictions came in, dropping duplicates scrambled the index alignment and crashed the service. The solution was parameterizing the `.clean()` function to strictly behave differently between training and inference states while retaining strict alignment on One-Hot Encoding schemas."

**Q: How do you know the model won't degrade in production?**
> "Models always degrade; it’s just a matter of when. We don't rely on 'feel'. I implemented a Population Stability Index (PSI) tracker in our monitoring module. By comparing the statistical probability distributions of incoming API requests against the original training dataset’s distributions, we mathematically track macro-economic data drift. Once the PSI crosses a threshold, we know it's time to trigger an automated pipeline retrain."

**Q: Why put so much effort into the 'Software Engineering' parts of a Data Science project?**
> "A model that sits locally on my laptop generates $0 in business value. By converting the logic into stateless OOP modules, enforcing strict data schemas via Pydantic, and wrapping it in an automated CI/CD pipeline, I ensured the model could be reliably deployed to a cloud cluster and utilized continuously by non-technical customer success teams via the web. Machine Learning is 10% math and 90% software engineering."

---
*Generated by Antigravity.*
