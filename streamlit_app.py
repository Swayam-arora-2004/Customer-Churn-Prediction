"""
Streamlit Cloud entry point — Customer Churn Prediction & Prevention Platform.

This is the landing page. Individual pages live in the pages/ directory
and are auto-discovered by Streamlit's multipage framework.
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

st.set_page_config(
    page_title="Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.shared import CUSTOM_CSS, get_metadata

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

metadata = get_metadata()

# ── Landing Page ──────────────────────────────────────────────────────────────

st.title("📊 Customer Churn Prediction & Prevention Platform")
st.caption("Production-grade ML system — predict, explain, prevent, and automate.")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        (
            metadata.get("model_name", "—").replace("_", " ").title()
            if metadata
            else "—"
        ),
    )
with col2:
    auc = metadata.get("test_roc_auc", 0) if metadata else 0
    st.metric("ROC-AUC", f"{auc:.4f}" if auc else "—")
with col3:
    f1 = metadata.get("test_f1", 0) if metadata else 0
    st.metric("F1 Score", f"{f1:.4f}" if f1 else "—")
with col4:
    features = metadata.get("feature_count", "—") if metadata else "—"
    st.metric("Features", features)

st.divider()

st.markdown(
    """
### 🚀 Navigate using the sidebar

| Page | Description |
|------|-------------|
| **🔍 Prediction** | Predict churn for a single customer with SHAP explanation & retention plan |
| **📂 Batch Analysis** | Upload any CSV for bulk predictions — no row limit, with pagination |
| **🎯 Risk Segments** | View churn risk across all customers with segment breakdown |
| **🧠 Model Insights** | ROC/PR curves, confusion matrix, feature importance, SHAP summary |
| **🖥 System Monitor** | Model metadata, API health check, audit log |
| **📡 Data Management** | **Ingest new data → Retrain → Model Registry → Drift Detection** |

---

### 🏗️ System Architecture

```
New Data (CSV Upload / API)
        │
        ▼
┌─────────────────────┐
│  Data Ingestion      │  ← Schema validation, audit trail
│  Service             │  ← Stores in data/pool/
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Retraining Pipeline │  ← Full model suite (LR, RF, XGB, CatBoost)
│  + Auto-Promote      │  ← Compares vs. current production model
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Model Registry      │  ← Version, promote, rollback
│  (v1, v2, v3, …)    │  ← Metrics + dataset stats per version
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Production Model    │  ← SHAP explainability
│  + REST API          │  ← Retention engine (12 action rules)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Drift Monitoring    │  ← PSI-based detection
│  + Alerts            │  ← Auto-retrain trigger
└─────────────────────┘
```
"""
)
