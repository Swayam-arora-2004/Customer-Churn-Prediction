"""
Streamlit Cloud entry point — Customer Churn Prediction Dashboard.

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
    page_title="Churn Prediction & Prevention",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.shared import CUSTOM_CSS, get_metadata

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

metadata = get_metadata()

# ── Landing Page ──────────────────────────────────────────────────────────────

st.title("📊 Customer Churn Prediction & Prevention")
st.caption("Production-grade ML system for predicting and preventing customer churn.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        metadata.get("model_name", "—").replace("_", " ").title() if metadata else "—",
    )
with col2:
    auc = metadata.get("test_roc_auc", 0)
    st.metric("ROC-AUC", f"{auc:.4f}" if auc else "—")
with col3:
    f1 = metadata.get("test_f1", 0)
    st.metric("F1 Score", f"{f1:.4f}" if f1 else "—")

st.divider()

st.markdown(
    """
### 🚀 Navigate using the sidebar

| Page | Description |
|------|-------------|
| **🔍 Prediction** | Predict churn for a single customer with SHAP explanation |
| **📂 Batch Analysis** | Upload a CSV for bulk predictions |
| **🎯 Risk Segments** | View churn risk across all customers |
| **🧠 Model Insights** | ROC curves, feature importance, SHAP summary |
| **🖥 System Monitor** | Model metadata, API health, audit log |
"""
)
