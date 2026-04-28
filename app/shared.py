"""
app/shared.py
─────────────────────────────────────────────────────────────────────────────
Shared cached resources for the Streamlit multipage dashboard.
Import this from any page to access model, preprocessor, and metadata.
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, PROCESSED_X_TRAIN_PATH
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.prevention import RetentionEngine


@st.cache_resource(show_spinner="Loading model …")
def get_model():
    try:
        return ModelTrainer.load_best_model(BEST_MODEL_PATH)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading preprocessor …")
def get_preprocessor():
    try:
        return DataPreprocessor.load(PREPROCESSOR_PATH)
    except FileNotFoundError:
        return None


@st.cache_resource
def get_metadata():
    return ModelTrainer.load_metadata()


@st.cache_resource
def get_retention_engine():
    return RetentionEngine()


@st.cache_resource(show_spinner="Initialising SHAP explainer …")
def get_shap_explainer(_model):
    """Only called when SHAP is actually needed."""
    try:
        from src.explainability import SHAPExplainer

        X_bg = pd.read_csv(PROCESSED_X_TRAIN_PATH).sample(100, random_state=42)
        return SHAPExplainer(_model, X_bg)
    except Exception:
        return None


# ── Common CSS ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark glassmorphism cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
    text-align: center;
}
.metric-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.85rem; color: #aaa; margin: 0; font-weight: 500; }

/* Risk badges */
.badge-high   { background: #E74C3C22; color: #E74C3C; border: 1px solid #E74C3C55;
                padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
.badge-medium { background: #F39C1222; color: #F39C12; border: 1px solid #F39C1255;
                padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
.badge-low    { background: #2ECC7122; color: #2ECC71; border: 1px solid #2ECC7155;
                padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }

/* Action card */
.action-card {
    background: rgba(52,152,219,0.07);
    border: 1px solid rgba(52,152,219,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.action-title { font-weight: 600; font-size: 1rem; margin: 0 0 6px; }
.action-desc  { font-size: 0.88rem; color: #bbb; margin: 0; }
.action-meta  { font-size: 0.78rem; color: #888; margin-top: 8px; }
</style>
"""
