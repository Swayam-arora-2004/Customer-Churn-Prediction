"""Page 5 — System Monitor"""

import pandas as pd
import streamlit as st

from app.shared import CUSTOM_CSS, get_metadata
from src.config import AUDIT_DB_PATH

st.set_page_config(page_title="System Monitor", page_icon="🖥", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

metadata = get_metadata()

st.title("🖥 System Monitor")

# ── Model Metadata ────────────────────────────────────────────────────────────
st.subheader("Model Metadata")
if metadata:
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Algorithm", metadata.get("model_name", "—").replace("_", " ").title())
    mc2.metric("ROC-AUC", f"{metadata.get('test_roc_auc', 0):.4f}")
    mc3.metric("F1 Score", f"{metadata.get('test_f1', 0):.4f}")
    mc4.metric("Features", metadata.get("feature_count", "—"))

    with st.expander("Full Metadata JSON"):
        st.json(metadata)
else:
    st.warning("No model metadata found. Train the model first.")

st.divider()

# ── API Health Check ──────────────────────────────────────────────────────────
st.subheader("API Health Check")
st.caption("Note: The Flask API runs as a separate process on port 5000 (via Docker or `python app/app.py`).")
api_url = st.text_input("API Base URL", value="http://localhost:5000")
if st.button("🔍 Check Health"):
    try:
        import requests

        resp = requests.get(f"{api_url}/v1/health", timeout=5)
        if resp.ok:
            st.success(f"API is healthy | Status: {resp.status_code}")
            st.json(resp.json())
        else:
            st.error(f"API returned {resp.status_code}")
    except Exception as e:
        st.warning(f"API unreachable: {e}")

st.divider()

# ── Audit Log ─────────────────────────────────────────────────────────────────
st.subheader("Recent Predictions (Audit Log)")
try:
    import sqlite3

    if AUDIT_DB_PATH.exists():
        conn = sqlite3.connect(AUDIT_DB_PATH)
        df_audit = pd.read_sql(
            "SELECT id, customer_id, churn_probability, will_churn, timestamp "
            "FROM predictions ORDER BY timestamp DESC LIMIT 50",
            conn,
        )
        conn.close()
        st.dataframe(df_audit, use_container_width=True)
    else:
        st.info("Audit log not yet created. Make predictions via the API first.")
except Exception as e:
    st.warning(f"Could not load audit log: {e}")
