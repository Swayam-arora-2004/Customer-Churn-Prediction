"""Page 4 — Model Insights"""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.shared import CUSTOM_CSS, get_model, get_preprocessor
from src.config import FIGURES_DIR

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

model = get_model()
preprocessor = get_preprocessor()

st.title("🧠 Model Insights")

if not model:
    st.error("Model not loaded.")
    st.stop()

tabs = st.tabs(["📈 Performance Curves", "📊 Feature Importance", "🔬 SHAP Summary"])

with tabs[0]:
    roc_path = FIGURES_DIR / "roc_curve.png"
    pr_path = FIGURES_DIR / "pr_curve.png"
    cm_path = FIGURES_DIR / "confusion_matrix.png"

    plots_available = [p for p in [roc_path, pr_path, cm_path] if p.exists()]
    if plots_available:
        cols = st.columns(len(plots_available))
        for col, path in zip(cols, plots_available):
            col.image(str(path), use_column_width=True)
    else:
        st.info(
            "Performance plots not yet generated. Run the evaluation notebook first."
        )

with tabs[1]:
    fi_path = FIGURES_DIR / "feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), use_column_width=True)
    elif hasattr(model, "feature_importances_") and preprocessor:
        try:
            feature_names = preprocessor.feature_names_
            importances = model.feature_importances_
            fi_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(20)
            )

            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importances (Top 20)",
                color="Importance",
                color_continuous_scale="RdYlGn",
            )
            fig_fi.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
                height=500,
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render feature importance: {e}")
    else:
        st.info("Feature importance not available for this model type.")

with tabs[2]:
    shap_path = FIGURES_DIR / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), use_column_width=True)
    else:
        st.info("SHAP summary plot not generated yet. Run notebook 04 first.")
