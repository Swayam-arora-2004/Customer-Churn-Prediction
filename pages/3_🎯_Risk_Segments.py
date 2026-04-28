"""Page 3 — Customer Risk Segments"""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.shared import CUSTOM_CSS, get_model, get_preprocessor

st.set_page_config(page_title="Risk Segments", page_icon="🎯", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

model = get_model()
preprocessor = get_preprocessor()

st.title("🎯 Customer Risk Segments")

if not model or not preprocessor:
    st.error("Model not loaded.")
    st.stop()


@st.cache_data(show_spinner="Running predictions on full dataset …", ttl=3600)
def compute_segments(_preprocessor, _model, data_path: str):
    df_raw = pd.read_csv(data_path).drop(columns=["Churn"], errors="ignore")
    customer_ids = (
        df_raw["customerID"].copy()
        if "customerID" in df_raw.columns
        else pd.Series(range(len(df_raw)))
    )
    df_input = df_raw.drop(columns=["customerID"], errors="ignore")
    X_all = _preprocessor.transform(df_input)
    probas = _model.predict_proba(X_all)[:, 1]
    return pd.DataFrame(
        {
            "Customer ID": customer_ids.values,
            "Churn Probability": probas.round(4),
            "Risk Segment": [
                "High Risk" if p >= 0.70 else "Medium Risk" if p >= 0.30 else "Low Risk"
                for p in probas
            ],
        }
    )


try:
    from src.config import RAW_DATA_PATH

    df_seg = compute_segments(preprocessor, model, str(RAW_DATA_PATH))

    seg_counts = df_seg["Risk Segment"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df_seg):,}")
    c2.metric("🔴 High Risk", f"{seg_counts.get('High Risk', 0):,}")
    c3.metric("🟡 Medium Risk", f"{seg_counts.get('Medium Risk', 0):,}")
    c4.metric("🟢 Low Risk", f"{seg_counts.get('Low Risk', 0):,}")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_pie = px.pie(
            values=seg_counts.values,
            names=seg_counts.index,
            title="Segment Distribution",
            color=seg_counts.index,
            color_discrete_map={
                "High Risk": "#E74C3C",
                "Medium Risk": "#F39C12",
                "Low Risk": "#2ECC71",
            },
            hole=0.45,
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        fig_box = px.box(
            df_seg,
            x="Risk Segment",
            y="Churn Probability",
            color="Risk Segment",
            color_discrete_map={
                "High Risk": "#E74C3C",
                "Medium Risk": "#F39C12",
                "Low Risk": "#2ECC71",
            },
            title="Probability Distribution by Segment",
        )
        fig_box.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("🔴 Top High-Risk Customers")
    high_risk_df = (
        df_seg[df_seg["Risk Segment"] == "High Risk"]
        .sort_values("Churn Probability", ascending=False)
        .head(20)
    )
    st.dataframe(high_risk_df, use_container_width=True)

except Exception as e:
    st.error(f"Could not load dataset: {e}")
