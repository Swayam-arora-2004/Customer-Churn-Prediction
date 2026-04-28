"""Page 2 — Batch Churn Analysis"""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.shared import CUSTOM_CSS, get_model, get_preprocessor

st.set_page_config(page_title="Batch Analysis", page_icon="📂", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

model = get_model()
preprocessor = get_preprocessor()

st.title("📂 Batch Churn Analysis")
st.caption("Upload a CSV of customers to get bulk churn predictions.")

if not model or not preprocessor:
    st.error("Model artefacts not found. Run training pipeline first.")
    st.stop()

uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        st.info(f"Loaded **{len(df_raw):,}** rows × {df_raw.shape[1]} columns")

        if len(df_raw) > 1000:
            st.warning("Only first 1,000 rows will be processed.")
            df_raw = df_raw.head(1000)

        with st.spinner("Running predictions …"):
            customer_ids = df_raw.get(
                "customerID", pd.Series(range(len(df_raw)))
            ).astype(str)
            df_clean = df_raw.drop(columns=["customerID"], errors="ignore")
            X = preprocessor.transform(df_clean)
            probas = model.predict_proba(X)[:, 1]

        df_results = pd.DataFrame(
            {
                "Customer ID": customer_ids,
                "Churn Probability": probas.round(4),
                "Will Churn": (probas >= 0.5),
                "Risk Segment": [
                    "High Risk"
                    if p >= 0.70
                    else "Medium Risk"
                    if p >= 0.30
                    else "Low Risk"
                    for p in probas
                ],
            }
        )

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total Customers", f"{len(df_results):,}")
        r2.metric(
            "High Risk",
            f"{(df_results['Risk Segment'] == 'High Risk').sum():,}",
            delta_color="inverse",
        )
        r3.metric(
            "Medium Risk",
            f"{(df_results['Risk Segment'] == 'Medium Risk').sum():,}",
        )
        r4.metric("Low Risk", f"{(df_results['Risk Segment'] == 'Low Risk').sum():,}")

        fig_hist = px.histogram(
            df_results,
            x="Churn Probability",
            nbins=40,
            color_discrete_sequence=["#3498DB"],
            title="Churn Probability Distribution",
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Prediction Results")
        st.dataframe(
            df_results.style.background_gradient(
                subset=["Churn Probability"], cmap="RdYlGn_r"
            ),
            use_container_width=True,
            height=400,
        )

        csv_out = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_out,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Processing failed: {e}")
else:
    st.markdown(
        """
        **Expected CSV columns:**
        `customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`
        """
    )
