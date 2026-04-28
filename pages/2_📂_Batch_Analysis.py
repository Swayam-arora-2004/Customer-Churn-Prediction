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
st.caption("Upload a CSV of customers to get bulk churn predictions — no row limit.")

if not model or not preprocessor:
    st.error("Model artefacts not found. Run training pipeline first.")
    st.stop()

uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        st.info(f"Loaded **{len(df_raw):,}** rows × {df_raw.shape[1]} columns")

        with st.spinner(f"Running predictions on {len(df_raw):,} rows …"):
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
                "Will Churn": probas >= 0.5,
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

        # ── Summary KPIs ──────────────────────────────────────────────────────
        st.divider()
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Total", f"{len(df_results):,}")
        r2.metric(
            "🔴 High Risk",
            f"{(df_results['Risk Segment'] == 'High Risk').sum():,}",
        )
        r3.metric(
            "🟡 Medium Risk",
            f"{(df_results['Risk Segment'] == 'Medium Risk').sum():,}",
        )
        r4.metric(
            "🟢 Low Risk",
            f"{(df_results['Risk Segment'] == 'Low Risk').sum():,}",
        )
        r5.metric("Avg Prob", f"{probas.mean():.1%}")

        # ── Charts ────────────────────────────────────────────────────────────
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_hist = px.histogram(
                df_results,
                x="Churn Probability",
                nbins=40,
                color_discrete_sequence=["#3498DB"],
                title="Churn Probability Distribution",
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_chart2:
            seg_counts = df_results["Risk Segment"].value_counts()
            fig_pie = px.pie(
                values=seg_counts.values,
                names=seg_counts.index,
                title="Risk Segment Breakdown",
                color=seg_counts.index,
                color_discrete_map={
                    "High Risk": "#E74C3C",
                    "Medium Risk": "#F39C12",
                    "Low Risk": "#2ECC71",
                },
                hole=0.4,
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Paginated results table ───────────────────────────────────────────
        st.subheader("Prediction Results")

        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
        with filter_col1:
            segment_filter = st.multiselect(
                "Filter by Segment",
                ["High Risk", "Medium Risk", "Low Risk"],
                default=["High Risk", "Medium Risk", "Low Risk"],
            )
        with filter_col2:
            sort_col = st.selectbox(
                "Sort by",
                ["Churn Probability", "Customer ID"],
                index=0,
            )
        with filter_col3:
            sort_order = st.selectbox("Order", ["Descending", "Ascending"])

        df_filtered = df_results[
            df_results["Risk Segment"].isin(segment_filter)
        ].sort_values(sort_col, ascending=(sort_order == "Ascending"))

        st.caption(f"Showing **{len(df_filtered):,}** of {len(df_results):,} rows")

        # Pagination
        page_size = st.select_slider(
            "Rows per page",
            options=[25, 50, 100, 250, 500],
            value=100,
        )
        total_pages = max(1, (len(df_filtered) + page_size - 1) // page_size)

        page_col1, page_col2, page_col3 = st.columns([1, 3, 1])
        with page_col2:
            page_num = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                label_visibility="collapsed",
            )

        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(df_filtered))
        st.caption(f"Page {page_num} of {total_pages}")

        st.dataframe(
            df_filtered.iloc[start_idx:end_idx].style.background_gradient(
                subset=["Churn Probability"], cmap="RdYlGn_r"
            ),
            use_container_width=True,
            height=450,
        )

        # ── Download ──────────────────────────────────────────────────────────
        csv_out = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download All Predictions CSV",
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

        > All rows will be processed — no row limit.
        """
    )
