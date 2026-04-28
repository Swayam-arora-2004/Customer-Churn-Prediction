"""
app/dashboard.py
─────────────────────────────────────────────────────────────────────────────
5-Page Streamlit Dashboard — Customer Churn Prediction + Prevention System

Pages:
  1. Single Prediction    — Form-based prediction + SHAP waterfall + Actions
  2. Batch Analysis       — CSV upload → bulk predictions + download
  3. Risk Segments        — Portfolio view across all predictions
  4. Model Insights       — Global SHAP, ROC/PR curves, feature importance
  5. System Monitor       — API health check, model metadata, audit trail

Run:
  streamlit run app/dashboard.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup (allow running from project root) ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    PREPROCESSOR_PATH,
    PROCESSED_X_TRAIN_PATH,
)
from src.data_preprocessing import DataPreprocessor
from src.explainability import SHAPExplainer
from src.model_training import ModelTrainer
from src.prevention import RetentionEngine

try:
    from app.app import _log_prediction, _init_db

    _HAS_FLASK_APP = True
except Exception:
    _HAS_FLASK_APP = False

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Prediction & Prevention",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
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

/* Probability gauge */
.prob-bar {
    height: 10px; border-radius: 5px;
    background: linear-gradient(to right, #2ECC71, #F39C12, #E74C3C);
    margin: 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Cached resource loaders ───────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model artefacts …")
def load_artefacts():
    try:
        model = ModelTrainer.load_best_model(BEST_MODEL_PATH)
        preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
        metadata = ModelTrainer.load_metadata()
        return model, preprocessor, metadata, None
    except FileNotFoundError as e:
        return None, None, {}, str(e)


@st.cache_resource(show_spinner="Initialising SHAP explainer …")
def load_explainer(_model):
    """Lazy-loaded: only called when SHAP is actually needed."""
    try:
        X_train = pd.read_csv(PROCESSED_X_TRAIN_PATH).sample(100, random_state=42)
        return SHAPExplainer(_model, X_train)
    except Exception:
        return None


@st.cache_data(show_spinner="Running predictions on full dataset …", ttl=3600)
def compute_risk_segments(_preprocessor, _model, data_path: str):
    """Cache full-dataset predictions so Risk Segments page is instant."""
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


# ── Load artefacts (model + preprocessor only — fast) ─────────────────────────

model, preprocessor, metadata, load_error = load_artefacts()
retention_engine = RetentionEngine()

# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Churn Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🔍  Single Prediction",
            "📂  Batch Analysis",
            "🎯  Risk Segments",
            "🧠  Model Insights",
            "🖥  System Monitor",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Model status badge
    if model:
        st.success(
            f"✅ Model: **{metadata.get('model_name','unknown').replace('_',' ').title()}**"
        )
        st.caption(f"ROC-AUC: **{metadata.get('test_roc_auc', 0):.4f}**")
    else:
        st.error("⚠️ Model not loaded")
        if load_error:
            st.caption(load_error)


# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════

if "Single" in page:
    st.title("🔍 Customer Churn Prediction")
    st.caption(
        "Enter customer details to get a real-time churn probability, explanation, and retention plan."
    )

    if not model:
        st.error("Model artefacts not found. Please run the training pipeline first.")
        st.stop()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.subheader("Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics & Account**")
            customer_id = st.text_input("Customer ID (optional)", value="CUST-001")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox(
                "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_bak = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            dev_prot = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_sup = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            stream_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            stream_mov = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

        with col3:
            st.markdown("**Billing**")
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.01)
            total = st.number_input(
                "Total Charges ($)", 0.0, 10000.0, float(monthly * max(1, tenure)), 0.01
            )

        submitted = st.form_submit_button(
            "🚀 Predict Churn", type="primary", use_container_width=True
        )

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        raw_input = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak,
            "DeviceProtection": dev_prot,
            "TechSupport": tech_sup,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        raw_df = pd.DataFrame([raw_input])
        X = preprocessor.transform(raw_df)
        prob = float(model.predict_proba(X)[0, 1])
        will_churn = prob >= 0.5

        # Ensure SQLite target exists and append audit trail
        if _HAS_FLASK_APP:
            try:
                _init_db()
                _log_prediction(customer_id, prob, will_churn, raw_input)
            except Exception as e:
                st.warning(f"Could not write to audit log: {e}")

        # ── Results header ────────────────────────────────────────────────────
        seg = (
            "High Risk"
            if prob >= 0.70
            else "Medium Risk"
            if prob >= 0.30
            else "Low Risk"
        )
        badge_cls = {
            "High Risk": "badge-high",
            "Medium Risk": "badge-medium",
            "Low Risk": "badge-low",
        }[seg]
        badge_colour = {
            "High Risk": "#E74C3C",
            "Medium Risk": "#F39C12",
            "Low Risk": "#2ECC71",
        }[seg]

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Risk Segment", seg)
        m3.metric("Will Churn?", "Yes ⚠️" if prob >= 0.5 else "No ✅")
        m4.metric("Customer", customer_id or "—")

        # Probability gauge
        st.markdown(
            f"""
            <div style="margin: 12px 0">
              <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#888">
                <span>0%</span><span>50%</span><span>100%</span>
              </div>
              <div style="position:relative; height:14px; border-radius:7px;
                          background:linear-gradient(to right,#2ECC71,#F39C12,#E74C3C);">
                <div style="position:absolute; top:50%; left:{prob*100:.1f}%;
                            transform:translate(-50%,-50%); width:20px; height:20px;
                            background:white; border:3px solid {badge_colour};
                            border-radius:50%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── SHAP explanation ──────────────────────────────────────────────────
        shap_drivers = []
        if st.checkbox("Show SHAP explanation", value=True):
            shap_explainer = load_explainer(model) if model else None
            if shap_explainer:
                st.subheader("🧠 Why this prediction?")
                exp = shap_explainer.explain_instance(X)
                shap_drivers = exp.get("top_drivers", [])

                shap_df = pd.DataFrame(shap_drivers)
                shap_df["color"] = shap_df["shap_value"].apply(
                    lambda v: "#E74C3C" if v > 0 else "#2ECC71"
                )
                shap_df = shap_df.sort_values("shap_value")

                fig = go.Figure(
                    go.Bar(
                        x=shap_df["shap_value"],
                        y=shap_df["feature"],
                        orientation="h",
                        marker_color=shap_df["color"].tolist(),
                        text=[f"{v:+.4f}" for v in shap_df["shap_value"]],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title="SHAP Feature Contributions",
                    xaxis_title="SHAP Value (impact on churn probability)",
                    height=400,
                    margin=dict(l=0, r=60, t=40, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SHAP explainer not available — train the model first.")

        # ── Retention recommendations ─────────────────────────────────────────
        st.subheader("🛡️ Retention Action Plan")
        features_dict = X.iloc[0].to_dict()
        recs = retention_engine.recommend(features_dict, prob, shap_drivers)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            for rec in recs["recommendations"]:
                priority_colour = {
                    "critical": "#E74C3C",
                    "high": "#F39C12",
                    "medium": "#3498DB",
                    "low": "#95A5A6",
                }.get(rec["priority"], "#aaa")
                st.markdown(
                    f"""
                    <div class="action-card">
                      <div class="action-title">{rec['title']}</div>
                      <div class="action-desc">{rec['description']}</div>
                      <div class="action-meta">
                        📂 {rec['category'].title()} &nbsp;|&nbsp;
                        <span style="color:{priority_colour}">● {rec['priority'].upper()}</span>
                        &nbsp;|&nbsp; Impact: {rec['impact_score']:.0%}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with col_b:
            lift = recs["estimated_retention_lift"]
            st.metric("Estimated Retention Lift", f"{lift:.1%}")
            st.metric("Risk Segment", recs["customer_segment"])
            st.metric("Actions", len(recs["recommendations"]))


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif "Batch" in page:
    st.title("📂 Batch Churn Analysis")
    st.caption("Upload a CSV of customers to get bulk churn predictions.")

    if not model:
        st.error("Model artefacts not found. Run training pipeline first.")
        st.stop()

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.info(f"Loaded **{len(df_raw):,}** rows × {df_raw.shape[1]} columns")

            # Subset to API limit
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

            # ── Summary metrics ───────────────────────────────────────────────
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
            r4.metric(
                "Low Risk", f"{(df_results['Risk Segment'] == 'Low Risk').sum():,}"
            )

            # ── Distribution chart ────────────────────────────────────────────
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

            # ── Results table ─────────────────────────────────────────────────
            st.subheader("Prediction Results")
            st.dataframe(
                df_results.style.background_gradient(
                    subset=["Churn Probability"], cmap="RdYlGn_r"
                ),
                use_container_width=True,
                height=400,
            )

            # ── Download ──────────────────────────────────────────────────────
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
        # Show sample from raw dataset
        try:
            from src.config import RAW_DATA_PATH

            sample = pd.read_csv(RAW_DATA_PATH).head(5)
            st.caption("Sample data (first 5 rows of raw dataset):")
            st.dataframe(sample, use_container_width=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Risk Segments
# ══════════════════════════════════════════════════════════════════════════════

elif "Segments" in page:
    st.title("🎯 Customer Risk Segments")

    if not model:
        st.error("Model not loaded.")
        st.stop()

    try:
        from src.config import RAW_DATA_PATH

        df_seg = compute_risk_segments(preprocessor, model, str(RAW_DATA_PATH))

        # ── KPIs ─────────────────────────────────────────────────────────────
        seg_counts = df_seg["Risk Segment"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(df_seg):,}")
        c2.metric("🔴 High Risk", f"{seg_counts.get('High Risk', 0):,}")
        c3.metric("🟡 Medium Risk", f"{seg_counts.get('Medium Risk', 0):,}")
        c4.metric("🟢 Low Risk", f"{seg_counts.get('Low Risk', 0):,}")

        col_left, col_right = st.columns(2)

        with col_left:
            # Pie chart
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
            # Box plot
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

        # High-risk table
        st.subheader("🔴 Top High-Risk Customers")
        high_risk_df = (
            df_seg[df_seg["Risk Segment"] == "High Risk"]
            .sort_values("Churn Probability", ascending=False)
            .head(20)
        )
        st.dataframe(high_risk_df, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load dataset: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Page 4 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════

elif "Insights" in page:
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
        elif hasattr(model, "feature_importances_"):
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


# ══════════════════════════════════════════════════════════════════════════════
# Page 5 — System Monitor
# ══════════════════════════════════════════════════════════════════════════════

elif "Monitor" in page:
    st.title("🖥 System Monitor")

    # ── Model Metadata ────────────────────────────────────────────────────────
    st.subheader("Model Metadata")
    if metadata:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "Algorithm", metadata.get("model_name", "—").replace("_", " ").title()
        )
        mc2.metric("ROC-AUC", f"{metadata.get('test_roc_auc', 0):.4f}")
        mc3.metric("F1 Score", f"{metadata.get('test_f1', 0):.4f}")
        mc4.metric("Features", metadata.get("feature_count", "—"))

        with st.expander("Full Metadata JSON"):
            st.json(metadata)
    else:
        st.warning("No model metadata found. Train the model first.")

    st.divider()

    # ── API Health Check ──────────────────────────────────────────────────────
    st.subheader("API Health Check")
    api_url = st.text_input("API Base URL", value="http://localhost:8000")
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

    # ── Audit Log ─────────────────────────────────────────────────────────────
    st.subheader("Recent Predictions (Audit Log)")
    from src.config import AUDIT_DB_PATH

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
