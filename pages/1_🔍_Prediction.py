"""Page 1 — Single Customer Prediction"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.shared import (
    CUSTOM_CSS,
    get_model,
    get_preprocessor,
    get_metadata,
    get_retention_engine,
    get_shap_explainer,
)

st.set_page_config(page_title="Churn Prediction", page_icon="🔍", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

model = get_model()
preprocessor = get_preprocessor()

st.title("🔍 Customer Churn Prediction")
st.caption(
    "Enter customer details to get a real-time churn probability, explanation, and retention plan."
)

if not model or not preprocessor:
    st.error("Model artefacts not found. Please run the training pipeline first.")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
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
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_bak = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        dev_prot = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        stream_mov = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
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

# ── Prediction ────────────────────────────────────────────────────────────────
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

    # ── Results header ────────────────────────────────────────────────────────
    seg = "High Risk" if prob >= 0.70 else "Medium Risk" if prob >= 0.30 else "Low Risk"
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

    # ── SHAP explanation ──────────────────────────────────────────────────────
    shap_drivers = []
    if st.checkbox("Show SHAP explanation", value=False):
        shap_explainer = get_shap_explainer(model)
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
            st.info("SHAP explainer not available.")

    # ── Retention recommendations ─────────────────────────────────────────────
    st.subheader("🛡️ Retention Action Plan")
    features_dict = X.iloc[0].to_dict()
    retention_engine = get_retention_engine()
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
