"""Page 6 — Data Management (Ingest, Retrain, Registry, Drift)"""

import pandas as pd
import streamlit as st

from app.shared import CUSTOM_CSS, get_model, get_preprocessor

st.set_page_config(page_title="Data Management", page_icon="📡", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📡 Data Management")
st.caption(
    "Ingest new datasets, retrain models, manage the model registry, "
    "and monitor data drift — all from one place."
)

tabs = st.tabs(["📥 Ingest Data", "🔄 Retrain Model", "📋 Model Registry", "📡 Data Drift"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Ingest Data
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.subheader("📥 Ingest New Dataset")
    st.markdown(
        "Upload a new batch of customer data. It will be validated "
        "and added to the **data pool** for future retraining."
    )

    source_label = st.text_input(
        "Source Label (e.g. Q1-2026-export)", value="manual-upload"
    )
    uploaded = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        key="ingest_upload",
    )

    if uploaded and st.button("📥 Ingest Dataset", type="primary"):
        from src.data_ingestion import DataIngestionService

        service = DataIngestionService()
        df = pd.read_csv(uploaded)
        result = service.ingest(df, source_label=source_label)

        if result["status"] == "success":
            st.success(
                f"✅ Ingested **{result['rows']:,}** rows "
                f"({result['columns']} columns) as `{result['filename']}`"
            )
        else:
            st.error(f"❌ {result['message']}")

    st.divider()
    st.subheader("📊 Data Pool Status")

    from src.data_ingestion import DataIngestionService

    service = DataIngestionService()
    stats = service.get_pool_stats()

    s1, s2, s3 = st.columns(3)
    s1.metric("Raw Dataset Rows", f"{stats['raw_dataset_rows']:,}")
    s2.metric("Pooled Batches", f"{stats['pool_batches']:,}")
    s3.metric("Total Available Rows", f"{stats['total_available_rows']:,}")

    if stats["batch_files"]:
        with st.expander("Pooled batch files"):
            for f in stats["batch_files"]:
                st.text(f"  📄 {f}")

    log = service.get_ingestion_log()
    if log:
        st.subheader("📜 Ingestion History")
        st.dataframe(
            pd.DataFrame(log)[
                ["timestamp", "source", "rows", "columns", "has_target", "filename"]
            ],
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Retrain Model
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.subheader("🔄 Retrain Model on Pooled Data")
    st.markdown(
        "Retrains the full model suite on **all available data** "
        "(original dataset + ingested batches). "
        "The best model is automatically compared against the current "
        "production model and promoted if it's better."
    )

    from src.data_ingestion import DataIngestionService as DIS2
    from src.config import RETRAINING

    dis = DIS2()
    pool_stats = dis.get_pool_stats()
    total = pool_stats["total_available_rows"]

    st.info(
        f"📊 **{total:,}** total rows available for training "
        f"(minimum required: {RETRAINING['min_samples_for_retrain']:,})"
    )

    if total < RETRAINING["min_samples_for_retrain"]:
        st.warning(
            f"Not enough data to retrain. Need at least "
            f"**{RETRAINING['min_samples_for_retrain']:,}** rows."
        )

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        retrain_clicked = st.button(
            "🚀 Start Retraining",
            type="primary",
            disabled=total < RETRAINING["min_samples_for_retrain"],
        )

    if retrain_clicked:
        with st.spinner("Retraining in progress — this may take 30-60 seconds …"):
            try:
                from src.data_ingestion import DataIngestionService as DIS3
                from src.data_preprocessing import DataPreprocessor
                from src.model_training import ModelTrainer
                from src.model_registry import ModelRegistry
                from src.config import (
                    PREPROCESSOR_PATH,
                    BEST_MODEL_PATH,
                    TARGET_COLUMN,
                )
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import roc_auc_score, f1_score

                # 1. Load all pooled data
                ingest = DIS3()
                df_all = ingest.get_training_data(include_raw=True)
                st.write(f"📦 Loaded {len(df_all):,} rows for training")

                # 2. Preprocess
                preprocessor = DataPreprocessor()
                df_clean = preprocessor.clean(df_all)
                df_encoded = preprocessor.encode(df_clean)

                from src.feature_engineering import FeatureEngineer

                fe = FeatureEngineer()
                df_features = fe.transform(df_encoded)

                y = df_features[TARGET_COLUMN]
                X = df_features.drop(columns=[TARGET_COLUMN])

                preprocessor.fit_scaler(X)
                X_scaled = preprocessor.scale(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )

                # 3. Train best model (XGBoost only for speed on cloud)
                from xgboost import XGBClassifier
                from src.config import MODEL_HYPERPARAMS

                model = XGBClassifier(**MODEL_HYPERPARAMS["xgboost"])
                model.fit(X_train, y_train)

                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)

                new_metrics = {
                    "roc_auc": round(float(roc_auc_score(y_test, y_pred_proba)), 4),
                    "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                }

                dataset_info = {
                    "rows": len(df_all),
                    "features": X_scaled.shape[1],
                    "churn_rate": round(float(y.mean()), 4),
                }

                # 4. Register in model registry
                registry = ModelRegistry()
                version_id = registry.register(
                    model, preprocessor, new_metrics, dataset_info
                )

                # 5. Compare with current production model
                current = registry.get_active_version()
                should_promote = True
                if current:
                    current_auc = current["metrics"].get("roc_auc", 0)
                    margin = RETRAINING["min_improvement_margin"]
                    should_promote = new_metrics["roc_auc"] >= current_auc - margin

                if should_promote and RETRAINING["auto_promote"]:
                    registry.promote(version_id)
                    st.success(
                        f"✅ New model **{version_id}** promoted to production! "
                        f"ROC-AUC: **{new_metrics['roc_auc']:.4f}** | "
                        f"F1: **{new_metrics['f1']:.4f}**"
                    )
                    st.cache_resource.clear()
                else:
                    st.warning(
                        f"⚠️ New model {version_id} registered but NOT promoted "
                        f"(AUC {new_metrics['roc_auc']:.4f} didn't beat current). "
                        f"You can promote it manually from the Registry tab."
                    )

                # Show metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("ROC-AUC", f"{new_metrics['roc_auc']:.4f}")
                m2.metric("F1 Score", f"{new_metrics['f1']:.4f}")
                m3.metric("Training Rows", f"{len(df_all):,}")

            except Exception as e:
                st.error(f"Retraining failed: {e}")
                import traceback

                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Model Registry
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.subheader("📋 Model Registry")

    from src.model_registry import ModelRegistry

    registry = ModelRegistry()
    versions = registry.list_versions()
    active = registry.get_active_version()

    if active:
        st.success(
            f"🟢 Active model: **{active['version_id']}** "
            f"({active['model_class']}) — "
            f"AUC: {active['metrics'].get('roc_auc', '—'):.4f}"
        )
    else:
        st.info("No model promoted yet. Retrain and promote a model first.")

    if versions:
        version_data = []
        for v in versions:
            version_data.append(
                {
                    "Version": v["version_id"],
                    "Model": v["model_class"],
                    "ROC-AUC": v["metrics"].get("roc_auc", "—"),
                    "F1": v["metrics"].get("f1", "—"),
                    "Rows": v["dataset_info"].get("rows", "—"),
                    "Status": v["status"].upper(),
                    "Registered": v["registered_at"][:19],
                }
            )
        st.dataframe(
            pd.DataFrame(version_data),
            use_container_width=True,
        )

        # Promote or rollback
        st.divider()
        col_promote, col_rollback = st.columns(2)

        with col_promote:
            promote_target = st.selectbox(
                "Promote Version",
                [v["version_id"] for v in versions],
            )
            if st.button("⬆️ Promote to Production"):
                registry.promote(promote_target)
                st.success(f"✅ {promote_target} promoted!")
                st.cache_resource.clear()
                st.rerun()

        with col_rollback:
            st.write("")  # spacing
            st.write("")
            if st.button("⏪ Rollback to Previous"):
                prev = registry.rollback()
                if prev:
                    st.success(f"✅ Rolled back to {prev}")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.warning("No previous version to rollback to.")
    else:
        st.info("No model versions registered yet. Train a model first.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Data Drift Detection
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.subheader("📡 Data Drift Detection")
    st.markdown(
        "Compares incoming data distributions against the training baseline "
        "using **Population Stability Index (PSI)**."
    )

    st.markdown(
        """
        | PSI Range | Status | Action |
        |-----------|--------|--------|
        | < 0.10 | 🟢 Stable | No action needed |
        | 0.10 – 0.20 | 🟡 Moderate Drift | Monitor closely |
        | > 0.20 | 🔴 Significant Drift | Consider retraining |
        """
    )

    drift_file = st.file_uploader(
        "Upload a dataset to check for drift",
        type=["csv"],
        key="drift_upload",
    )

    if drift_file and st.button("🔍 Run Drift Detection", type="primary"):
        try:
            from src.monitoring import DataDriftDetector

            df_check = pd.read_csv(drift_file)
            df_check = df_check.drop(columns=["customerID", "Churn"], errors="ignore")

            model = get_model()
            preprocessor = get_preprocessor()
            if preprocessor:
                df_check = preprocessor.transform(df_check)

            detector = DataDriftDetector()
            report = detector.detect(df_check)

            if report.get("error"):
                st.warning(report["error"])
            else:
                if report["overall_drift_detected"]:
                    st.error(
                        f"🔴 Drift detected in **{len(report['drifted_columns'])}** "
                        f"column(s): {report['drifted_columns']}"
                    )
                else:
                    st.success("🟢 No significant data drift detected.")

                # PSI per feature
                if report["columns"]:
                    drift_df = pd.DataFrame.from_dict(
                        report["columns"], orient="index"
                    ).reset_index()
                    drift_df.columns = ["Feature", "PSI", "Status"]
                    drift_df = drift_df.sort_values("PSI", ascending=False)

                    import plotly.express as px

                    fig = px.bar(
                        drift_df.head(20),
                        x="PSI",
                        y="Feature",
                        orientation="h",
                        color="Status",
                        color_discrete_map={
                            "stable": "#2ECC71",
                            "moderate_drift": "#F39C12",
                            "significant_drift": "#E74C3C",
                        },
                        title="Population Stability Index by Feature",
                    )
                    fig.add_vline(
                        x=0.2,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Drift threshold",
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(drift_df, use_container_width=True)

        except Exception as e:
            st.error(f"Drift detection failed: {e}")
