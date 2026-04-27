"""
tests/test_model_training.py
─────────────────────────────────────────────────────────────────────────────
Unit + integration tests for src/model_training.py and src/explainability.py.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_data():
    """Synthetic preprocessed dataset for model tests (no file I/O needed)."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        np.random.randn(n, 10),
        columns=[f"feature_{i}" for i in range(10)],
    )
    y = pd.Series(np.random.randint(0, 2, n), name="Churn")
    return X, y


@pytest.fixture
def trained_lr(synthetic_data):
    """Return a fitted LogisticRegression for reuse."""
    X, y = synthetic_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_rf(synthetic_data):
    """Return a fitted RandomForestClassifier for reuse."""
    X, y = synthetic_data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X, y


# ── ModelTrainer unit tests ───────────────────────────────────────────────────


class TestModelTrainer:
    """Test the ModelTrainer class without running the full pipeline."""

    def test_apply_smote_balances_classes(self, synthetic_data):
        """SMOTE should produce an approximately balanced dataset."""
        from src.model_training import ModelTrainer

        X, y = synthetic_data
        # Force imbalance
        minority_idx = y[y == 1].index[:20]
        y_imbalanced = y.copy()
        y_imbalanced[minority_idx] = 1
        y_imbalanced[~y_imbalanced.index.isin(minority_idx)] = 0

        trainer = ModelTrainer(use_smote=True)
        X_res, y_res = trainer.apply_smote(X, y_imbalanced)
        ratio = y_res.mean()
        assert 0.4 <= ratio <= 0.6, f"SMOTE ratio {ratio} not balanced"

    def test_load_best_model_raises_when_missing(self, tmp_path):
        from src.model_training import ModelTrainer

        with pytest.raises(FileNotFoundError):
            ModelTrainer.load_best_model(tmp_path / "nonexistent.pkl")

    def test_load_metadata_returns_empty_when_missing(self, tmp_path):
        from src.model_training import ModelTrainer

        result = ModelTrainer.load_metadata(tmp_path / "metadata.json")
        assert result == {}

    def test_save_and_load_model(self, trained_rf, tmp_path):
        import joblib
        from src.model_training import ModelTrainer

        model, X, y = trained_rf
        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)
        loaded = ModelTrainer.load_best_model(model_path)
        assert hasattr(loaded, "predict_proba")


# ── Prediction correctness ────────────────────────────────────────────────────


class TestPredictions:
    def test_predict_proba_shape(self, trained_lr):
        model, X, y = trained_lr
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sum_to_one(self, trained_lr):
        model, X, y = trained_lr
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, trained_lr):
        model, X, y = trained_lr
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_rf_predict_proba_in_range(self, trained_rf):
        model, X, y = trained_rf
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_threshold_produces_binary_output(self, trained_lr):
        model, X, y = trained_lr
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        assert set(pred).issubset({0, 1})


# ── ModelEvaluator tests ──────────────────────────────────────────────────────


class TestModelEvaluator:
    def test_evaluate_returns_all_metrics(self, trained_rf, tmp_path):
        from src.evaluation import ModelEvaluator

        model, X, y = trained_rf
        evaluator = ModelEvaluator(figures_dir=tmp_path)
        metrics = evaluator.evaluate(model, X, y, save_plots=False)
        assert "roc_auc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "average_precision" in metrics

    def test_roc_auc_between_0_and_1(self, trained_rf, tmp_path):
        from src.evaluation import ModelEvaluator

        model, X, y = trained_rf
        evaluator = ModelEvaluator(figures_dir=tmp_path)
        metrics = evaluator.evaluate(model, X, y, save_plots=False)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_roc_curve_plot_saved(self, trained_rf, tmp_path):
        from src.evaluation import ModelEvaluator

        model, X, y = trained_rf
        evaluator = ModelEvaluator(figures_dir=tmp_path)
        y_proba = model.predict_proba(X)[:, 1]
        path = evaluator.plot_roc_curve(y, y_proba, "test_model")
        assert path.exists()

    def test_confusion_matrix_plot_saved(self, trained_rf, tmp_path):
        from src.evaluation import ModelEvaluator

        model, X, y = trained_rf
        evaluator = ModelEvaluator(figures_dir=tmp_path)
        y_pred = model.predict(X)
        path = evaluator.plot_confusion_matrix(y, y_pred, "test_model")
        assert path.exists()

    def test_feature_importance_plot_saved(self, trained_rf, tmp_path):
        from src.evaluation import ModelEvaluator

        model, X, y = trained_rf
        evaluator = ModelEvaluator(figures_dir=tmp_path)
        path = evaluator.plot_feature_importance(
            model.feature_importances_, X.columns.tolist(), "test_rf"
        )
        assert path.exists()


# ── SHAPExplainer tests ───────────────────────────────────────────────────────


class TestSHAPExplainer:
    def test_explain_instance_structure(self, trained_rf):
        from src.explainability import SHAPExplainer

        model, X, y = trained_rf
        explainer = SHAPExplainer(model, X)
        explanation = explainer.explain_instance(X.iloc[[0]])
        assert "expected_value" in explanation
        assert "churn_probability" in explanation
        assert "top_drivers" in explanation

    def test_churn_probability_in_range(self, trained_rf):
        from src.explainability import SHAPExplainer

        model, X, y = trained_rf
        explainer = SHAPExplainer(model, X)
        for i in range(min(5, len(X))):
            exp = explainer.explain_instance(X.iloc[[i]])
            prob = exp["churn_probability"]
            assert 0.0 <= prob <= 1.0, f"Row {i}: probability {prob} out of range"

    def test_top_drivers_has_required_keys(self, trained_rf):
        from src.explainability import SHAPExplainer

        model, X, y = trained_rf
        explainer = SHAPExplainer(model, X)
        exp = explainer.explain_instance(X.iloc[[0]])
        for driver in exp["top_drivers"]:
            assert "feature" in driver
            assert "shap_value" in driver
            assert "direction" in driver
            assert driver["direction"] in ("increases_churn", "decreases_churn")

    def test_lr_explainer_works(self, trained_lr):
        """LinearExplainer path should work for LogisticRegression."""
        from src.explainability import SHAPExplainer

        model, X, y = trained_lr
        explainer = SHAPExplainer(model, X)
        exp = explainer.explain_instance(X.iloc[[0]])
        assert "churn_probability" in exp

    def test_top_n_respected(self, trained_rf):
        from src.explainability import SHAPExplainer

        model, X, y = trained_rf
        explainer = SHAPExplainer(model, X)
        exp = explainer.explain_instance(X.iloc[[0]], top_n=3)
        assert len(exp["top_drivers"]) <= 3
