"""
src/explainability.py
─────────────────────────────────────────────────────────────────────────────
SHAPExplainer — model-agnostic SHAP wrapper.

Provides:
  • Per-customer waterfall explanation (JSON-serializable for REST API)
  • Global feature importance summary plot
  • Force plot export

Supported model types:
  • Tree-based (RF, XGBoost, CatBoost)  → TreeExplainer
  • Linear (Logistic Regression)        → LinearExplainer
  • Any other                           → KernelExplainer (slow, last resort)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import EXPLAINABILITY, FIGURES_DIR

logger = logging.getLogger(__name__)

# Tree-based model classes (for auto-detection)
_TREE_MODELS = (
    "RandomForestClassifier",
    "XGBClassifier",
    "CatBoostClassifier",
    "GradientBoostingClassifier",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
)
_LINEAR_MODELS = ("LogisticRegression", "LinearSVC", "SGDClassifier")


class SHAPExplainer:
    """
    Unified SHAP explanation interface.

    Usage
    -----
    explainer = SHAPExplainer(model, X_train)
    explanation = explainer.explain_instance(X_single_row)
    explainer.plot_summary(X_test)
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        background_samples: int = EXPLAINABILITY["background_samples"],
    ) -> None:
        self.model = model
        self.feature_names = X_train.columns.tolist()
        self.background_samples = background_samples
        self._explainer = self._build_explainer(model, X_train)
        logger.info(
            "SHAPExplainer initialised | type=%s features=%d",
            type(self._explainer).__name__,
            len(self.feature_names),
        )

    # ── Explainer Factory ────────────────────────────────────────────────────

    def _build_explainer(self, model: Any, X_train: pd.DataFrame) -> Any:
        model_class = type(model).__name__

        if model_class in _TREE_MODELS:
            logger.info("Using TreeExplainer for %s", model_class)
            return shap.TreeExplainer(model)

        if model_class in _LINEAR_MODELS:
            logger.info("Using LinearExplainer for %s", model_class)
            background = shap.maskers.Independent(X_train)
            return shap.LinearExplainer(model, background)

        # Fallback: KernelExplainer (model-agnostic but slow)
        logger.warning(
            "Unknown model type %s — using KernelExplainer (slow). "
            "Consider using a tree or linear model.",
            model_class,
        )
        background = X_train.sample(
            min(self.background_samples, len(X_train)),
            random_state=42,
        )
        return shap.KernelExplainer(model.predict_proba, background)

    # ── Per-instance explanation ──────────────────────────────────────────────

    def explain_instance(
        self,
        X: pd.DataFrame,
        top_n: int = EXPLAINABILITY["top_n_features"],
    ) -> Dict:
        """
        Generate a JSON-serializable SHAP explanation for one or more rows.

        Returns
        -------
        {
          "expected_value": float,
          "churn_probability": float,
          "top_drivers": [
            {"feature": str, "value": float, "shap_value": float,
             "direction": "increases_churn" | "decreases_churn"},
            ...
          ]
        }
        """
        shap_values = self._explainer.shap_values(X)

        # For binary classifiers some explainers return a list [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = churn

        base_value = float(
            self._explainer.expected_value[1]
            if isinstance(self._explainer.expected_value, (list, np.ndarray))
            else self._explainer.expected_value
        )

        row_shap = shap_values[0] if shap_values.ndim == 2 else shap_values
        feature_vals = X.iloc[0].values if hasattr(X, "iloc") else X[0]

        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(row_shap))[::-1][:top_n]

        top_drivers = [
            {
                "feature": self.feature_names[i],
                "value": float(feature_vals[i]),
                "shap_value": float(row_shap[i]),
                "direction": (
                    "increases_churn" if row_shap[i] > 0 else "decreases_churn"
                ),
            }
            for i in indices
        ]

        churn_prob = float(base_value + row_shap.sum())
        # Clamp to [0, 1]
        churn_prob = max(0.0, min(1.0, churn_prob))

        return {
            "expected_value": base_value,
            "churn_probability": churn_prob,
            "top_drivers": top_drivers,
        }

    # ── Global summary plot ───────────────────────────────────────────────────

    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        save: bool = True,
    ) -> Optional[Path]:
        """Generate a SHAP summary (beeswarm) plot for the full dataset."""
        logger.info("Computing SHAP values for summary plot (%d rows) …", len(X))
        shap_values = self._explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Feature Importance — Global Summary", fontsize=14,
                  fontweight="bold")
        plt.tight_layout()

        if save:
            path = FIGURES_DIR / "shap_summary.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("SHAP summary plot saved → %s", path)
            return path

        plt.show()
        return None

    def plot_waterfall(
        self,
        X: pd.DataFrame,
        row_index: int = 0,
        save: bool = True,
        filename: str = "shap_waterfall.png",
    ) -> Optional[Path]:
        """Generate a SHAP waterfall plot for a single prediction."""
        shap_values = self._explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        base_value = (
            self._explainer.expected_value[1]
            if isinstance(self._explainer.expected_value, (list, np.ndarray))
            else self._explainer.expected_value
        )

        explanation = shap.Explanation(
            values=shap_values[row_index],
            base_values=float(base_value),
            data=X.iloc[row_index].values,
            feature_names=self.feature_names,
        )

        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()

        if save:
            path = FIGURES_DIR / filename
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("Waterfall plot saved → %s", path)
            return path

        plt.show()
        return None

# ── Entry point ───────────────────────────────────────────────────────────────

def generate_shap_plots():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    try:
        from src.model_training import ModelTrainer
        from src.config import BEST_MODEL_PATH
        
        logger.info("Loading model and data for SHAP evaluation…")
        model = ModelTrainer.load_best_model(BEST_MODEL_PATH)
        trainer = ModelTrainer(use_smote=False)
        X_train, X_test, _, _ = trainer.load_data()
        
        # Take a sample for summary plot to be fast
        bg_sample = X_train.sample(min(100, len(X_train)), random_state=42)
        explainer = SHAPExplainer(model, bg_sample)
        
        test_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer.plot_summary(test_sample, save=True)
        logger.info("SHAP plots generated successfully!")
    except Exception as e:
        logger.error(f"SHAP Extraction failed: {e}")

if __name__ == "__main__":
    generate_shap_plots()
