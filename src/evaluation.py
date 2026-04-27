"""
src/evaluation.py
─────────────────────────────────────────────────────────────────────────────
ModelEvaluator — computes metrics, generates and saves all evaluation plots.

Outputs saved to figures/:
  • roc_curve.png
  • pr_curve.png
  • confusion_matrix.png
  • feature_importance.png
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import EVALUATION, FIGURES_DIR

logger = logging.getLogger(__name__)

# ── Plotting style ────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = {"positive": "#E74C3C", "negative": "#2ECC71", "neutral": "#3498DB"}


class ModelEvaluator:
    """
    Evaluates a trained model on a held-out test set and produces:
      • Classification metrics dict
      • ROC-AUC curve plot
      • Precision-Recall curve plot
      • Confusion matrix heatmap
      • Feature importance bar chart
    """

    def __init__(
        self,
        threshold: float = EVALUATION["classification_threshold"],
        figures_dir: Path = FIGURES_DIR,
    ) -> None:
        self.threshold = threshold
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Main evaluate method ──────────────────────────────────────────────────

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        save_plots: bool = True,
    ) -> Dict[str, float]:
        """
        Full evaluation suite.

        Returns
        -------
        metrics : dict with all scalar metrics
        """
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        metrics = self._compute_metrics(y_test, y_pred, y_proba)

        logger.info(
            "[%s] AUC=%.4f F1=%.4f P=%.4f R=%.4f",
            model_name,
            metrics["roc_auc"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )

        if save_plots:
            self.plot_roc_curve(y_test, y_proba, model_name)
            self.plot_pr_curve(y_test, y_proba, model_name)
            self.plot_confusion_matrix(y_test, y_pred, model_name)

            if hasattr(model, "feature_importances_"):
                self.plot_feature_importance(
                    model.feature_importances_, X_test.columns.tolist(), model_name
                )

        return metrics

    # ── Metrics ──────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "average_precision": float(average_precision_score(y_true, y_proba)),
        }

    def classification_report(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> str:
        """Return sklearn classification report as a formatted string."""
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)
        return classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"]
        )

    # ── Plots ────────────────────────────────────────────────────────────────

    def plot_roc_curve(
        self,
        y_test: pd.Series,
        y_proba: np.ndarray,
        model_name: str = "model",
    ) -> Path:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color=PALETTE["positive"], lw=2,
                label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["positive"])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.figures_dir / "roc_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curve saved → %s", path)
        return path

    def plot_pr_curve(
        self,
        y_test: pd.Series,
        y_proba: np.ndarray,
        model_name: str = "model",
    ) -> Path:
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        baseline = y_test.mean()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.step(recall, precision, color=PALETTE["neutral"], lw=2, where="post",
                label=f"PR (AP = {ap:.4f})")
        ax.axhline(baseline, ls="--", color="grey", lw=1,
                   label=f"Baseline (churn rate = {baseline:.2f})")
        ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["neutral"])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14,
                     fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.figures_dir / "pr_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("PR curve saved → %s", path)
        return path

    def plot_confusion_matrix(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        model_name: str = "model",
    ) -> Path:
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        labels = np.array(
            [[f"{v}\n({p:.1%})" for v, p in zip(row_v, row_p)]
             for row_v, row_p in zip(cm, cm_pct)]
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_pct,
            annot=labels,
            fmt="",
            cmap="RdYlGn_r",
            ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            linewidths=0.5,
            cbar_kws={"label": "Proportion"},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14,
                     fontweight="bold")
        plt.tight_layout()

        path = self.figures_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", path)
        return path

    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        model_name: str = "model",
        top_n: int = 20,
    ) -> Path:
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = [PALETTE["positive"] if i == 0 else PALETTE["neutral"]
                  for i in range(len(top_features))]
        bars = ax.barh(top_features[::-1], top_importances[::-1],
                       color=colors[::-1], edgecolor="white", height=0.7)
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                     fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, top_importances[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        plt.tight_layout()

        path = self.figures_dir / "feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_plots():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    try:
        from src.model_training import ModelTrainer
        from src.config import BEST_MODEL_PATH
        
        logger.info("Loading model and data for evaluation…")
        model = ModelTrainer.load_best_model(BEST_MODEL_PATH)
        trainer = ModelTrainer(use_smote=False)  # just using it for load_data
        _, X_test, _, y_test = trainer.load_data()
        
        evaluator = ModelEvaluator()
        evaluator.evaluate(model, X_test, y_test, model_name="Best Model", save_plots=True)
        logger.info("All performance plots generated successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    generate_plots()
