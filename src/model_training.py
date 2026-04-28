"""
src/model_training.py
─────────────────────────────────────────────────────────────────────────────
ModelTrainer — trains, cross-validates, and persists ML models for churn
prediction. Integrates MLflow for full experiment tracking.

Supported algorithms:
  • Logistic Regression (baseline)
  • Random Forest
  • XGBoost
  • CatBoost

Usage:
    trainer = ModelTrainer()
    best_model, results = trainer.train_all()
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

try:
    import mlflow
    import mlflow.sklearn

    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_NAMES_PATH,
    MODEL_METADATA_PATH,
    MLFLOW,
    MODEL_HYPERPARAMS,
    PROCESSED_X_TRAIN_PATH,
    PROCESSED_X_TEST_PATH,
    PROCESSED_Y_TRAIN_PATH,
    PROCESSED_Y_TEST_PATH,
    TRAINING,
)

logger = logging.getLogger(__name__)


# ── Model Factory ─────────────────────────────────────────────────────────────


def _build_models() -> Dict[str, Any]:
    """Instantiate all classifiers from config hyperparameters."""
    return {
        "logistic_regression": LogisticRegression(
            **MODEL_HYPERPARAMS["logistic_regression"]
        ),
        "random_forest": RandomForestClassifier(**MODEL_HYPERPARAMS["random_forest"]),
        "xgboost": XGBClassifier(**MODEL_HYPERPARAMS["xgboost"]),
        "catboost": CatBoostClassifier(**MODEL_HYPERPARAMS["catboost"]),
    }


# ── Main Trainer Class ────────────────────────────────────────────────────────


class ModelTrainer:
    """
    Orchestrates full model training lifecycle:
      1. Load processed data
      2. Optionally apply SMOTE to address class imbalance
      3. Train each model with stratified cross-validation
      4. Log all metrics and artifacts to MLflow
      5. Select and persist the best model

    Attributes
    ----------
    results : dict — CV scores for each trained model
    best_model_name : str — name of the winning model
    best_model : fitted sklearn-compatible estimator
    """

    def __init__(self, use_smote: bool = TRAINING["use_smote"]) -> None:
        self.use_smote = use_smote
        self.cv = StratifiedKFold(
            n_splits=TRAINING["cv_folds"],
            shuffle=True,
            random_state=TRAINING["random_state"],
        )
        self.scoring_metric: str = TRAINING["scoring_metric"]
        self.results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None

        # Configure MLflow
        if _HAS_MLFLOW:
            mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
            mlflow.set_experiment(MLFLOW["experiment_name"])

        logger.info(
            "ModelTrainer initialised | use_smote=%s cv_folds=%d scoring=%s",
            use_smote,
            TRAINING["cv_folds"],
            self.scoring_metric,
        )

    # ── Data Loaders ──────────────────────────────────────────────────────────

    def load_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load processed train/test splits from disk."""
        logger.info("Loading processed data …")
        X_train = pd.read_csv(PROCESSED_X_TRAIN_PATH)
        X_test = pd.read_csv(PROCESSED_X_TEST_PATH)
        y_train = pd.read_csv(PROCESSED_Y_TRAIN_PATH).squeeze()
        y_test = pd.read_csv(PROCESSED_Y_TEST_PATH).squeeze()
        logger.info(
            "Loaded | X_train=%s X_test=%s churn_rate=%.3f",
            X_train.shape,
            X_test.shape,
            y_train.mean(),
        )
        return X_train, X_test, y_train, y_test

    # ── SMOTE ─────────────────────────────────────────────────────────────────

    def apply_smote(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Oversample the minority class using SMOTE."""
        smote = SMOTE(random_state=TRAINING["smote_random_state"])
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE applied | before=%d after=%d churn_rate=%.3f",
            len(y_train),
            len(y_res),
            y_res.mean(),
        )
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)

    # ── Cross-Validation ──────────────────────────────────────────────────────

    def cross_validate_model(
        self,
        name: str,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """
        Manual stratified K-fold CV — computes all metrics fold-by-fold.

        Deliberately avoids sklearn's cross_validate() which calls is_classifier()
        → get_tags() → __sklearn_tags__(), breaking on CatBoost ≤1.2 and sklearn ≥1.6.
        Using clone() + predict_proba() directly is fully compatible with all estimators.
        """
        logger.info("Cross-validating %s …", name)

        fold_auc, fold_f1, fold_prec, fold_rec, fold_ap = [], [], [], [], []

        X_arr = X.reset_index(drop=True)
        y_arr = y.reset_index(drop=True)

        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(X_arr, y_arr), start=1
        ):
            X_fold_train = X_arr.iloc[train_idx]
            X_fold_val = X_arr.iloc[val_idx]
            y_fold_train = y_arr.iloc[train_idx]
            y_fold_val = y_arr.iloc[val_idx]

            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)

            y_proba = fold_model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            fold_auc.append(roc_auc_score(y_fold_val, y_proba))
            fold_f1.append(f1_score(y_fold_val, y_pred, zero_division=0))
            fold_prec.append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_rec.append(recall_score(y_fold_val, y_pred, zero_division=0))
            fold_ap.append(average_precision_score(y_fold_val, y_proba))

        scores = {
            "roc_auc": float(np.mean(fold_auc)),
            "f1": float(np.mean(fold_f1)),
            "precision": float(np.mean(fold_prec)),
            "recall": float(np.mean(fold_rec)),
            "average_precision": float(np.mean(fold_ap)),
        }
        logger.info(
            "%s CV | AUC=%.4f F1=%.4f P=%.4f R=%.4f",
            name,
            scores["roc_auc"],
            scores["f1"],
            scores["precision"],
            scores["recall"],
        )
        return scores

    # ── Train All ────────────────────────────────────────────────────────────

    def train_all(
        self,
    ) -> Tuple[Any, Dict[str, Dict]]:
        """
        End-to-end training run:
          - Cross-validates all models
          - Selects the best by ROC-AUC
          - Re-trains best model on full training set
          - Logs everything to MLflow
          - Saves best model + metadata to disk

        Returns
        -------
        best_model, results_dict
        """
        X_train, X_test, y_train, y_test = self.load_data()

        # Apply SMOTE on training set only
        X_smote, y_smote = (
            self.apply_smote(X_train, y_train) if self.use_smote else (X_train, y_train)
        )

        models = _build_models()
        feature_names = X_train.columns.tolist()

        for name, model in models.items():
            _mlflow_ctx = (
                mlflow.start_run(run_name=name, tags=MLFLOW.get("run_tags", {}))
                if _HAS_MLFLOW
                else None
            )
            try:
                if _mlflow_ctx:
                    _mlflow_ctx.__enter__()
                    # Log model params
                    mlflow.log_params(MODEL_HYPERPARAMS.get(name, {}))
                    mlflow.log_param("use_smote", self.use_smote)
                    mlflow.log_param("cv_folds", TRAINING["cv_folds"])

                # Cross-validation
                t0 = time.time()
                cv_scores = self.cross_validate_model(name, model, X_smote, y_smote)
                cv_time = time.time() - t0

                if _HAS_MLFLOW:
                    # Log CV metrics
                    for metric, value in cv_scores.items():
                        mlflow.log_metric(f"cv_{metric}", value)
                    mlflow.log_metric("cv_time_seconds", cv_time)

                # Full refit on training data
                t1 = time.time()
                model.fit(X_smote, y_smote)
                fit_time = time.time() - t1

                # Hold-out test set evaluation
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)

                test_scores = {
                    "test_roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
                    "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "test_precision": float(
                        precision_score(y_test, y_pred, zero_division=0)
                    ),
                    "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "test_avg_precision": float(
                        average_precision_score(y_test, y_pred_proba)
                    ),
                }
                if _HAS_MLFLOW:
                    mlflow.log_metrics(test_scores)
                    mlflow.log_metric("fit_time_seconds", fit_time)
                    # Log model artefact
                    mlflow.sklearn.log_model(model, name)

                self.results[name] = {
                    "cv": cv_scores,
                    "test": test_scores,
                    "model": model,
                    "cv_time": cv_time,
                    "fit_time": fit_time,
                }
            finally:
                if _mlflow_ctx:
                    _mlflow_ctx.__exit__(None, None, None)
                logger.info(
                    "%s | Test AUC=%.4f F1=%.4f",
                    name,
                    test_scores["test_roc_auc"],
                    test_scores["test_f1"],
                )

        # Select best model by CV ROC-AUC
        self.best_model_name = max(
            self.results,
            key=lambda n: self.results[n]["cv"]["roc_auc"],
        )
        self.best_model = self.results[self.best_model_name]["model"]
        logger.info("Best model: %s", self.best_model_name)

        # Save artifacts
        self._save_best_model(feature_names, y_test, X_test)

        return self.best_model, self.results

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_best_model(
        self,
        feature_names: List[str],
        y_test: pd.Series,
        X_test: pd.DataFrame,
    ) -> None:
        """Save best model, feature names, and metadata to disk."""
        BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Model pickle
        joblib.dump(self.best_model, BEST_MODEL_PATH)
        logger.info("Best model saved → %s", BEST_MODEL_PATH)

        # Feature names
        with open(FEATURE_NAMES_PATH, "w") as f:
            json.dump(feature_names, f, indent=2)

        # Model metadata (for /health endpoint)
        best_test = self.results[self.best_model_name]["test"]
        metadata = {
            "model_name": self.best_model_name,
            "model_class": type(self.best_model).__name__,
            "feature_count": len(feature_names),
            "test_roc_auc": best_test["test_roc_auc"],
            "test_f1": best_test["test_f1"],
            "test_precision": best_test["test_precision"],
            "test_recall": best_test["test_recall"],
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "use_smote": self.use_smote,
        }
        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Model metadata saved → %s", MODEL_METADATA_PATH)

    # ── Static Loaders ────────────────────────────────────────────────────────

    @staticmethod
    def load_best_model(path: Path = BEST_MODEL_PATH) -> Any:
        """Load the persisted best model from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
        model = joblib.load(path)
        logger.info("Model loaded ← %s", path)
        return model

    @staticmethod
    def load_metadata(path: Path = MODEL_METADATA_PATH) -> Dict:
        """Load model metadata JSON."""
        if not Path(path).exists():
            return {}
        with open(path) as f:
            return json.load(f)


# ── Entry point ───────────────────────────────────────────────────────────────


def run_training() -> Tuple[Any, Dict]:
    """Run the complete training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    trainer = ModelTrainer()
    best_model, results = trainer.train_all()
    logger.info("Training complete. Best model: %s", trainer.best_model_name)
    return best_model, results


if __name__ == "__main__":
    run_training()
