"""
src/model_registry.py
─────────────────────────────────────────────────────────────────────────────
ModelRegistry — local model versioning, promotion, and rollback.

Every trained model gets a version entry with:
  • Model + preprocessor pickle paths
  • Training metrics (ROC-AUC, F1, precision, recall)
  • Dataset stats (rows, features, churn rate)
  • Timestamp and status (active / archived)

The registry supports:
  • register()  — save a new model version
  • promote()   — set a version as the active production model
  • rollback()  — revert to the previous active version
  • compare()   — compare two versions side-by-side

Usage:
    registry = ModelRegistry()
    version  = registry.register(model, preprocessor, metrics, dataset_info)
    registry.promote(version)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.config import (
    BEST_MODEL_PATH,
    MODEL_REGISTRY_DIR,
    PREPROCESSOR_PATH,
    REGISTRY_INDEX_PATH,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    File-based model registry with versioning and promotion semantics.

    Registry layout:
        models/registry/
        ├── registry.json          # version index
        ├── v1/
        │   ├── model.pkl
        │   └── preprocessor.pkl
        ├── v2/
        │   ├── model.pkl
        │   └── preprocessor.pkl
        └── ...
    """

    def __init__(self, registry_dir: Path = MODEL_REGISTRY_DIR) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = REGISTRY_INDEX_PATH
        self._index = self._load_index()

    # ── Index persistence ─────────────────────────────────────────────────────

    def _load_index(self) -> Dict:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {"active_version": None, "versions": []}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    # ── Register ──────────────────────────────────────────────────────────────

    def register(
        self,
        model: Any,
        preprocessor: Any,
        metrics: Dict[str, float],
        dataset_info: Optional[Dict] = None,
    ) -> str:
        """
        Save a new model version to the registry.

        Parameters
        ----------
        model : trained sklearn-compatible estimator
        preprocessor : fitted DataPreprocessor
        metrics : {"roc_auc": float, "f1": float, ...}
        dataset_info : {"rows": int, "features": int, "churn_rate": float, ...}

        Returns
        -------
        version_id : str (e.g. "v3")
        """
        version_num = len(self._index["versions"]) + 1
        version_id = f"v{version_num}"
        version_dir = self.registry_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save artefacts
        model_path = version_dir / "model.pkl"
        preprocessor_path = version_dir / "preprocessor.pkl"
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        entry = {
            "version_id": version_id,
            "model_class": type(model).__name__,
            "metrics": metrics,
            "dataset_info": dataset_info or {},
            "status": "registered",
            "registered_at": datetime.utcnow().isoformat(),
            "promoted_at": None,
        }
        self._index["versions"].append(entry)
        self._save_index()

        logger.info(
            "Registered model %s | AUC=%.4f F1=%.4f",
            version_id,
            metrics.get("roc_auc", 0),
            metrics.get("f1", 0),
        )
        return version_id

    # ── Promote ───────────────────────────────────────────────────────────────

    def promote(self, version_id: str) -> Dict:
        """
        Set a version as the active production model.

        Copies model + preprocessor to the standard BEST_MODEL_PATH so
        the API and dashboard pick it up automatically.
        """
        entry = self._find_version(version_id)
        if not entry:
            raise ValueError(f"Version {version_id} not found in registry")

        version_dir = self.registry_dir / version_id
        model_src = version_dir / "model.pkl"
        preprocessor_src = version_dir / "preprocessor.pkl"

        if not model_src.exists():
            raise FileNotFoundError(f"Model artefact missing for {version_id}")

        # Copy to production paths
        shutil.copy2(model_src, BEST_MODEL_PATH)
        shutil.copy2(preprocessor_src, PREPROCESSOR_PATH)

        # Update statuses
        for v in self._index["versions"]:
            if v["status"] == "active":
                v["status"] = "archived"
        entry["status"] = "active"
        entry["promoted_at"] = datetime.utcnow().isoformat()
        self._index["active_version"] = version_id
        self._save_index()

        logger.info("Promoted %s to production", version_id)
        return entry

    # ── Rollback ──────────────────────────────────────────────────────────────

    def rollback(self) -> Optional[str]:
        """Revert to the most recent archived version."""
        archived = [v for v in self._index["versions"] if v["status"] == "archived"]
        if not archived:
            logger.warning("No archived version to rollback to")
            return None

        previous = archived[-1]
        self.promote(previous["version_id"])
        logger.info("Rolled back to %s", previous["version_id"])
        return previous["version_id"]

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_active_version(self) -> Optional[Dict]:
        """Return metadata for the currently active model."""
        for v in self._index["versions"]:
            if v["status"] == "active":
                return v
        return None

    def list_versions(self) -> List[Dict]:
        """Return all registered versions (newest first)."""
        return list(reversed(self._index["versions"]))

    def compare(self, v1_id: str, v2_id: str) -> Dict:
        """Compare metrics of two versions side-by-side."""
        v1 = self._find_version(v1_id)
        v2 = self._find_version(v2_id)
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        return {
            v1_id: v1["metrics"],
            v2_id: v2["metrics"],
            "diff": {
                k: round(v2["metrics"].get(k, 0) - v1["metrics"].get(k, 0), 6)
                for k in v1["metrics"]
            },
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_version(self, version_id: str) -> Optional[Dict]:
        for v in self._index["versions"]:
            if v["version_id"] == version_id:
                return v
        return None
