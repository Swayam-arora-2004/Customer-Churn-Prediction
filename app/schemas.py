"""
app/schemas.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 request and response models for the Churn Prediction REST API.

All models are strictly typed and validated at request time.
─────────────────────────────────────────────────────────────────────────────
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request Models ────────────────────────────────────────────────────────────


class CustomerFeatures(BaseModel):
    """Raw customer features as received from the caller (pre-encoding)."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=720, description="Months with company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0)
    TotalCharges: float = Field(..., ge=0.0)

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_consistent(cls, v: float, info: Any) -> float:
        tenure = info.data.get("tenure", 0)
        monthly = info.data.get("MonthlyCharges", 0)
        if tenure == 0 and v != 0.0:
            return 0.0  # auto-correct, don't reject
        return v

    model_config = {"extra": "forbid"}


class PredictRequest(BaseModel):
    customer_id: Optional[str] = Field(None, description="Optional customer identifier")
    features: CustomerFeatures


class BatchPredictRequest(BaseModel):
    customers: List[PredictRequest] = Field(
        ..., min_length=1, max_length=1000
    )


# ── Response Models ───────────────────────────────────────────────────────────


class SHAPDriver(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: Literal["increases_churn", "decreases_churn"]


class RetentionAction(BaseModel):
    action_id: str
    title: str
    description: str
    category: str
    impact_score: float
    priority: str
    tags: List[str]


class PredictionResult(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    will_churn: bool
    risk_segment: str
    confidence: str   # "high" | "medium" | "low" based on probability distance from 0.5


class ExplanationResult(BaseModel):
    customer_id: Optional[str]
    expected_value: float
    churn_probability: float
    top_drivers: List[SHAPDriver]


class RecommendationResult(BaseModel):
    customer_id: Optional[str]
    customer_segment: str
    churn_probability: float
    recommendations: List[RetentionAction]
    estimated_retention_lift: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_class: str
    feature_count: int
    test_roc_auc: float
    test_f1: float
    trained_at: str
    uptime_seconds: float
    version: str


class MetricsResponse(BaseModel):
    model_name: str
    test_roc_auc: float
    test_f1: float
    test_precision: float
    test_recall: float
    trained_at: str


# ── API Envelope ──────────────────────────────────────────────────────────────


class APIResponse(BaseModel):
    """Standard JSON envelope for all API responses."""
    status: Literal["success", "error"]
    data: Optional[Any] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
