"""
src/prevention.py
─────────────────────────────────────────────────────────────────────────────
RetentionEngine — generates targeted, ranked retention actions for at-risk
customers based on their churn probability and SHAP-identified drivers.

Design principles:
  • Rule-driven decision trees grounded in telecom domain knowledge
  • Each action has an `impact_score` (estimated churn reduction probability)
  • Actions are ranked by impact and deduplicated
  • Maximum N recommendations returned (configurable via config.py)
  • Fully JSON-serializable output for REST API consumption

Risk Segments:
  • High   : churn_prob ≥ 0.70  → aggressive immediate intervention
  • Medium : 0.30 ≤ churn_prob < 0.70 → targeted incentives
  • Low    : churn_prob < 0.30  → light-touch engagement
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import PREVENTION

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────


@dataclass
class RetentionAction:
    """A single recommended retention action."""

    action_id: str
    title: str
    description: str
    category: str  # "pricing" | "contract" | "service" | "support" | "engagement"
    impact_score: float  # Estimated reduction in churn probability [0.0 – 1.0]
    priority: str  # "critical" | "high" | "medium" | "low"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "impact_score": self.impact_score,
            "priority": self.priority,
            "tags": self.tags,
        }


# ── Action Library ────────────────────────────────────────────────────────────
# All possible retention actions — the engine selects & ranks relevant ones.

_ACTION_LOYALTY_DISCOUNT = RetentionAction(
    action_id="ACT_LOYALTY_DISCOUNT",
    title="Loyalty Discount Offer",
    description=(
        "Offer a 15–20% monthly discount for the next 3 months as a loyalty reward. "
        "Include a personal outreach call from the account manager."
    ),
    category="pricing",
    impact_score=0.72,
    priority="critical",
    tags=["high_charges", "high_tenure", "discount"],
)

_ACTION_CONTRACT_UPGRADE = RetentionAction(
    action_id="ACT_CONTRACT_UPGRADE",
    title="1-Year Contract Incentive",
    description=(
        "Offer month-to-month customers a transition to a 1-year contract with "
        "a free month and a locked-in rate guarantee. "
        "Reduces price anxiety and increases switching cost."
    ),
    category="contract",
    impact_score=0.68,
    priority="critical",
    tags=["month_to_month", "contract"],
)

_ACTION_AUTOPAY_INCENTIVE = RetentionAction(
    action_id="ACT_AUTOPAY_INCENTIVE",
    title="Auto-Pay Enrollment Incentive",
    description=(
        "Offer a $5/month bill credit for enrolling in automatic bank transfer payment. "
        "Reduces churn risk associated with payment friction."
    ),
    category="pricing",
    impact_score=0.45,
    priority="high",
    tags=["electronic_check", "payment"],
)

_ACTION_SECURITY_BUNDLE = RetentionAction(
    action_id="ACT_SECURITY_BUNDLE",
    title="Free Online Security & Backup Trial",
    description=(
        "Provide a 3-month free trial of Online Security and Online Backup services. "
        "Increases product stickiness and perceived value."
    ),
    category="service",
    impact_score=0.55,
    priority="high",
    tags=["no_online_security", "no_online_backup", "upsell"],
)

_ACTION_TECH_SUPPORT_BUNDLE = RetentionAction(
    action_id="ACT_TECH_SUPPORT_BUNDLE",
    title="Dedicated Tech Support Plan",
    description=(
        "Assign a dedicated support advisor with a 1-hour SLA. "
        "Add TechSupport + DeviceProtection at 50% off for 6 months."
    ),
    category="support",
    impact_score=0.60,
    priority="high",
    tags=["no_tech_support", "support"],
)

_ACTION_SENIOR_PLAN = RetentionAction(
    action_id="ACT_SENIOR_PLAN",
    title="Senior Citizen Care Package",
    description=(
        "Enroll the customer in the Senior Care Program: simplified billing, "
        "priority phone support, and a dedicated in-store assistant."
    ),
    category="support",
    impact_score=0.58,
    priority="high",
    tags=["senior_citizen", "support"],
)

_ACTION_STREAMING_BUNDLE = RetentionAction(
    action_id="ACT_STREAMING_BUNDLE",
    title="Premium Streaming Bundle Upgrade",
    description=(
        "Offer StreamingTV + StreamingMovies as a free add-on for 2 months "
        "to increase entertainment value and daily engagement."
    ),
    category="service",
    impact_score=0.40,
    priority="medium",
    tags=["no_streaming", "entertainment"],
)

_ACTION_FAMILY_PLAN = RetentionAction(
    action_id="ACT_FAMILY_PLAN",
    title="Family / Partner Plan Offer",
    description=(
        "Suggest adding a Partner or Dependent line at a discounted bundle rate. "
        "Multi-line customers have significantly lower churn rates."
    ),
    category="contract",
    impact_score=0.50,
    priority="medium",
    tags=["no_partner", "no_dependents", "family"],
)

_ACTION_FIBER_UPGRADE = RetentionAction(
    action_id="ACT_FIBER_UPGRADE",
    title="Fiber Optic Speed Upgrade",
    description=(
        "Offer a free speed upgrade to Fiber Optic Internet with no setup fee "
        "and a 30-day satisfaction guarantee."
    ),
    category="service",
    impact_score=0.35,
    priority="medium",
    tags=["dsl_internet", "upgrade"],
)

_ACTION_WINBACK_CALL = RetentionAction(
    action_id="ACT_WINBACK_CALL",
    title="Executive Win-Back Outreach",
    description=(
        "Escalate to customer success team for a personalised executive outreach call. "
        "Present a custom retention package tailored to usage patterns."
    ),
    category="engagement",
    impact_score=0.80,
    priority="critical",
    tags=["high_risk", "escalation"],
)

_ACTION_ENGAGEMENT_EMAIL = RetentionAction(
    action_id="ACT_ENGAGEMENT_EMAIL",
    title="Personalised Engagement Campaign",
    description=(
        "Enroll the customer in a personalised email journey highlighting "
        "underutilised features and value-adds they are paying for."
    ),
    category="engagement",
    impact_score=0.25,
    priority="low",
    tags=["low_risk", "engagement"],
)

_ACTION_PAPERLESS_INCENTIVE = RetentionAction(
    action_id="ACT_PAPERLESS_INCENTIVE",
    title="Paperless Billing Reward",
    description=(
        "Offer a $3/month credit for switching to paperless billing. "
        "Reduces billing-related service calls and friction."
    ),
    category="pricing",
    impact_score=0.20,
    priority="low",
    tags=["paper_billing"],
)


# ── RetentionEngine ───────────────────────────────────────────────────────────


class RetentionEngine:
    """
    Rule-based retention recommendation engine.

    Input:
      - customer_features : dict or pd.Series of raw/encoded feature values
      - churn_probability : float [0, 1]
      - shap_drivers      : list of top SHAP driver dicts from SHAPExplainer

    Output:
      {
        "customer_segment": "High Risk" | "Medium Risk" | "Low Risk",
        "churn_probability": float,
        "recommendations": [ RetentionAction.to_dict(), ... ],
        "estimated_retention_lift": float
      }
    """

    HIGH_THRESHOLD = PREVENTION["risk_thresholds"]["high"]
    MEDIUM_THRESHOLD = PREVENTION["risk_thresholds"]["medium"]
    MAX_RECOMMENDATIONS = PREVENTION["max_recommendations"]

    def __init__(self) -> None:
        logger.info(
            "RetentionEngine initialised | high=%.2f medium=%.2f max_recs=%d",
            self.HIGH_THRESHOLD,
            self.MEDIUM_THRESHOLD,
            self.MAX_RECOMMENDATIONS,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        customer_features: Dict[str, Any],
        churn_probability: float,
        shap_drivers: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Generate ranked retention recommendations for a customer.

        Parameters
        ----------
        customer_features : dict of feature_name → value (encoded or raw)
        churn_probability : model output probability of churning [0, 1]
        shap_drivers      : top SHAP drivers from SHAPExplainer.explain_instance()

        Returns
        -------
        Full recommendation dict (JSON-serializable)
        """
        segment = self._segment(churn_probability)
        shap_drivers = shap_drivers or []
        {d["feature"] for d in shap_drivers}

        candidate_actions: List[RetentionAction] = []

        # ── Segment-level base actions ─────────────────────────────
        if segment == "High Risk":
            candidate_actions.append(_ACTION_WINBACK_CALL)

        # ── Feature-driven rule matching ───────────────────────────
        fv = customer_features  # shorthand

        # Contract type
        if self._feature_matches(fv, "Contract_Month-to-month", 1):
            candidate_actions.append(_ACTION_CONTRACT_UPGRADE)

        # Payment method
        if self._feature_matches(fv, "PaymentMethod_Electronic check", 1):
            candidate_actions.append(_ACTION_AUTOPAY_INCENTIVE)

        # High monthly charges relative to service count
        monthly = fv.get("MonthlyCharges", 0)
        if isinstance(monthly, (int, float)) and monthly > 70:
            candidate_actions.append(_ACTION_LOYALTY_DISCOUNT)

        # Online security missing
        if self._feature_matches(fv, "OnlineSecurity", 0):
            candidate_actions.append(_ACTION_SECURITY_BUNDLE)

        # Tech support missing
        if self._feature_matches(fv, "TechSupport", 0):
            candidate_actions.append(_ACTION_TECH_SUPPORT_BUNDLE)

        # Senior citizen
        if self._feature_matches(fv, "SeniorCitizen", 1):
            candidate_actions.append(_ACTION_SENIOR_PLAN)

        # No streaming
        if self._feature_matches(fv, "StreamingTV", 0) and self._feature_matches(
            fv, "StreamingMovies", 0
        ):
            candidate_actions.append(_ACTION_STREAMING_BUNDLE)

        # No partner and no dependents
        if self._feature_matches(fv, "Partner", 0) and self._feature_matches(
            fv, "Dependents", 0
        ):
            candidate_actions.append(_ACTION_FAMILY_PLAN)

        # DSL internet service
        if self._feature_matches(fv, "InternetService_DSL", 1):
            candidate_actions.append(_ACTION_FIBER_UPGRADE)

        # Paper billing
        if self._feature_matches(fv, "PaperlessBilling", 0):
            candidate_actions.append(_ACTION_PAPERLESS_INCENTIVE)

        # Low risk baseline engagement
        if segment == "Low Risk":
            candidate_actions.append(_ACTION_ENGAGEMENT_EMAIL)

        # ── Boost actions whose tags match SHAP top drivers ────────
        top_driver_features = {
            d["feature"] for d in shap_drivers if d["shap_value"] > 0
        }
        for action in candidate_actions:
            if any(tag in " ".join(top_driver_features).lower() for tag in action.tags):
                action.impact_score = min(1.0, action.impact_score * 1.15)

        # ── Deduplicate, sort by impact, trim ─────────────────────
        seen: set = set()
        unique_actions: List[RetentionAction] = []
        for action in sorted(candidate_actions, key=lambda a: -a.impact_score):
            if action.action_id not in seen:
                seen.add(action.action_id)
                unique_actions.append(action)
            if len(unique_actions) >= self.MAX_RECOMMENDATIONS:
                break

        # Estimated retention lift = 1 - P(all actions fail)
        if unique_actions:
            retention_lift = 1.0
            for act in unique_actions:
                retention_lift *= 1.0 - act.impact_score
            retention_lift = round(1.0 - retention_lift, 4)
        else:
            retention_lift = 0.0

        result = {
            "customer_segment": segment,
            "churn_probability": round(churn_probability, 4),
            "recommendations": [a.to_dict() for a in unique_actions],
            "estimated_retention_lift": retention_lift,
        }

        logger.info(
            "Retention recommendation | segment=%s prob=%.3f actions=%d lift=%.3f",
            segment,
            churn_probability,
            len(unique_actions),
            retention_lift,
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _segment(self, prob: float) -> str:
        if prob >= self.HIGH_THRESHOLD:
            return "High Risk"
        if prob >= self.MEDIUM_THRESHOLD:
            return "Medium Risk"
        return "Low Risk"

    @staticmethod
    def _feature_matches(
        features: Dict[str, Any], feature_name: str, expected_value: Any
    ) -> bool:
        """Check if a feature equals an expected encoded value."""
        val = features.get(feature_name)
        if val is None:
            return False
        try:
            return float(val) == float(expected_value)
        except (TypeError, ValueError):
            return str(val) == str(expected_value)
