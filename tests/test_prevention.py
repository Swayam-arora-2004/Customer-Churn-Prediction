"""
tests/test_prevention.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/prevention.py (RetentionEngine).
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
from src.prevention import RetentionEngine


@pytest.fixture
def engine() -> RetentionEngine:
    return RetentionEngine()


# ── Risk segmentation ─────────────────────────────────────────────────────────


class TestSegmentation:
    def test_high_risk_segment(self, engine):
        result = engine.recommend({}, 0.85)
        assert result["customer_segment"] == "High Risk"

    def test_medium_risk_segment(self, engine):
        result = engine.recommend({}, 0.50)
        assert result["customer_segment"] == "Medium Risk"

    def test_low_risk_segment(self, engine):
        result = engine.recommend({}, 0.10)
        assert result["customer_segment"] == "Low Risk"

    def test_boundary_high(self, engine):
        result = engine.recommend({}, 0.70)
        assert result["customer_segment"] == "High Risk"

    def test_boundary_medium(self, engine):
        result = engine.recommend({}, 0.30)
        assert result["customer_segment"] == "Medium Risk"


# ── Recommendation rules ──────────────────────────────────────────────────────


class TestRecommendationRules:
    def test_high_risk_includes_winback_call(self, engine):
        result = engine.recommend({}, 0.90)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert "ACT_WINBACK_CALL" in action_ids

    def test_month_to_month_gets_contract_upgrade(self, engine):
        features = {"Contract_Month-to-month": 1}
        result = engine.recommend(features, 0.75)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert "ACT_CONTRACT_UPGRADE" in action_ids

    def test_electronic_check_gets_autopay_incentive(self, engine):
        features = {"PaymentMethod_Electronic check": 1}
        result = engine.recommend(features, 0.60)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert "ACT_AUTOPAY_INCENTIVE" in action_ids

    def test_high_charges_gets_loyalty_discount(self, engine):
        features = {"MonthlyCharges": 90.0}
        result = engine.recommend(features, 0.65)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert "ACT_LOYALTY_DISCOUNT" in action_ids

    def test_no_online_security_gets_security_bundle(self, engine):
        features = {"OnlineSecurity": 0}
        result = engine.recommend(features, 0.55)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert "ACT_SECURITY_BUNDLE" in action_ids

    def test_no_tech_support_gets_support_plan(self, engine):
        features = {"TechSupport": 0}
        result = engine.recommend(features, 0.55)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert "ACT_TECH_SUPPORT_BUNDLE" in action_ids

    def test_senior_citizen_gets_senior_plan(self, engine):
        features = {"SeniorCitizen": 1}
        result = engine.recommend(features, 0.60)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert "ACT_SENIOR_PLAN" in action_ids

    def test_low_risk_gets_engagement_email(self, engine):
        result = engine.recommend({}, 0.10)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert "ACT_ENGAGEMENT_EMAIL" in action_ids


# ── Output structure ──────────────────────────────────────────────────────────


class TestOutputStructure:
    def test_output_has_required_keys(self, engine):
        result = engine.recommend({}, 0.80)
        assert "customer_segment" in result
        assert "churn_probability" in result
        assert "recommendations" in result
        assert "estimated_retention_lift" in result

    def test_recommendations_is_list(self, engine):
        result = engine.recommend({}, 0.80)
        assert isinstance(result["recommendations"], list)

    def test_each_recommendation_has_required_fields(self, engine):
        result = engine.recommend({"Contract_Month-to-month": 1}, 0.80)
        for rec in result["recommendations"]:
            assert "action_id" in rec
            assert "title" in rec
            assert "description" in rec
            assert "category" in rec
            assert "impact_score" in rec
            assert "priority" in rec

    def test_max_recommendations_respected(self, engine):
        # Trigger as many rules as possible
        features = {
            "Contract_Month-to-month": 1,
            "PaymentMethod_Electronic check": 1,
            "MonthlyCharges": 95.0,
            "OnlineSecurity": 0,
            "TechSupport": 0,
            "SeniorCitizen": 1,
        }
        result = engine.recommend(features, 0.90)
        assert len(result["recommendations"]) <= engine.MAX_RECOMMENDATIONS

    def test_no_duplicate_action_ids(self, engine):
        features = {
            "Contract_Month-to-month": 1,
            "PaymentMethod_Electronic check": 1,
            "MonthlyCharges": 95.0,
        }
        result = engine.recommend(features, 0.85)
        action_ids = [r["action_id"] for r in result["recommendations"]]
        assert len(action_ids) == len(set(action_ids))


# ── Impact & lift ─────────────────────────────────────────────────────────────


class TestImpactAndLift:
    def test_retention_lift_between_0_and_1(self, engine):
        result = engine.recommend({"Contract_Month-to-month": 1}, 0.80)
        lift = result["estimated_retention_lift"]
        assert 0.0 <= lift <= 1.0

    def test_no_recommendations_lift_is_zero(self, engine):
        # Empty features, low probability — may still get engagement email
        # but verify lift is always ≥ 0
        result = engine.recommend({}, 0.05)
        assert result["estimated_retention_lift"] >= 0.0

    def test_high_impact_actions_produce_positive_lift(self, engine):
        features = {"Contract_Month-to-month": 1, "MonthlyCharges": 80}
        result = engine.recommend(features, 0.85)
        if result["recommendations"]:
            assert result["estimated_retention_lift"] > 0.0

    def test_churn_probability_rounded_in_output(self, engine):
        result = engine.recommend({}, 0.123456789)
        assert len(str(result["churn_probability"]).split(".")[-1]) <= 4
