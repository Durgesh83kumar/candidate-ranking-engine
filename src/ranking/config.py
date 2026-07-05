from typing import Dict, Any

class RankingConfig:
    """Manages the scoring weights and penalty coefficients for the Hybrid Ranking Engine."""

    def __init__(self, **kwargs):
        # Default Weights (must sum to 1.0)
        self.weight_retrieval = float(kwargs.get("weight_retrieval", 0.20))
        self.weight_cross_encoder = float(kwargs.get("weight_cross_encoder", 0.40))
        self.weight_career = float(kwargs.get("weight_career", 0.25))
        self.weight_profile = float(kwargs.get("weight_profile", 0.15))

        # Business Rule Penalties
        self.penalty_location_mismatch = float(kwargs.get("penalty_location_mismatch", 0.90))
        self.penalty_notice_period_exceeded = float(kwargs.get("penalty_notice_period_exceeded", 0.95))
        self.penalty_salary_expectation_exceeded = float(kwargs.get("penalty_salary_expectation_exceeded", 0.97))
        self.salary_ceiling_lpa = float(kwargs.get("salary_ceiling_lpa", 50.0))
        self.max_notice_days = int(kwargs.get("max_notice_days", 30))

        # Honeypot Penalties
        self.honeypot_multipliers = {
            0: 1.00,
            1: 0.90,
            2: 0.70,
            3: 0.10
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_retrieval": self.weight_retrieval,
            "weight_cross_encoder": self.weight_cross_encoder,
            "weight_career": self.weight_career,
            "weight_profile": self.weight_profile,
            "penalty_location_mismatch": self.penalty_location_mismatch,
            "penalty_notice_period_exceeded": self.penalty_notice_period_exceeded,
            "penalty_salary_expectation_exceeded": self.penalty_salary_expectation_exceeded,
            "salary_ceiling_lpa": self.salary_ceiling_lpa,
            "max_notice_days": self.max_notice_days,
            "honeypot_multipliers": self.honeypot_multipliers
        }
