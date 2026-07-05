from typing import Dict, Any, List, Tuple
from src.ranking.config import RankingConfig

class BusinessRulesEngine:
    """Applies lightweight recruiting filters and calculates penalty coefficients."""

    def __init__(self, config: RankingConfig):
        self.config = config

    def evaluate(self, candidate_record: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Evaluates business rules for location match, notice periods, and salary caps.
        
        Args:
            candidate_record: Raw candidate profile dictionary.
            
        Returns:
            Tuple[float, List[str]]: Score multiplier factor and list of triggered warnings.
        """
        multiplier = 1.0
        warnings = []
        
        # 1. Location and Relocation Check
        location = str(candidate_record.get("location", "")).lower()
        country = str(candidate_record.get("country", "")).lower()
        relocate = bool(candidate_record.get("relocation", False))
        
        is_local = "noida" in location or "pune" in location
        is_india = "india" in country or is_local
        
        if not is_india:
            multiplier *= self.config.penalty_location_mismatch
            warnings.append(f"Out of Country ({country})")
        elif not is_local and not relocate:
            multiplier *= self.config.penalty_location_mismatch
            warnings.append("Location Mismatch (Not Noida/Pune & Unwilling to relocate)")
            
        # 2. Notice Period Check
        notice = int(candidate_record.get("notice_period_days", 90))
        if notice > self.config.max_notice_days:
            multiplier *= self.config.penalty_notice_period_exceeded
            warnings.append(f"Notice Period Exceeded ({notice} days)")
            
        # 3. Salary expectation ceiling check
        salary = candidate_record.get("expected_salary_lpa")
        if salary is not None and float(salary) > self.config.salary_ceiling_lpa:
            # Apply sliding penalty
            sal_diff = float(salary) - self.config.salary_ceiling_lpa
            sal_multiplier = max(0.90, 1.0 - (sal_diff * 0.005))
            multiplier *= sal_multiplier
            warnings.append(f"Salary Ceiling Exceeded ({salary:.1f} LPA)")
            
        # 4. Work Mode Check
        work_mode = str(candidate_record.get("work_mode", "")).lower()
        if "remote" in work_mode and not relocate:
            # Company prefers hybrid Noida/Pune
            multiplier *= 0.98
            warnings.append("Work Mode Mismatch (Remote preferred but JD wants hybrid/office)")

        return max(0.50, min(1.0, float(multiplier))), warnings
