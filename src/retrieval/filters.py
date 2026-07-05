from typing import Dict, Any, List, Tuple
from src.retrieval.exceptions import FilterError

class SoftFiltersEvaluator:
    """Evaluates recruiter preferences as optional soft filters, adjusting confidence weights."""

    def __init__(self, filter_prefs: Dict[str, Any]):
        self.prefs = filter_prefs

    def evaluate_candidate(self, candidate: Dict[str, Any]) -> Tuple[List[str], float]:
        """Checks soft constraints. Returns a list of failed constraint names and a penalty multiplier.
        
        Recruiter preferences parsed:
        - min_experience: float
        - preferred_country: str (or List[str])
        - preferred_city: str (or List[str])
        - salary_max_lpa: float
        - max_notice_period_days: int
        - relocation_required: bool
        - preferred_work_modes: List[str]
        
        Args:
            candidate: Dictionary representing candidate metadata.
            
        Returns:
            Tuple[List[str], float]: Failed soft constraints list, and confidence penalty multiplier.
        """
        failed_filters = []
        penalty = 1.0
        
        try:
            # 1. Experience Check
            min_exp = self.prefs.get("min_experience")
            if min_exp is not None:
                cand_exp = candidate.get("years_of_experience", 0.0)
                if cand_exp < float(min_exp):
                    failed_filters.append("Minimum Experience")
                    # Scale penalty depending on experience gap
                    gap = float(min_exp) - cand_exp
                    penalty *= max(0.7, 1.0 - (gap * 0.1))
                    
            # 2. Preferred Country Check
            pref_country = self.prefs.get("preferred_country")
            if pref_country:
                cand_country = str(candidate.get("country", "")).lower().strip()
                countries = [str(c).lower().strip() for c in (pref_country if isinstance(pref_country, list) else [pref_country])]
                if cand_country not in countries:
                    failed_filters.append("Preferred Country")
                    penalty *= 0.85
                    
            # 3. Preferred City Check
            pref_city = self.prefs.get("preferred_city")
            if pref_city:
                cand_city = str(candidate.get("location", "")).lower().strip()
                cities = [str(c).lower().strip() for c in (pref_city if isinstance(pref_city, list) else [pref_city])]
                if not any(city in cand_city for city in cities):
                    failed_filters.append("Preferred City")
                    penalty *= 0.90
                    
            # 4. Expected Salary Check
            salary_max = self.prefs.get("salary_max_lpa")
            if salary_max is not None:
                cand_sal = candidate.get("expected_salary_lpa")
                if cand_sal is not None and cand_sal > float(salary_max):
                    failed_filters.append("Salary Cap")
                    penalty *= 0.80
                    
            # 5. Notice Period Check
            max_notice = self.prefs.get("max_notice_period_days")
            if max_notice is not None:
                cand_notice = candidate.get("notice_period_days", 90)
                if cand_notice > int(max_notice):
                    failed_filters.append("Notice Period")
                    penalty *= 0.85
                    
            # 6. Relocation Check
            reloc_req = self.prefs.get("relocation_required", False)
            if reloc_req:
                cand_reloc = candidate.get("relocation", False)
                if not cand_reloc:
                    failed_filters.append("Relocation Status")
                    penalty *= 0.90
                    
            # 7. Work Mode Check
            work_modes = self.prefs.get("preferred_work_modes")
            if work_modes:
                cand_mode = str(candidate.get("work_mode", "")).lower().strip()
                modes = [str(m).lower().strip() for m in (work_modes if isinstance(work_modes, list) else [work_modes])]
                if cand_mode not in modes:
                    failed_filters.append("Work Mode")
                    penalty *= 0.95

            # 8. Mandatory exclusions hard elimination
            # If the candidate matches the negative preferences, we mark it as an exclusion
            if candidate.get("is_excluded", False):
                failed_filters.append("Mandatory Exclusions (Exclusion Query Match)")
                penalty = 0.0  # Zero score triggers exclusion from top candidates or deprioritizes
                
            return failed_filters, penalty
            
        except Exception as e:
            raise FilterError(f"Failed to evaluate soft constraints: {str(e)}") from e
