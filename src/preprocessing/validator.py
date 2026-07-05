import re
from typing import Dict, Any, Tuple, Optional
from src.preprocessing.exceptions import SchemaValidationError, CustomRuleValidationError

try:
    import jsonschema
    _has_jsonschema = True
except ImportError:
    _has_jsonschema = False

class CandidateSchemaValidator:
    """Validates candidate record structural schemas and custom business rules."""
    
    CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
    DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def __init__(self, schema_dict: Optional[Dict[str, Any]] = None):
        self.schema_dict = schema_dict

    def validate_candidate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Performs full validation (schema check + custom rules) on a candidate record.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # 1. Structural schema validation
        try:
            if _has_jsonschema and self.schema_dict:
                jsonschema.validate(instance=data, schema=self.schema_dict)
            else:
                self._fallback_schema_validate(data)
        except Exception as e:
            return False, f"Schema validation error: {str(e)}"
            
        # 2. Custom semantic/domain rules validation
        try:
            self.validate_custom_rules(data)
        except CustomRuleValidationError as e:
            return False, f"Custom rule validation error: {str(e)}"
            
        return True, None

    def validate_custom_rules(self, data: Dict[str, Any]) -> None:
        """Enforces custom domain rules (e.g., date chronological ordering, signal value limits)."""
        # Career History validation
        career_history = data.get("career_history", [])
        for idx, job in enumerate(career_history):
            start_date = job.get("start_date")
            end_date = job.get("end_date")
            
            if start_date and not self.DATE_PATTERN.match(start_date):
                raise CustomRuleValidationError(f"career_history[{idx}] has invalid start_date format: {start_date}")
            if end_date and not self.DATE_PATTERN.match(end_date):
                raise CustomRuleValidationError(f"career_history[{idx}] has invalid end_date format: {end_date}")
                
            if start_date and end_date and start_date > end_date:
                raise CustomRuleValidationError(
                    f"career_history[{idx}] start_date '{start_date}' is after end_date '{end_date}'."
                )

        # Education validation
        education = data.get("education", [])
        for idx, edu in enumerate(education):
            start_year = edu.get("start_year")
            end_year = edu.get("end_year")
            if start_year and end_year and start_year > end_year:
                raise CustomRuleValidationError(
                    f"education[{idx}] start_year '{start_year}' is after end_year '{end_year}'."
                )

        # Redrob Signals validation
        signals = data.get("redrob_signals", {})
        if signals:
            completeness = signals.get("profile_completeness_score")
            if completeness is not None and not (0 <= completeness <= 100):
                raise CustomRuleValidationError(
                    f"redrob_signals.profile_completeness_score must be between 0 and 100, got: {completeness}"
                )
                
            response_rate = signals.get("recruiter_response_rate")
            if response_rate is not None and not (0 <= response_rate <= 1.0):
                raise CustomRuleValidationError(
                    f"redrob_signals.recruiter_response_rate must be between 0.0 and 1.0, got: {response_rate}"
                )

            salary = signals.get("expected_salary_range_inr_lpa")
            if salary and isinstance(salary, dict):
                salary_min = salary.get("min")
                salary_max = salary.get("max")
                if salary_min is not None and salary_max is not None and salary_min > salary_max:
                    raise CustomRuleValidationError(
                        f"redrob_signals expected salary min ({salary_min}) is greater than max ({salary_max})."
                    )

    def _fallback_schema_validate(self, data: Dict[str, Any]) -> None:
        """Fallback validation checking required fields and basic types without jsonschema module."""
        if not isinstance(data, dict):
            raise SchemaValidationError("Candidate record must be a JSON object/dict.")
            
        required_top_level = ["candidate_id", "profile", "career_history", "education", "skills", "redrob_signals"]
        for field in required_top_level:
            if field not in data:
                raise SchemaValidationError(f"Missing required top-level field: {field}")
                
        cid = data["candidate_id"]
        if not isinstance(cid, str) or not self.CANDIDATE_ID_PATTERN.match(cid):
            raise SchemaValidationError(f"Invalid candidate_id format: {cid}")
            
        # Validate profile
        profile = data["profile"]
        if not isinstance(profile, dict):
            raise SchemaValidationError("profile field must be a dictionary.")
            
        required_profile = [
            "anonymized_name", "headline", "summary", "location", "country",
            "years_of_experience", "current_title", "current_company",
            "current_company_size", "current_industry"
        ]
        for field in required_profile:
            if field not in profile:
                raise SchemaValidationError(f"Missing required profile field: {field}")
                
        if not isinstance(profile["years_of_experience"], (int, float)):
            raise SchemaValidationError("profile.years_of_experience must be a number.")
            
        # Validate career history
        career = data["career_history"]
        if not isinstance(career, list):
            raise SchemaValidationError("career_history must be an array/list.")
        if len(career) < 1:
            raise SchemaValidationError("career_history must contain at least 1 record.")
            
        required_job = [
            "company", "title", "start_date", "end_date", "duration_months",
            "is_current", "industry", "company_size", "description"
        ]
        for idx, job in enumerate(career):
            if not isinstance(job, dict):
                raise SchemaValidationError(f"career_history[{idx}] must be a dictionary.")
            for field in required_job:
                if field not in job:
                    raise SchemaValidationError(f"Missing required field in career_history[{idx}]: {field}")
                    
        # Validate education
        edu = data["education"]
        if not isinstance(edu, list):
            raise SchemaValidationError("education must be an array/list.")
            
        # Validate skills
        skills = data["skills"]
        if not isinstance(skills, list):
            raise SchemaValidationError("skills must be an array/list.")
            
        # Validate redrob signals
        signals = data["redrob_signals"]
        if not isinstance(signals, dict):
            raise SchemaValidationError("redrob_signals must be a dictionary.")
