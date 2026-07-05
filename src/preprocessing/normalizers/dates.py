from datetime import date, datetime
from typing import Dict, Any, Optional
from src.preprocessing.exceptions import DateNormalizationError
from src.preprocessing.normalizers.base import BaseNormalizer

class DateNormalizer(BaseNormalizer):
    """Parses date fields and handles current/null dates with a pipeline reference date."""
    
    def __init__(self, reference_date_str: str = "2026-06-30"):
        self.reference_date_str = reference_date_str
        try:
            self.reference_date = date.fromisoformat(reference_date_str)
        except ValueError as e:
            raise DateNormalizationError(f"Invalid reference date format: {reference_date_str}") from e

    def parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parses an ISO format date string into a datetime.date object. Handles nulls."""
        if not date_str:
            return None
            
        try:
            # First try ISO format
            return date.fromisoformat(date_str)
        except ValueError:
            # Fallback to general parsing if string formats vary slightly
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.date()
            except ValueError as e:
                raise DateNormalizationError(f"Could not parse date string: {date_str}") from e

    def normalize(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes dates within candidate profile (career history and redrob signals)."""
        # Career History Date Normalization
        for idx, job in enumerate(candidate_data.get("career_history", [])):
            start_str = job.get("start_date")
            end_str = job.get("end_date")
            
            try:
                start_dt = self.parse_date(start_str)
                if not start_dt:
                    raise DateNormalizationError(f"career_history[{idx}] has missing start_date.")
                job["start_date_parsed"] = start_dt
                job["start_date"] = start_dt.isoformat()
            except DateNormalizationError as e:
                raise DateNormalizationError(f"Error parsing job start_date: {str(e)}") from e
                
            try:
                if job.get("is_current") is True or end_str is None:
                    job["end_date_parsed"] = self.reference_date
                    job["end_date"] = self.reference_date_str
                    job["is_current"] = True
                else:
                    end_dt = self.parse_date(end_str)
                    if end_dt:
                        job["end_date_parsed"] = end_dt
                        job["end_date"] = end_dt.isoformat()
                        job["is_current"] = False
                    else:
                        job["end_date_parsed"] = self.reference_date
                        job["end_date"] = self.reference_date_str
                        job["is_current"] = True
            except DateNormalizationError as e:
                raise DateNormalizationError(f"Error parsing job end_date: {str(e)}") from e
                
            # Chronological date sequence swap if start_date is after end_date
            if job.get("start_date_parsed") and job.get("end_date_parsed"):
                if job["start_date_parsed"] > job["end_date_parsed"]:
                    job["start_date_parsed"], job["end_date_parsed"] = job["end_date_parsed"], job["start_date_parsed"]
                    job["start_date"] = job["start_date_parsed"].isoformat()
                    job["end_date"] = job["end_date_parsed"].isoformat()

        # Redrob Signals Date Normalization
        signals = candidate_data.get("redrob_signals", {})
        if signals:
            for field in ["signup_date", "last_active_date"]:
                date_val = signals.get(field)
                if date_val:
                    try:
                        parsed = self.parse_date(date_val)
                        if parsed:
                            signals[f"{field}_parsed"] = parsed
                            signals[field] = parsed.isoformat()
                    except DateNormalizationError as e:
                        raise DateNormalizationError(f"Error parsing signal field {field}: {str(e)}") from e
                        
            # Calculate days since active relative to reference date
            last_active = signals.get("last_active_date_parsed")
            if last_active:
                delta = self.reference_date - last_active
                signals["days_since_active"] = max(0, delta.days)
            else:
                signals["days_since_active"] = 365 # Default fallback to a high inactive indicator

        return candidate_data
