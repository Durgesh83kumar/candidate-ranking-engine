import re
from typing import Dict, Any, Tuple
from src.verification.config import VerificationConfig
from src.verification.exceptions import PhraseScannerError

class PhraseScanner:
    """Uses regular expressions to distinguish actual engineering experience from recruiting/management roles."""

    def __init__(self, config: VerificationConfig):
        self.config = config

    def scan(self, search_doc: str) -> Tuple[float, bool, bool]:
        """Scans candidate search document text for recruiter or engineering cues.
        
        Args:
            search_doc: Candidate search document text string (v2).
            
        Returns:
            Tuple[float, bool, bool]: Multiplier coefficient, has_recruiter_cues, has_engineering_cues.
        """
        if not search_doc:
            return 1.0, False, False
            
        try:
            has_recruiter = bool(self.config.recruiter_pattern.search(search_doc))
            has_engineering = bool(self.config.engineering_pattern.search(search_doc))
            
            multiplier = 1.0
            
            # Apply recruiter penalty
            if has_recruiter:
                multiplier *= self.config.penalty_recruiter
                
            # Apply engineering boost
            if has_engineering:
                multiplier *= self.config.boost_engineering
                
            return float(multiplier), has_recruiter, has_engineering
            
        except Exception as e:
            raise PhraseScannerError(f"Regex phrase scanning failure: {str(e)}") from e
