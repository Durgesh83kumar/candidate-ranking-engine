from typing import Dict, Any, List, Tuple
from src.verification.config import VerificationConfig
from src.verification.exceptions import AISpecialistError

class AISpecialist:
    """Verifies that candidates claiming AI/ML specialization actually possess real framework tools."""

    def __init__(self, config: VerificationConfig):
        self.config = config

    def check_specialist(self, candidate_skills: List[Any], search_doc: str, claims_ai_experience: bool) -> Tuple[float, List[str]]:
        """Verifies if candidate contains any true AI engineering frameworks.
        
        Args:
            candidate_skills: Standardized list of skills (dictionaries or strings).
            search_doc: Candidate search document text string (v2).
            claims_ai_experience: True if candidate metadata/scores indicate AI/ML career focus.
            
        Returns:
            Tuple[float, List[str]]: Multiplier coefficient and list of matched frameworks.
        """
        try:
            cand_skills_set = set()
            if candidate_skills:
                for s in candidate_skills:
                    if isinstance(s, dict):
                        name = s.get("name_normalized") or s.get("name_raw") or ""
                        cand_skills_set.add(str(name).lower().strip())
                    else:
                        cand_skills_set.add(str(s).lower().strip())
                        
            doc_lower = str(search_doc).lower()
            
            matched_frameworks = []
            for fw in self.config.ai_frameworks:
                if fw in cand_skills_set or fw in doc_lower:
                    matched_frameworks.append(fw)
                    
            # If the candidate claims AI experience but has ZERO verified frameworks
            if claims_ai_experience and not matched_frameworks:
                return self.config.penalty_ai_frameworks_missing, []
                
            return 1.0, matched_frameworks
            
        except Exception as e:
            raise AISpecialistError(f"AI specialization verification failure: {str(e)}") from e
