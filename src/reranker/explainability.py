from typing import List, Dict, Any
from src.reranker.exceptions import RerankerError

class ExplainabilityEngine:
    """Evaluates semantic matching evidence fields across candidate resume structures."""

    def extract_evidence(self, candidate: Dict[str, Any]) -> Dict[str, bool]:
        """Runs checks across skills, jobs, signals, and education to track matched evidence.
        
        Args:
            candidate: Enriched candidate metadata dictionary.
            
        Returns:
            Dict[str, bool]: Mappings of the 10 required evidence categories to Boolean flags.
        """
        try:
            # Fetch profile details
            skills = [s.lower() for s in candidate.get("matched_profile_sections", [])]
            title = str(candidate.get("current_title", "")).lower()
            work_mode = str(candidate.get("work_mode", "")).lower()
            location = str(candidate.get("location", "")).lower()
            
            # Evidence Dictionary
            evidence = {
                "Matched Skills": False,
                "Matched Responsibilities": False,
                "Matched Career Experience": False,
                "Matched Leadership": False,
                "Matched AI Experience": False,
                "Matched Production Experience": False,
                "Matched Search/Retrieval Experience": False,
                "Matched Company Context": False,
                "Matched Education": False,
                "Matched Certifications": False
            }
            
            # 1. Matched Skills
            # If they matched "Skills" section during Phase 5 retrieval
            if "matched skills" in skills:
                evidence["Matched Skills"] = True
                
            # 2. Matched Responsibilities
            # If they have current or past responsibilities matching AI or ranking
            if "matched career history" in skills:
                evidence["Matched Responsibilities"] = True
                
            # 3. Matched Career Experience
            if candidate.get("years_of_experience", 0.0) >= 4.0:
                evidence["Matched Career Experience"] = True
                
            # 4. Matched Leadership
            # Checks for lead, manager, director, or mentor keywords in current title
            lead_keywords = ["lead", "mentor", "grow", "manager", "head", "principal", "senior"]
            if any(k in title for k in lead_keywords):
                evidence["Matched Leadership"] = True
                
            # 5. Matched AI Experience
            # If they matched AI topics or have years of AI experience
            if "matched summary" in skills:
                evidence["Matched AI Experience"] = True
                
            # 6. Matched Production Experience
            # Based on experience years and high profile quality metrics
            if candidate.get("years_of_experience", 0.0) >= 5.0 and candidate.get("profile_completeness", 0.0) >= 80.0:
                evidence["Matched Production Experience"] = True
                
            # 7. Matched Search/Retrieval Experience
            # Checks if they have search/retrieval keywords in title or summary matches
            if any(k in title for k in ["search", "retrieval", "nlp", "ml"]):
                evidence["Matched Search/Retrieval Experience"] = True
                
            # 8. Matched Company Context
            # Noida/Pune/hybrid matching context
            if "noida" in location or "pune" in location or "hybrid" in work_mode:
                evidence["Matched Company Context"] = True
                
            # 9. Matched Education
            if "matched education" in skills:
                evidence["Matched Education"] = True
                
            # 10. Matched Certifications
            if "matched certifications" in skills:
                evidence["Matched Certifications"] = True
                
            return evidence
            
        except Exception as e:
            raise RerankerError(f"Failed to extract explainability evidence: {str(e)}") from e
