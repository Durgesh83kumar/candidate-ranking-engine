from typing import Dict, Any

class ReasoningEngine:
    """Generates structured, grammatically correct, deterministic candidate reasoning transcripts."""

    def generate_reasoning(self, candidate_record: Dict[str, Any]) -> str:
        """Assembles 1-2 sentences of explanation based on evidence.
        
        Args:
            candidate_record: Rich candidate dictionary containing explainability and timeline details.
            
        Returns:
            str: Reasoning text (10-50 words).
        """
        evidence = candidate_record.get("explainability_evidence", {})
        skills = [s.lower() for s in candidate_record.get("matched_profile_sections", [])]
        exp = float(candidate_record.get("years_of_experience", 0.0))
        
        # 1. Compile Sentence 1: Tech and Domain alignment
        has_ai = evidence.get("Matched AI Experience", False) or "matched summary" in skills
        has_retrieval = evidence.get("Matched Search/Retrieval Experience", False)
        
        if has_ai and has_retrieval:
            s1 = "Strong experience building large language models and search/retrieval systems."
        elif has_ai:
            s1 = "Demonstrated expertise building large language models and neural networks."
        elif has_retrieval:
            s1 = "Strong background in search indexing, information retrieval, and python databases."
        else:
            s1 = "Solid technical background in python software development and systems engineering."
            
        # 2. Compile Sentence 2: Leadership, Stability, or Context
        has_lead = evidence.get("Matched Leadership", False)
        has_prod = evidence.get("Matched Production Experience", False)
        
        if has_lead and exp >= 5.0:
            s2 = "Demonstrates stable career progression with proven leadership mentoring software engineering teams."
        elif has_prod:
            s2 = "Demonstrates a proven track record of shipping production-grade applications at scale."
        elif exp >= 4.0:
            s2 = "Aligned with core career experience targets and software delivery practices."
        else:
            s2 = "Shows alignment to core engineering requirements and candidate profile specifications."
            
        reasoning = f"{s1} {s2}"
        
        # Enforce strict length constraints
        words = reasoning.split()
        word_count = len(words)
        
        if word_count < 10 or word_count > 50:
            # Safe default fitting exactly 20 words
            reasoning = (
                "Proven experience in software engineering and technical system design. "
                "High alignment to core python development and skills requirements."
            )
            
        return reasoning
