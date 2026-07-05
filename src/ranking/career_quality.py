from typing import Dict, Any

class CareerQualityEvaluator:
    """Computes career progression, stability, and domain depth metrics in the [0, 1] range."""

    def evaluate(self, candidate_record: Dict[str, Any]) -> float:
        """Calculates a career score based on available preprocessed timeline features.
        
        Signals evaluated:
        - career_progression_score
        - job_hopping_score (converted to stability)
        - years_of_relevant_ai_experience
        - technical_depth_score
        
        Args:
            candidate_record: Rich candidate dictionary containing profile key.
            
        Returns:
            float: Normalized career score in [0.0, 1.0].
        """
        profile = candidate_record.get("profile", {})
        if not profile:
            # Handle flat candidate dictionary from prior parquet loads
            profile = candidate_record
            
        # 1. Progression Score
        prog_raw = profile.get("career_progression_score", 50.0)
        s_progression = float(prog_raw) / 100.0
        
        # 2. Stability Score (1.0 - hopping_ratio)
        hopping_raw = profile.get("job_hopping_score", 30.0)
        # Assuming higher job hopping is worse, compute stability
        s_stability = max(0.0, min(1.0, (100.0 - float(hopping_raw)) / 100.0))
        
        # 3. AI Relevance Score (capped at 8 ideal years)
        ai_exp_raw = profile.get("years_of_relevant_ai_experience", 0.0)
        s_ai_relevance = min(1.0, float(ai_exp_raw) / 8.0) if ai_exp_raw else 0.0
        
        # 4. Technical Depth Score
        depth_raw = profile.get("technical_depth_score", 50.0)
        s_tech_depth = float(depth_raw) / 100.0
        
        # Weighted Aggregation
        w_prog, w_stab, w_ai, w_depth = 0.30, 0.30, 0.20, 0.20
        score = (w_prog * s_progression) + (w_stab * s_stability) + (w_ai * s_ai_relevance) + (w_depth * s_tech_depth)
        
        return max(0.0, min(1.0, float(score)))
