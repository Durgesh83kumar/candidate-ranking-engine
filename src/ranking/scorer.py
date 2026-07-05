from typing import List, Dict, Any, Tuple
from src.ranking.config import RankingConfig
from src.ranking.career_quality import CareerQualityEvaluator
from src.ranking.profile_quality import ProfileQualityEvaluator
from src.ranking.business_rules import BusinessRulesEngine
from src.ranking.honeypot import HoneypotDetector
from src.ranking.exceptions import ScorerError

class HybridScorer:
    """Computes the final Unified Hybrid Score (UHS) combining dense retrieval, CE matching, quality, and penalties."""

    def __init__(self, config: RankingConfig):
        self.config = config
        self.career_eval = CareerQualityEvaluator()
        self.profile_eval = ProfileQualityEvaluator()
        self.business_engine = BusinessRulesEngine(config)
        self.honeypot_detector = HoneypotDetector(config.honeypot_multipliers)

    def compute_hybrid_score(self, candidate_record: Dict[str, Any], max_rrf_score: float = 0.10) -> Dict[str, Any]:
        """Calculates Unified Hybrid Score and preserves components breakdown.
        
        Args:
            candidate_record: Candidate dictionary containing retrieval, CE, and profile metrics.
            max_rrf_score: Maximum RRF score observed in the pool (used to scale RRF to [0.0, 1.0]).
            
        Returns:
            Dict[str, Any]: Unified scores and triggered rules metadata.
        """
        try:
            # 1. Retrieval Score (scaled RRF score)
            rrf_score = float(candidate_record.get("rrf_score", 0.0))
            s_retrieval = min(1.0, rrf_score / max_rrf_score) if max_rrf_score > 0 else rrf_score
            
            # 2. Cross Encoder Score
            s_ce = float(candidate_record.get("cross_encoder_probability", 0.0))
            
            # 3. Career Quality Score
            s_career = self.career_eval.evaluate(candidate_record)
            
            # 4. Profile Quality Score
            s_profile = self.profile_eval.evaluate(candidate_record)
            
            # 5. Business Multiplier Penalty
            p_business, triggered_biz = self.business_engine.evaluate(candidate_record)
            
            # 6. Honeypot Multiplier Penalty
            p_honeypot, triggered_honeypot = self.honeypot_detector.scan_candidate(candidate_record)
            
            # Compute Weighted base score
            base_score = (
                (self.config.weight_retrieval * s_retrieval) +
                (self.config.weight_cross_encoder * s_ce) +
                (self.config.weight_career * s_career) +
                (self.config.weight_profile * s_profile)
            )
            
            # Apply Multipliers
            final_score = base_score * p_business * p_honeypot
            
            return {
                "candidate_id": candidate_record["candidate_id"],
                "retrieval_score": round(s_retrieval, 4),
                "cross_encoder_score": round(s_ce, 4),
                "career_score": round(s_career, 4),
                "profile_score": round(s_profile, 4),
                "business_multiplier": round(p_business, 4),
                "honeypot_multiplier": round(p_honeypot, 4),
                "final_score": round(float(final_score), 4),
                "triggered_business_rules": triggered_biz,
                "triggered_honeypot_checks": triggered_honeypot
            }
            
        except Exception as e:
            raise ScorerError(f"Scoring algorithm failure for candidate {candidate_record.get('candidate_id')}: {str(e)}") from e
ClassScorer = HybridScorer
