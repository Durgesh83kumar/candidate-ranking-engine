import pandas as pd
from typing import List, Dict, Any
from src.reranker.exceptions import RerankerError

class FeatureEngineeringManager:
    """Generates rich semantic and structural ranking features for downstream Hybrid Rankers."""

    def compile_features(self, reranked_candidates: List[Dict[str, Any]]) -> pd.DataFrame:
        """Assembles a structured feature dataset from candidate statistics and search matches.
        
        Features compiled:
        - candidate_id: str
        - cross_encoder_score: float (probability)
        - query_coverage: float (matched queries fraction)
        - matched_skills_count: int
        - matched_responsibilities_count: int
        - matched_experience_count: float (years of experience)
        - matched_leadership_signals: int (0 or 1)
        - matched_ai_signals: int (0 or 1)
        - matched_retrieval_signals: int (0 or 1)
        - semantic_similarity: float (mean cosine similarity)
        - retrieval_rank: int (original FAISS position)
        - rrf_rank: int (RRF rank position)
        - original_retrieval_confidence: float
        """
        features_list = []
        
        try:
            for idx, cand in enumerate(reranked_candidates):
                cid = cand["candidate_id"]
                
                # Retrieval metrics
                q_sims = cand.get("query_similarities", {})
                mean_sim = float(pd.Series(list(q_sims.values())).mean()) if q_sims else 0.0
                
                matched_queries = cand.get("matched_queries", [])
                query_coverage = len(matched_queries) / 6.0
                
                # Fetch original ranks (if available, otherwise fallback to pool index)
                q_ranks = cand.get("query_ranks", {})
                mean_retrieval_rank = float(pd.Series(list(q_ranks.values())).mean()) if q_ranks else float(idx + 1)
                
                # Explainability mappings checks
                evidence = cand.get("explainability_evidence", {})
                
                leadership_val = 1 if evidence.get("Matched Leadership", False) else 0
                ai_val = 1 if evidence.get("Matched AI Experience", False) else 0
                retrieval_val = 1 if evidence.get("Matched Search/Retrieval Experience", False) else 0
                
                # Skill matches count (comma values)
                skills_match = cand.get("matched_profile_sections", [])
                skills_count = len([s for s in skills_match if "Skills" in s])
                
                # Responsibilities matches count
                resp_count = 1 if evidence.get("Matched Responsibilities", False) else 0
                
                features = {
                    "candidate_id": cid,
                    "cross_encoder_score": float(cand.get("cross_encoder_probability", 0.0)),
                    "query_coverage": float(query_coverage),
                    "matched_skills_count": int(skills_count),
                    "matched_responsibilities_count": int(resp_count),
                    "matched_experience_count": float(cand.get("years_of_experience", 0.0)),
                    "matched_leadership_signals": int(leadership_val),
                    "matched_ai_signals": int(ai_val),
                    "matched_retrieval_signals": int(retrieval_val),
                    "semantic_similarity": float(mean_sim),
                    "retrieval_rank": int(mean_retrieval_rank),
                    "rrf_rank": int(idx + 1),
                    "original_retrieval_confidence": float(cand.get("confidence_score", 0.0))
                }
                features_list.append(features)
                
            return pd.DataFrame(features_list)
            
        except Exception as e:
            raise RerankerError(f"Failed to compile ranking features dataset: {str(e)}") from e
