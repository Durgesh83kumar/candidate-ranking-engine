from typing import Dict, List, Any, Tuple
from collections import defaultdict
from src.retrieval.exceptions import FusionError

class ReciprocalRankFusion:
    """Fuses multiple ranked search candidate pools using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, k: int = 60):
        self.k = k

    def compute_rrf(self, query_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Applies RRF formula across multiple retrieved pools.
        
        RRF Score = Sum( 1 / (k + rank_i) )
        
        Args:
            query_results: Dict mapping query_type -> List of candidates with candidate_id & score.
            
        Returns:
            List[Dict[str, Any]]: Merged and rank-sorted list of candidates with RRF scores.
        """
        rrf_scores = defaultdict(float)
        query_matches = defaultdict(list)
        semantic_scores = defaultdict(dict)
        candidate_ranks = defaultdict(dict)
        
        try:
            for q_type, hits in query_results.items():
                # Sort by score descending to get ranks
                sorted_hits = sorted(hits, key=lambda x: x.get("score", 0.0), reverse=True)
                
                for rank_idx, hit in enumerate(sorted_hits):
                    cid = hit.get("candidate_id")
                    score = float(hit.get("score", 0.0))
                    
                    # 1-based rank
                    rank = rank_idx + 1
                    
                    # RRF contribution
                    rrf_scores[cid] += 1.0 / (self.k + rank)
                    query_matches[cid].append(q_type)
                    semantic_scores[cid][q_type] = score
                    candidate_ranks[cid][q_type] = rank
                    
            # Assemble fused results list
            fused_list = []
            for cid, rrf_score in rrf_scores.items():
                fused_list.append({
                    "candidate_id": cid,
                    "rrf_score": rrf_score,
                    "matched_queries": query_matches[cid],
                    "query_similarities": semantic_scores[cid],
                    "query_ranks": candidate_ranks[cid]
                })
                
            # Sort final pool by fused RRF score descending
            fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
            return fused_list
            
        except Exception as e:
            raise FusionError(f"Failed to calculate reciprocal rank fusion: {str(e)}") from e
