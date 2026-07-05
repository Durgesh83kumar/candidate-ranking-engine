import time
import numpy as np
from typing import List, Dict, Any
from collections import Counter

class RetrievalEvaluator:
    """Measures semantic retrieval similarity spreads, overlap, query contribution, and diversity."""

    def __init__(self):
        self.start_time = time.time()

    def evaluate_pool(self, final_pool: List[Dict[str, Any]], stats: Dict[str, Any], elapsed_seconds: float) -> Dict[str, Any]:
        """Runs audit checks on the retrieved candidate pool.
        
        Args:
            final_pool: Final retrieved list.
            stats: Summary metrics from retriever.
            elapsed_seconds: Pipeline execution time.
            
        Returns:
            Dict[str, Any]: Compiled evaluation report.
        """
        pool_size = len(final_pool)
        if pool_size == 0:
            return {
                "pool_size": 0,
                "recall_evaluation_notice": "Recall evaluation requires ground-truth labels. No candidates were retrieved.",
                "latency_seconds": elapsed_seconds
            }
            
        # 1. Similarity stats
        all_sims = []
        for cand in final_pool:
            all_sims.extend(list(cand.get("query_similarities", {}).values()))
        mean_sim = float(np.mean(all_sims)) if all_sims else 0.0
        max_sim = float(np.max(all_sims)) if all_sims else 0.0
        min_sim = float(np.min(all_sims)) if all_sims else 0.0
        
        # 2. Query contribution stats (which query retrieved the most candidates)
        query_contributions = Counter()
        for cand in final_pool:
            for q_type in cand.get("matched_queries", []):
                query_contributions[q_type] += 1
                
        # 3. Diversity stats (Title & Location spreads)
        titles = [cand.get("current_title", "Unknown") for cand in final_pool]
        locations = [cand.get("location", "Unknown") for cand in final_pool]
        
        title_dist = dict(Counter(titles).most_common(10))
        loc_dist = dict(Counter(locations).most_common(10))
        
        # 4. Overlap metrics
        overlap_rate = stats.get("overlap_rate", 0.0)

        report = {
            "retrieval_latency_seconds": round(elapsed_seconds, 4),
            "pool_size": pool_size,
            "average_similarity": round(mean_sim, 4),
            "similarity_range": {
                "min": round(min_sim, 4),
                "max": round(max_sim, 4)
            },
            "query_contribution_analysis": dict(query_contributions),
            "candidate_overlap_rate": round(overlap_rate, 4),
            "diversity_statistics": {
                "top_titles_distribution": title_dist,
                "top_locations_distribution": loc_dist,
                "unique_titles_count": len(set(titles)),
                "unique_locations_count": len(set(locations))
            },
            "recall_evaluation_notice": (
                "Recall evaluation requires ground-truth labels. Standard metrics "
                "like Recall@10, Recall@50, and Recall@100 are omitted because no "
                "relevance labels are present in the dataset."
            )
        }
        
        return report
