import time
import numpy as np
from typing import List, Dict, Any
from collections import Counter

class RankingEvaluator:
    """Measures scoring distributions, latencies, business rule trigger frequencies, and honeypot metrics."""

    def __init__(self):
        self.start_time = time.time()

    def evaluate_ranking(self, ranked_pool: List[Dict[str, Any]], elapsed_seconds: float) -> Dict[str, Any]:
        """Runs audit checks on the scored candidate pool.
        
        Args:
            ranked_pool: Final ranked candidates list.
            elapsed_seconds: Total execution latency.
            
        Returns:
            Dict[str, Any]: Compiled evaluation report.
        """
        pool_size = len(ranked_pool)
        if pool_size == 0:
            return {"pool_size": 0, "status": "empty"}
            
        # 1. Score stats
        scores = [c["final_score"] for c in ranked_pool]
        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))
        std_score = float(np.std(scores))
        
        # 2. Business rule triggering metrics
        all_warnings = []
        for c in ranked_pool:
            all_warnings.extend(c.get("triggered_business_rules", []))
        warning_counts = dict(Counter(all_warnings))
        
        # 3. Honeypot triggers
        all_anomalies = []
        honeypot_penalized_count = 0
        for c in ranked_pool:
            anom = c.get("triggered_honeypot_checks", [])
            all_anomalies.extend(anom)
            if len(anom) > 0:
                honeypot_penalized_count += 1
        anomaly_counts = dict(Counter(all_anomalies))
        
        # 4. Feature summary comparisons (Top 100 vs remaining 200)
        top_100_ce = [c["cross_encoder_score"] for c in ranked_pool[:100]]
        bottom_200_ce = [c["cross_encoder_score"] for c in ranked_pool[100:]]
        
        avg_top_100_ce = float(np.mean(top_100_ce)) if top_100_ce else 0.0
        avg_bottom_200_ce = float(np.mean(bottom_200_ce)) if bottom_200_ce else 0.0

        return {
            "ranking_latency_seconds": round(elapsed_seconds, 4),
            "pool_size": pool_size,
            "score_statistics": {
                "min": round(min_score, 4),
                "max": round(max_score, 4),
                "mean": round(mean_score, 4),
                "std": round(std_score, 4)
            },
            "business_rules_triggered_summary": warning_counts,
            "honeypot_detection_summary": {
                "total_candidates_penalized": honeypot_penalized_count,
                "anomalies_distribution": anomaly_counts
            },
            "ranking_quality_checks": {
                "average_cross_encoder_score_top_100": round(avg_top_100_ce, 4),
                "average_cross_encoder_score_bottom_200": round(avg_bottom_200_ce, 4)
            }
        }
ClassEvaluator = RankingEvaluator
