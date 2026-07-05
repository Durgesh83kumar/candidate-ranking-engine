import time
import os
import psutil
import numpy as np
from typing import List, Dict, Any
from src.reranker.exceptions import RerankerError

class RerankerEvaluator:
    """Measures re-ranking latencies, throughput, logit distributions, and system footprints."""

    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        # Record baseline memory
        self.baseline_mem = self.process.memory_info().rss / (1024 * 1024)

    def evaluate(self, initial_pool: List[Dict[str, Any]], reranked_pool: List[Dict[str, Any]], elapsed_seconds: float) -> Dict[str, Any]:
        """Runs audit check stats on the re-ranked candidates list.
        
        Args:
            initial_pool: Candidates list prior to re-ranking.
            reranked_pool: Final top candidates list.
            elapsed_seconds: Total time elapsed.
            
        Returns:
            Dict[str, Any]: Benchmark and distribution results.
        """
        pool_size = len(reranked_pool)
        if pool_size == 0:
            return {"pool_size": 0, "status": "empty"}
            
        # 1. Score Distributions (Logits & Probabilities)
        logits = [c["cross_encoder_logit"] for c in reranked_pool if "cross_encoder_logit" in c]
        probs = [c["cross_encoder_probability"] for c in reranked_pool if "cross_encoder_probability" in c]
        
        # 2. System resources
        current_mem = self.process.memory_info().rss / (1024 * 1024)
        peak_ram = max(current_mem - self.baseline_mem, 10.0) # MB overhead
        cpu_pct = psutil.cpu_percent(interval=None)
        
        # 3. Candidate text lengths
        # Estimating tokens: approx. 1 token = 4 characters or 0.75 words
        candidate_lengths = []
        for cand in reranked_pool:
            matched_sections = cand.get("matched_profile_sections", [])
            candidate_lengths.append(sum(len(str(s)) for s in matched_sections))
        avg_char_length = float(np.mean(candidate_lengths)) if candidate_lengths else 0.0
        avg_tokens = avg_char_length / 4.0
        
        # 4. Ranking Stability (Mean Rank Change)
        # Compute difference in rank indices between retrieval list and final list
        rank_changes = []
        initial_order = {c["candidate_id"]: idx for idx, c in enumerate(initial_pool)}
        
        for new_idx, cand in enumerate(reranked_pool):
            cid = cand["candidate_id"]
            if cid in initial_order:
                rank_changes.append(abs(new_idx - initial_order[cid]))
        mean_rank_change = float(np.mean(rank_changes)) if rank_changes else 0.0

        throughput = len(initial_pool) / elapsed_seconds if elapsed_seconds > 0 else 0.0

        return {
            "rerank_benchmark": {
                "inference_time_seconds": round(elapsed_seconds, 4),
                "throughput_candidates_per_sec": round(throughput, 2),
                "latency_per_candidate_ms": round((elapsed_seconds / len(initial_pool) * 1000) if initial_pool else 0.0, 2),
                "peak_ram_overhead_mb": round(peak_ram, 2),
                "cpu_usage_pct": round(cpu_pct, 2)
            },
            "candidate_metrics": {
                "average_candidate_character_length": round(avg_char_length, 2),
                "average_candidate_estimated_tokens": round(avg_tokens, 2)
            },
            "score_distributions": {
                "logits": {
                    "min": round(float(np.min(logits)), 4) if logits else 0.0,
                    "max": round(float(np.max(logits)), 4) if logits else 0.0,
                    "mean": round(float(np.mean(logits)), 4) if logits else 0.0,
                    "std": round(float(np.std(logits)), 4) if logits else 0.0
                },
                "probabilities": {
                    "min": round(float(np.min(probs)), 4) if probs else 0.0,
                    "max": round(float(np.max(probs)), 4) if probs else 0.0,
                    "mean": round(float(np.mean(probs)), 4) if probs else 0.0,
                    "std": round(float(np.std(probs)), 4) if probs else 0.0
                }
            },
            "ranking_stability": {
                "mean_rank_change_positions": round(mean_rank_change, 2)
            }
        }
