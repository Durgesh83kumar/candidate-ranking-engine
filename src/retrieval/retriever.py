import numpy as np
from typing import List, Dict, Any, Tuple

from src.retrieval.config import RetrievalConfig
from src.retrieval.query_builder import MultiQueryBuilder
from src.retrieval.search import BatchSearcher
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.metadata import EnrichmentJoiner
from src.retrieval.filters import SoftFiltersEvaluator
from src.retrieval.deduplicator import CandidateDeduplicator
from src.retrieval.exceptions import RetrievalError

class SemanticRetriever:
    """Orchestrates candidate retrieval, rank fusion, metadata joins, soft filtering, and confidence checks."""

    def __init__(self, config: RetrievalConfig, spec_path: str, queries_path: str, index_path: str, index_metadata_path: str, candidates_parquet_path: str):
        self.config = config
        self.query_builder = MultiQueryBuilder(spec_path, queries_path)
        self.searcher = BatchSearcher(config, index_path, index_metadata_path)
        self.fusion = ReciprocalRankFusion(config.fusion_constant)
        self.joiner = EnrichmentJoiner(candidates_parquet_path)
        self.deduplicator = CandidateDeduplicator()

    def calculate_confidence(self, rrf_score: float, query_sims: Dict[str, float], matched_queries: List[str], num_total_queries: int, penalty_multiplier: float) -> float:
        """Calculates multi-signal confidence score.
        
        Formula:
            Confidence = (w1 * RRF_Norm + w2 * Mean_Similarity + w3 * Query_Coverage) * Penalty_Multiplier
            
            Where:
            - RRF_Norm = RRF_Score / Max_Possible_RRF
            - Mean_Similarity = Average of similarity scores for matched queries
            - Query_Coverage = Count of matched queries / Total queries
            
            w1 = 0.3, w2 = 0.4, w3 = 0.3
        """
        if num_total_queries <= 0:
            return 0.0
            
        # Max possible RRF for 1 candidate at rank 1 for all queries
        max_rrf = num_total_queries * (1.0 / (self.config.fusion_constant + 1))
        rrf_norm = min(1.0, rrf_score / max_rrf) if max_rrf > 0 else 0.0
        
        # Mean similarity score
        sims_list = list(query_sims.values())
        mean_sim = float(np.mean(sims_list)) if sims_list else 0.0
        
        # Query coverage
        coverage = len(matched_queries) / num_total_queries
        
        # Weighted sum
        w1, w2, w3 = 0.3, 0.4, 0.3
        base_confidence = (w1 * rrf_norm) + (w2 * mean_sim) + (w3 * coverage)
        
        # Apply soft constraints penalty multiplier
        final_confidence = float(base_confidence * penalty_multiplier)
        return min(1.0, max(0.0, final_confidence))

    def retrieve_candidates(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Executes the full semantic candidate retrieval pipeline.
        
        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, Any]]: List of enriched candidates, and summary run statistics.
        """
        # 1. Build queries
        self.query_builder.load_requirements()
        queries = self.query_builder.build_queries()
        negatives = self.query_builder.get_negative_keywords()
        
        # 2. Perform Batch FAISS search
        self.searcher.load_index()
        raw_retrieved = self.searcher.search_all(queries)
        
        # 3. Apply Reciprocal Rank Fusion (RRF)
        fused = self.fusion.compute_rrf(raw_retrieved)
        
        # 4. Join Profile Metadata
        enriched = self.joiner.join_and_enrich(fused, queries, negatives)
        
        # 5. Deduplicate candidates
        deduplicated = self.deduplicator.deduplicate(enriched)
        
        # 6. Apply Recruiter Soft Filters and calculate Confidence
        filter_evaluator = SoftFiltersEvaluator(self.config.filters)
        
        final_pool = []
        num_total_queries = len(queries)
        
        for cand in deduplicated:
            # Check exclusions / soft filters
            failed_filters, penalty = filter_evaluator.evaluate_candidate(cand)
            
            # Apply exclusion filter (hard filter)
            if cand.get("is_excluded", False):
                # Discard candidates who match the negative constraints explicitly
                continue
                
            cand["failed_filters"] = failed_filters
            cand["penalty_multiplier"] = penalty
            
            # Compute confidence score
            conf = self.calculate_confidence(
                rrf_score=cand["rrf_score"],
                query_sims=cand["query_similarities"],
                matched_queries=cand["matched_queries"],
                num_total_queries=num_total_queries,
                penalty_multiplier=penalty
            )
            cand["confidence_score"] = conf
            
            # If similarity scores are lower than minimum threshold, skip
            sims_list = list(cand["query_similarities"].values())
            max_sim = max(sims_list) if sims_list else 0.0
            if max_sim < self.config.minimum_similarity:
                continue
                
            final_pool.append(cand)
            
        # Sort by confidence score descending
        final_pool.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        # Slice to output pool size
        final_pool = final_pool[:self.config.output_pool_size]
        
        # 7. Collect statistics
        total_retrieved = len(fused)
        overlap_rate = (total_retrieved - len(deduplicated)) / total_retrieved if total_retrieved > 0 else 0.0
        
        stats = {
            "total_queries_run": num_total_queries,
            "raw_total_retrieved": total_retrieved,
            "deduplicated_count": len(deduplicated),
            "final_pool_count": len(final_pool),
            "overlap_rate": overlap_rate
        }
        
        return final_pool, stats
