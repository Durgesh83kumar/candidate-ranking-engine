import os
import sys
import json
import time
import argparse
import pandas as pd
from typing import Dict, Any, List

from src.reranker.config import RerankerConfig
from src.reranker.pair_builder import RerankingPairBuilder
from src.reranker.scorer import BatchRerankingScorer
from src.reranker.feature_engineering import FeatureEngineeringManager
from src.reranker.explainability import ExplainabilityEngine
from src.reranker.evaluator import RerankerEvaluator
from src.reranker.exceptions import RerankerError

def run_reranking_pipeline(retrieval_path: str, search_docs_path: str, spec_path: str, output_dir: str, config: RerankerConfig) -> None:
    """Orchestrates candidate loading, pair construction, batch inference, features compile, and exports."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing Cross-Encoder Re-ranking Engine...")
    evaluator = RerankerEvaluator()
    pair_builder = RerankingPairBuilder(spec_path=spec_path)
    scorer = BatchRerankingScorer(config=config)
    feature_mgr = FeatureEngineeringManager()
    explain_engine = ExplainabilityEngine()
    
    # 1. Load retrieved candidates
    if not os.path.exists(retrieval_path):
        raise RerankerError(f"Retrieval candidates file not found: {retrieval_path}")
    print(f"Loading retrieved candidates from {retrieval_path}...")
    df_retrieved = pd.read_parquet(retrieval_path)
    
    # Convert candidates DataFrame rows back to Dict records
    initial_candidates = df_retrieved.to_dict(orient="records")
    print(f"Ingested {len(initial_candidates)} candidates from Phase 5.")
    
    # 2. Load search document texts (v2) from Phase 3
    if not os.path.exists(search_docs_path):
        raise RerankerError(f"Search documents file not found: {search_docs_path}")
    print(f"Loading candidate search documents from {search_docs_path}...")
    # Load only necessary columns for efficiency
    df_search_docs = pd.read_parquet(search_docs_path, columns=["candidate_id", "search_document_v2"])
    
    # 3. Build text pairs (Query + Candidate Doc)
    print("Constructing sequence pairs using comprehensive recruiter query...")
    pairs = pair_builder.construct_pairs(initial_candidates, df_search_docs)
    
    # 4. Execute Batch inference
    print("Executing Cross-Encoder CPU scoring...")
    scored_candidates = scorer.score_pairs(pairs)
    
    # 5. Enrich candidate pool records with logits and probabilities
    enriched_pool = []
    for cand in initial_candidates:
        cid = cand["candidate_id"]
        if cid in scored_candidates:
            logit, prob = scored_candidates[cid]
            cand_copy = dict(cand)
            cand_copy["cross_encoder_logit"] = logit
            cand_copy["cross_encoder_probability"] = prob
            
            # Map semantic similarity score (using maximum general query similarity or mean)
            sims = cand.get("query_similarities", {})
            cand_copy["semantic_similarity"] = float(pd.Series(list(sims.values())).mean()) if sims else 0.0
            
            # Preserve retrieval_score (mapping from rrf_score or general similarity)
            cand_copy["retrieval_score"] = cand.get("rrf_score", 0.0)
            
            # Generate explainability evidence
            evidence = explain_engine.extract_evidence(cand_copy)
            cand_copy["explainability_evidence"] = evidence
            
            enriched_pool.append(cand_copy)
            
    # Sort by cross_encoder_probability score descending
    enriched_pool.sort(key=lambda x: x["cross_encoder_probability"], reverse=True)
    
    # Slice to top 300 candidates
    top_300_candidates = enriched_pool[:config.top_candidates]
    print(f"Sliced candidate pool to Top {len(top_300_candidates)} re-ranked records.")
    
    elapsed_time = time.time() - start_time
    print(f"Re-ranking completed in {elapsed_time:.2f} seconds.")
    
    # 6. Evaluate metrics
    print("Evaluating pool quality and score distribution...")
    eval_report = evaluator.evaluate(initial_candidates, top_300_candidates, elapsed_time)
    
    # 7. Compile features DataFrame
    print("Generating rich ranking features dataset...")
    df_features = feature_mgr.compile_features(top_300_candidates)
    
    # 8. Save Deliverables
    # A. output/reranked_candidates.parquet
    reranked_parquet_out = os.path.join(output_dir, "reranked_candidates.parquet")
    df_reranked = pd.DataFrame(top_300_candidates)
    df_reranked.to_parquet(reranked_parquet_out, compression="snappy", index=False)
    
    # B. output/reranker_features.parquet
    features_parquet_out = os.path.join(output_dir, "reranker_features.parquet")
    df_features.to_parquet(features_parquet_out, compression="snappy", index=False)
    
    # C. output/reranker_scores.json
    scores_out = os.path.join(output_dir, "reranker_scores.json")
    scores_registry = [
        {
            "candidate_id": c["candidate_id"],
            "retrieval_score": c.get("retrieval_score", 0.0),
            "rrf_score": c.get("rrf_score", 0.0),
            "semantic_similarity": c.get("semantic_similarity", 0.0),
            "cross_encoder_logit": c["cross_encoder_logit"],
            "cross_encoder_probability": c["cross_encoder_probability"],
            "confidence_score": c.get("confidence_score", 0.0)
        } for c in top_300_candidates
    ]
    with open(scores_out, "w", encoding="utf-8") as f:
        json.dump(scores_registry, f, indent=2)
        
    # D. output/reranker_statistics.json
    stats_out = os.path.join(output_dir, "reranker_statistics.json")
    stats_data = {
        "total_candidates_evaluated": len(initial_candidates),
        "total_candidates_retained": len(top_300_candidates),
        "score_distributions": eval_report.get("score_distributions", {}),
        "ranking_stability": eval_report.get("ranking_stability", {})
    }
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)
        
    # E. output/reranker_benchmark.json
    benchmark_out = os.path.join(output_dir, "reranker_benchmark.json")
    benchmark_data = {
        "execution_latency_seconds": elapsed_time,
        "rerank_benchmark": eval_report.get("rerank_benchmark", {}),
        "candidate_metrics": eval_report.get("candidate_metrics", {})
    }
    with open(benchmark_out, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
        
    print("Cross-Encoder Re-ranking Engine completed successfully!")
    print("--------------------------------------------------")
    print(f"Top 300 Candidates: {reranked_parquet_out}")
    print(f"Scores Registry:     {scores_out}")
    print(f"Features Parquet:    {features_parquet_out}")
    print(f"Benchmarks report:   {benchmark_out}")
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Run Cross-Encoder Re-ranking Engine (Phase 6)."
    )
    parser.add_argument(
        "--retrieval",
        type=str,
        default="output/retrieval_candidates.parquet",
        help="Path to Phase 5 retrieval candidates parquet file."
    )
    parser.add_argument(
        "--search-docs",
        type=str,
        default="output/search_documents.parquet",
        help="Path to Phase 3 search documents parquet file."
    )
    parser.add_argument(
        "--jd",
        type=str,
        default="output/hiring_specification.json",
        help="Path to hiring specification JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory to save re-ranked parquet and score registries."
    )
    
    # Reranking configs
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-base", help="Primary Cross-Encoder model name.")
    parser.add_argument("--fallback-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Fallback Cross-Encoder model name.")
    parser.add_argument("--batch-size", type=int, default=32, choices=[16, 32, 64], help="Inference batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length for Cross-Encoder.")
    parser.add_argument("--device", type=str, default="cpu", help="Device context (forces cpu).")
    parser.add_argument("--top-k", type=int, default=300, help="Number of final candidates to retain.")

    args = parser.parse_args()
    
    config = RerankerConfig(
        model_name=args.model,
        fallback_model_name=args.fallback_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        top_candidates=args.top_k
    )
    
    try:
        run_reranking_pipeline(
            retrieval_path=args.retrieval,
            search_docs_path=args.search_docs,
            spec_path=args.jd,
            output_dir=args.output,
            config=config
        )
    except Exception as e:
        print(f"Re-ranking engine execution failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
