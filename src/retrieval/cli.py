import os
import sys
import json
import time
import argparse
import pandas as pd
from typing import Dict, Any

from src.retrieval.config import RetrievalConfig
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.evaluator import RetrievalEvaluator
from src.retrieval.exceptions import RetrievalError

def run_pipeline(jd_spec_path: str, search_queries_path: str, faiss_index_path: str, index_metadata_path: str, candidate_parquet_path: str, output_dir: str, config: RetrievalConfig) -> None:
    """Orchestrates candidate retrieval, rank fusion, metadata joins, filtering, and evaluation."""
    start_time = time.time()
    evaluator = RetrievalEvaluator()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing Candidate Retrieval Engine...")
    retriever = SemanticRetriever(
        config=config,
        spec_path=jd_spec_path,
        queries_path=search_queries_path,
        index_path=faiss_index_path,
        index_metadata_path=index_metadata_path,
        candidates_parquet_path=candidate_parquet_path
    )
    
    # Run retrieval
    print("Running multi-query searches and RRF fusion...")
    final_candidates, stats = retriever.retrieve_candidates()
    
    elapsed_time = time.time() - start_time
    print(f"Retrieval completed in {elapsed_time:.2f} seconds. Retrieved {len(final_candidates)} candidates.")
    
    # Evaluate
    print("Evaluating retrieval pool quality and diversity...")
    eval_report = evaluator.evaluate_pool(final_candidates, stats, elapsed_time)
    
    # Prepare serializable deliverables
    # 1. Parquet
    df_pool = pd.DataFrame([dict(c) for c in final_candidates])
    
    # Save parquet candidate pool
    parquet_out = os.path.join(output_dir, "retrieval_candidates.parquet")
    if not df_pool.empty:
        df_pool.to_parquet(parquet_out, compression="snappy", index=False)
    else:
        # Save empty placeholder df with correct schema columns
        pd.DataFrame(columns=[
            "candidate_id", "anonymized_name", "rrf_score", "confidence_score",
            "matched_queries", "matched_profile_sections", "years_of_experience",
            "current_title", "current_company", "location", "country", "expected_salary_lpa",
            "notice_period_days", "open_to_work", "github_activity", "profile_completeness",
            "work_mode", "relocation", "query_similarities", "query_ranks", "failed_filters"
        ]).to_parquet(parquet_out, compression="snappy", index=False)
        
    print(f"Saved candidate pool: {parquet_out}")
    
    # 2. retrieval_scores.json
    scores_out = os.path.join(output_dir, "retrieval_scores.json")
    scores_list = [
        {
            "candidate_id": c["candidate_id"],
            "rrf_score": c["rrf_score"],
            "confidence_score": c["confidence_score"]
        } for c in final_candidates
    ]
    with open(scores_out, "w", encoding="utf-8") as f:
        json.dump(scores_list, f, indent=2)
        
    # 3. retrieval_statistics.json
    stats_out = os.path.join(output_dir, "retrieval_statistics.json")
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    # 4. query_analysis.json
    analysis_out = os.path.join(output_dir, "query_analysis.json")
    query_analysis = {
        "query_contributions": eval_report.get("query_contribution_analysis", {}),
        "recall_evaluation_notice": eval_report.get("recall_evaluation_notice")
    }
    with open(analysis_out, "w", encoding="utf-8") as f:
        json.dump(query_analysis, f, indent=2)
        
    # 5. retrieval_benchmark.json
    benchmark_out = os.path.join(output_dir, "retrieval_benchmark.json")
    benchmark_metrics = {
        "execution_latency_seconds": elapsed_time,
        "average_similarity": eval_report.get("average_similarity", 0.0),
        "similarity_range": eval_report.get("similarity_range", {}),
        "candidate_overlap_rate": eval_report.get("candidate_overlap_rate", 0.0),
        "diversity_statistics": eval_report.get("diversity_statistics", {})
    }
    with open(benchmark_out, "w", encoding="utf-8") as f:
        json.dump(benchmark_metrics, f, indent=2)
        
    print("Semantic Candidate Retrieval completed successfully!")
    print("--------------------------------------------------")
    print(f"Total Candidates: {len(final_candidates)}")
    print(f"Scores written:   {scores_out}")
    print(f"Metrics written:  {stats_out}")
    print(f"Benchmarks:       {benchmark_out}")
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Run Semantic Candidate Retrieval Engine (Phase 5)."
    )
    parser.add_argument(
        "--jd",
        type=str,
        default="output/hiring_specification.json",
        help="Path to Hiring Specification JSON."
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="output/search_queries.json",
        help="Path to search queries JSON."
    )
    parser.add_argument(
        "--index",
        type=str,
        default="output/faiss.index",
        help="Path to FAISS index FlatIP file."
    )
    parser.add_argument(
        "--index-metadata",
        type=str,
        default="output/index_metadata.json",
        help="Path to index candidate mappings metadata JSON."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="output/processed_candidates.parquet",
        help="Path to preprocessed candidates Parquet data file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory to save retrieval parquet pool and statistics reports."
    )
    
    # Config parameters
    parser.add_argument("--top-k", type=int, default=300, help="Retrieval candidate limit per query.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank decay constant.")
    parser.add_argument("--min-sim", type=float, default=0.0, help="Minimum cosine similarity threshold.")
    parser.add_argument("--pool-size", type=int, default=1000, help="Max candidates in final retrieval pool.")
    
    # Recruiting filters
    parser.add_argument("--min-exp", type=float, default=None, help="Recruiter preference: min years of experience.")
    parser.add_argument("--pref-country", type=str, default=None, help="Recruiter preference: preferred country.")
    parser.add_argument("--pref-city", type=str, default=None, help="Recruiter preference: preferred city.")
    parser.add_argument("--max-salary", type=float, default=None, help="Recruiter preference: max salary in LPA.")
    parser.add_argument("--max-notice", type=int, default=None, help="Recruiter preference: max notice period in days.")
    parser.add_argument("--require-reloc", action="store_true", help="Recruiter preference: requires relocation if not local.")
    parser.add_argument("--pref-work-mode", type=str, default=None, help="Recruiter preference: preferred work mode.")

    args = parser.parse_args()
    
    # Map recruiter preferences
    filters = {}
    if args.min_exp is not None:
        filters["min_experience"] = args.min_exp
    if args.pref_country is not None:
        filters["preferred_country"] = args.pref_country
    if args.pref_city is not None:
        filters["preferred_city"] = args.pref_city
    if args.max_salary is not None:
        filters["salary_max_lpa"] = args.max_salary
    if args.max_notice is not None:
        filters["max_notice_period_days"] = args.max_notice
    if args.require_reloc:
        filters["relocation_required"] = True
    if args.pref_work_mode is not None:
        filters["preferred_work_modes"] = [args.pref_work_mode]

    config = RetrievalConfig(
        top_k_per_query=args.top_k,
        fusion_constant=args.rrf_k,
        minimum_similarity=args.min_sim,
        output_pool_size=args.pool_size,
        filters=filters
    )
    
    try:
        run_pipeline(
            jd_spec_path=args.jd,
            search_queries_path=args.queries,
            faiss_index_path=args.index,
            index_metadata_path=args.index_metadata,
            candidate_parquet_path=args.metadata,
            output_dir=args.output,
            config=config
        )
    except Exception as e:
        print(f"Retrieval engine failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
