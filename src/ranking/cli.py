import os
import sys
import json
import time
import argparse
import psutil
import pandas as pd
from typing import Dict, Any, List

from src.ranking.config import RankingConfig
from src.ranking.scorer import HybridScorer
from src.ranking.reasoning import ReasoningEngine
from src.ranking.submission import SubmissionGenerator
from src.ranking.evaluator import RankingEvaluator
from src.ranking.exceptions import RankingError

def run_ranking_pipeline(reranked_path: str, reranker_features_path: str, processed_path: str, spec_path: str, output_dir: str, config: RankingConfig) -> None:
    """Orchestrates candidate enrichments, hybrid scoring, reasoning generation, validation, and exports."""
    start_time = time.time()
    process = psutil.Process(os.getpid())
    baseline_mem = process.memory_info().rss / (1024 * 1024)
    
    os.makedirs(output_dir, exist_ok=True)
    print("Initializing Hybrid Ranking Engine (Phase 7)...")
    
    # Instantiate modules
    scorer = HybridScorer(config=config)
    reasoning_engine = ReasoningEngine()
    submission_gen = SubmissionGenerator()
    evaluator = RankingEvaluator()
    
    # 1. Load Phase 6 Reranked candidates Parquet
    if not os.path.exists(reranked_path):
        raise RankingError(f"Reranked candidates Parquet not found: {reranked_path}")
    print(f"Loading Phase 6 reranked candidates from {reranked_path}...")
    df_reranked = pd.read_parquet(reranked_path)
    
    # 2. Ingest rich processed candidates fields
    if not os.path.exists(processed_path):
        raise RankingError(f"Processed candidates Parquet not found: {processed_path}")
    print(f"Loading raw candidate schema fields from {processed_path}...")
    # Load only necessary structural columns
    df_processed = pd.read_parquet(processed_path, columns=["candidate_id", "career_history", "education", "skills", "certifications"])
    df_processed.set_index("candidate_id", inplace=True)
    
    # Ingest reranked features
    if not os.path.exists(reranker_features_path):
        raise RankingError(f"Reranker features Parquet not found: {reranker_features_path}")
    print(f"Loading re-ranking features from {reranker_features_path}...")
    df_rerank_features = pd.read_parquet(reranker_features_path)
    df_rerank_features.set_index("candidate_id", inplace=True)
    
    # 3. Enrich, Score, and Generate Explanations
    candidates_list = df_reranked.to_dict(orient="records")
    print(f"Evaluating {len(candidates_list)} candidates...")
    
    # Find max RRF score for retrieval component scaling
    max_rrf = df_reranked["rrf_score"].max() if "rrf_score" in df_reranked.columns else 0.10
    if pd.isna(max_rrf) or max_rrf <= 0:
        max_rrf = 0.10
        
    scored_pool = []
    explanations = {}
    honeypot_log = {}
    
    for cand in candidates_list:
        cid = cand["candidate_id"]
        cand_copy = dict(cand)
        
        # Join rich career history and education fields from raw parquet
        if cid in df_processed.index:
            row_proc = df_processed.loc[cid]
            if isinstance(row_proc, pd.DataFrame):
                row_proc = row_proc.iloc[0]
                
            career_history = row_proc.get("career_history")
            education = row_proc.get("education")
            skills = row_proc.get("skills")
            certifications = row_proc.get("certifications")
            
            # Convert PyArrow/numpy ndarrays to standard Python lists
            if hasattr(career_history, "tolist"):
                career_history = career_history.tolist()
            if hasattr(education, "tolist"):
                education = education.tolist()
            if hasattr(skills, "tolist"):
                skills = skills.tolist()
            if hasattr(certifications, "tolist"):
                certifications = certifications.tolist()
                
            # Ensure elements inside list are dictionaries (in case of double wrapping)
            if isinstance(career_history, list):
                career_history = [c.tolist() if hasattr(c, "tolist") else c for c in career_history]
                # Filter out any None values
                career_history = [c for c in career_history if c is not None]
            else:
                career_history = []
                
            if isinstance(education, list):
                education = [e.tolist() if hasattr(e, "tolist") else e for e in education]
                education = [e for e in education if e is not None]
            else:
                education = []
                
            if not isinstance(skills, list):
                skills = []
            if not isinstance(certifications, list):
                certifications = []
                
            cand_copy["career_history"] = career_history
            cand_copy["education"] = education
            cand_copy["skills"] = skills
            cand_copy["certifications"] = certifications
        else:
            cand_copy["career_history"] = []
            cand_copy["education"] = []
            cand_copy["skills"] = []
            cand_copy["certifications"] = []
            
        # Run scoring engine
        scores_breakdown = scorer.compute_hybrid_score(cand_copy, max_rrf_score=max_rrf)
        
        # Merge scores back to record
        cand_copy.update(scores_breakdown)
        
        # Generate reasoning
        reasoning = reasoning_engine.generate_reasoning(cand_copy)
        cand_copy["reasoning"] = reasoning
        
        scored_pool.append(cand_copy)
        
        # Record explainability
        explanations[cid] = {
            "retrieval_score_normalized": scores_breakdown["retrieval_score"],
            "cross_encoder_score": scores_breakdown["cross_encoder_score"],
            "career_quality_score": scores_breakdown["career_score"],
            "profile_quality_score": scores_breakdown["profile_score"],
            "business_multiplier": scores_breakdown["business_multiplier"],
            "honeypot_multiplier": scores_breakdown["honeypot_multiplier"],
            "final_hybrid_score": scores_breakdown["final_score"],
            "triggered_business_rules": scores_breakdown["triggered_business_rules"],
            "triggered_honeypot_checks": scores_breakdown["triggered_honeypot_checks"]
        }
        
        # Record honeypot analysis
        if scores_breakdown["triggered_honeypot_checks"]:
            honeypot_log[cid] = scores_breakdown["triggered_honeypot_checks"]
            
    # Sort final pool descending by final_score
    scored_pool.sort(key=lambda x: x["final_score"], reverse=True)
    
    elapsed_time = time.time() - start_time
    print(f"Hybrid scoring finished. Execution time: {elapsed_time:.2f} seconds.")
    
    # 4. Generate and Validate submission.csv (Exactly Top 100)
    submission_csv_out = os.path.join(output_dir, "submission.csv")
    submission_gen.validate_and_save(scored_pool, submission_csv_out)
    
    # 5. Evaluate Metrics
    eval_report = evaluator.evaluate_ranking(scored_pool, elapsed_time)
    
    # 6. Save expected outputs
    # A. output/final_ranked_candidates.parquet (Snappy, top 300)
    ranked_parquet_out = os.path.join(output_dir, "final_ranked_candidates.parquet")
    # Drop python list/object columns before writing parquet for simplicity
    df_ranked = pd.DataFrame(scored_pool)
    parquet_df = df_ranked.drop(columns=["career_history", "education", "skills", "certifications"], errors="ignore")
    parquet_df.to_parquet(ranked_parquet_out, compression="snappy", index=False)
    
    # B. output/ranking_features.parquet (Aggregated features)
    features_parquet_out = os.path.join(output_dir, "ranking_features.parquet")
    df_features_all = df_rerank_features.copy()
    # Add final scores and rankings
    final_scores_map = {c["candidate_id"]: c["final_score"] for c in scored_pool}
    final_ranks_map = {c["candidate_id"]: idx + 1 for idx, c in enumerate(scored_pool)}
    df_features_all["final_hybrid_score"] = df_features_all.index.map(final_scores_map)
    df_features_all["final_rank"] = df_features_all.index.map(final_ranks_map)
    df_features_all.reset_index(inplace=True)
    df_features_all.to_parquet(features_parquet_out, compression="snappy", index=False)
    
    # C. output/ranking_statistics.json
    stats_out = os.path.join(output_dir, "ranking_statistics.json")
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
        
    # D. output/candidate_reasoning.json
    reasoning_out = os.path.join(output_dir, "candidate_reasoning.json")
    reasoning_registry = {c["candidate_id"]: c["reasoning"] for c in scored_pool}
    with open(reasoning_out, "w", encoding="utf-8") as f:
        json.dump(reasoning_registry, f, indent=2)
        
    # E. output/honeypot_analysis.json
    honeypot_out = os.path.join(output_dir, "honeypot_analysis.json")
    with open(honeypot_out, "w", encoding="utf-8") as f:
        json.dump(honeypot_log, f, indent=2)
        
    # F. output/ranking_explanation.json
    explanation_out = os.path.join(output_dir, "ranking_explanation.json")
    with open(explanation_out, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)
        
    # G. output/ranking_benchmark.json
    benchmark_out = os.path.join(output_dir, "ranking_benchmark.json")
    current_mem = process.memory_info().rss / (1024 * 1024)
    peak_ram = max(current_mem - baseline_mem, 10.0)
    benchmark_report = {
        "execution_latency_seconds": elapsed_time,
        "peak_ram_overhead_mb": round(peak_ram, 2),
        "cpu_usage_pct": round(psutil.cpu_percent(), 2),
        "throughput_candidates_per_second": round(len(candidates_list) / elapsed_time, 2) if elapsed_time > 0 else 0.0
    }
    with open(benchmark_out, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
        
    print("--------------------------------------------------")
    print(f"Final hybrid rankings compiled successfully!")
    print(f"Top 100 submission: {submission_csv_out}")
    print(f"Benchmark Report:   {benchmark_out}")
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Run Hybrid Ranking Engine (Phase 7)."
    )
    parser.add_argument(
        "--reranked",
        type=str,
        default="output/reranked_candidates.parquet",
        help="Path to Phase 6 reranked candidates parquet file."
    )
    parser.add_argument(
        "--reranker-features",
        type=str,
        default="output/reranker_features.parquet",
        help="Path to Phase 6 reranked features parquet file."
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="output/processed_candidates.parquet",
        help="Path to preprocessed candidates Parquet file containing raw timelines."
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
        help="Output directory to save ranking and submission CSV files."
    )
    
    # Custom weight configs
    parser.add_argument("--w-retrieval", type=float, default=0.20, help="Retrieval score weight.")
    parser.add_argument("--w-cross-encoder", type=float, default=0.40, help="Cross Encoder score weight.")
    parser.add_argument("--w-career", type=float, default=0.25, help="Career score weight.")
    parser.add_argument("--w-profile", type=float, default=0.15, help="Profile score weight.")

    args = parser.parse_args()
    
    config = RankingConfig(
        weight_retrieval=args.w_retrieval,
        weight_cross_encoder=args.w_cross_encoder,
        weight_career=args.w_career,
        weight_profile=args.w_profile
    )
    
    try:
        run_ranking_pipeline(
            reranked_path=args.reranked,
            reranker_features_path=args.reranker_features,
            processed_path=args.processed,
            spec_path=args.jd,
            output_dir=args.output,
            config=config
        )
    except Exception as e:
        print(f"Hybrid ranking engine failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
