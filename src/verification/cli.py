import os
import sys
import json
import time
import argparse
import psutil
import pandas as pd
from typing import Dict, Any, List

from src.verification.config import VerificationConfig
from src.verification.calibrator import ScoreCalibrator
from src.verification.exceptions import VerificationError, VerificationWarning

def validate_and_save_submission(ranked_pool: List[Dict[str, Any]], output_path: str) -> None:
    """Checks Top 100, validates constraints (rank, score ordering, word limits), and saves CSV.
    
    Args:
        ranked_pool: Verification-calibrated candidates pool sorted descending by score.
        output_path: Destination path for submission.csv.
    """
    top_100 = ranked_pool[:100]
    
    # Task 3: Automated Quality Assertions
    if len(top_100) != 100:
        raise VerificationWarning(f"Failed Quality Assertion: Top 100 candidates list must have exactly 100 rows. Got {len(top_100)}.")
        
    for cand in top_100:
        reasoning = cand.get("reasoning", "")
        if "Failed verification" in reasoning:
            raise VerificationWarning(
                f"Failed Quality Assertion: Candidate '{cand.get('candidate_id')}' in the Top 100 has a failed verification reasoning. "
                "Retrieval pool tuning needs further adjustment to push penalized profiles out."
            )
            
    if len(top_100) < 100:
        raise VerificationError(f"Verification pool contains only {len(top_100)} candidates. Exactly 100 are required.")
        
    rows = []
    seen_ids = set()
    
    for idx, cand in enumerate(top_100):
        cid = cand.get("candidate_id")
        score = cand.get("calibrated_score")
        reasoning = cand.get("reasoning", "")
        
        if not cid:
            raise VerificationError(f"Missing candidate ID at index {idx}.")
        if cid in seen_ids:
            raise VerificationError(f"Duplicate candidate ID '{cid}' detected in verified ranking.")
        seen_ids.add(cid)
        
        if score is None or pd.isna(score):
            raise VerificationError(f"Candidate {cid} has NaN calibrated score.")
        score_val = float(score)
        
        # Word limits validation
        words = reasoning.split()
        word_count = len(words)
        if word_count < 10 or word_count > 50:
            raise VerificationError(
                f"Candidate {cid}: Reasoning word count of {word_count} violates "
                f"the constraint [10, 50] words. Text: '{reasoning}'"
            )
            
        rows.append({
            "candidate_id": cid,
            "rank": idx + 1,
            "score": score_val,
            "reasoning": reasoning
        })
        
    df_sub = pd.DataFrame(rows)
    
    # Ranks validation
    if not df_sub["rank"].is_unique:
        raise VerificationError("Ranks are not unique.")
        
    # Sorted score ordering check
    scores_list = df_sub["score"].tolist()
    is_sorted = all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1))
    if not is_sorted:
        raise VerificationError("Verified hybrid scores are not sorted in descending order.")
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save CSV with strictly ordered columns
    df_sub[["candidate_id", "rank", "score", "reasoning"]].to_csv(output_path, index=False)
    print(f"Verified submission successfully written to {output_path}.")

def run_verification_pipeline(final_candidates_path: str, processed_path: str, spec_path: str, output_dir: str, config: VerificationConfig) -> None:
    """Orchestrates Phase 8 Candidate Verification pipeline."""
    start_time = time.time()
    process = psutil.Process(os.getpid())
    baseline_mem = process.memory_info().rss / (1024 * 1024)
    
    os.makedirs(output_dir, exist_ok=True)
    print("Initializing Candidate Verification Engine (Phase 8)...")
    
    # Initialize calibrator
    calibrator = ScoreCalibrator(config, spec_path)
    
    # 1. Load Phase 7 Ranked Candidates
    if not os.path.exists(final_candidates_path):
        # Fall back to outputs folder in current workspace if path is not found
        alt_path = os.path.join("output", "final_ranked_candidates.parquet")
        if os.path.exists(alt_path):
            final_candidates_path = alt_path
        else:
            raise VerificationError(f"Phase 7 ranked candidates not found: {final_candidates_path}")
            
    print(f"Loading Phase 7 final candidates from {final_candidates_path}...")
    df_ranked = pd.read_parquet(final_candidates_path)
    
    # 2. Ingest raw candidate skills lists
    if not os.path.exists(processed_path):
        alt_proc = os.path.join("output", "processed_candidates.parquet")
        if os.path.exists(alt_proc):
            processed_path = alt_proc
        else:
            raise VerificationError(f"Processed candidates Parquet not found: {processed_path}")
            
    print(f"Loading candidate skills from {processed_path}...")
    df_processed = pd.read_parquet(processed_path, columns=["candidate_id", "skills", "search_document_v2"])
    df_processed.set_index("candidate_id", inplace=True)
    
    # 3. Perform Calibration
    candidates_list = df_ranked.to_dict(orient="records")
    calibrated_pool = []
    
    # Tracking statistics counts
    stats = {
        "total_evaluated": len(candidates_list),
        "recruiter_phrasing_detected": 0,
        "engineering_phrasing_detected": 0,
        "skills_missing_penalized": 0,
        "ai_frameworks_missing_penalized": 0,
        "honeypot_disqualified": 0
    }
    
    for cand in candidates_list:
        cid = cand["candidate_id"]
        cand_copy = dict(cand)
        
        # Pull original skills and search doc v2
        if cid in df_processed.index:
            row_proc = df_processed.loc[cid]
            if isinstance(row_proc, pd.DataFrame):
                row_proc = row_proc.iloc[0]
            cand_copy["skills"] = row_proc.get("skills")
            cand_copy["search_document_v2"] = row_proc.get("search_document_v2")
        else:
            cand_copy["skills"] = []
            cand_copy["search_document_v2"] = ""
            
        # Calibrate
        cal_res = calibrator.calibrate_candidate(cand_copy)
        cand_copy.update(cal_res)
        
        # Accumulate stats
        if cal_res["has_recruiter_phrasing"]:
            stats["recruiter_phrasing_detected"] += 1
        if cal_res["has_engineering_phrasing"]:
            stats["engineering_phrasing_detected"] += 1
        if cal_res["skills_multiplier"] < 1.0:
            stats["skills_missing_penalized"] += 1
        if cal_res["ai_specialist_multiplier"] < 1.0:
            stats["ai_frameworks_missing_penalized"] += 1
        if cal_res["is_flaged_honeypot"]:
            stats["honeypot_disqualified"] += 1
            
        calibrated_pool.append(cand_copy)
        
    # Re-sort pool descending by calibrated score
    calibrated_pool.sort(key=lambda x: x["calibrated_score"], reverse=True)
    
    # 4. Save submission.csv
    sub_csv_path = os.path.join(output_dir, "submission.csv")
    validate_and_save_submission(calibrated_pool, sub_csv_path)
    
    # 5. Save intermediate parquet and report logs
    # Verified Candidates Parquet (Top 300)
    verified_parquet = os.path.join(output_dir, "verified_candidates.parquet")
    df_verified = pd.DataFrame(calibrated_pool)
    # Drop list/object fields to keep it flat
    df_verified_flat = df_verified.drop(columns=["skills", "search_document_v2", "career_history", "education", "certifications"], errors="ignore")
    df_verified_flat.to_parquet(verified_parquet, compression="snappy", index=False)
    
    # Stats JSON
    stats_json = os.path.join(output_dir, "verification_statistics.json")
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    # Benchmarks
    elapsed = time.time() - start_time
    mem_overhead = max(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) - baseline_mem, 5.0)
    benchmark_json = os.path.join(output_dir, "verification_benchmark.json")
    benchmark_report = {
        "verification_latency_seconds": elapsed,
        "peak_ram_overhead_mb": round(mem_overhead, 2),
        "throughput_candidates_per_second": round(len(candidates_list) / elapsed, 2) if elapsed > 0 else 0.0
    }
    with open(benchmark_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
        
    print("--------------------------------------------------")
    print(f"Verification pipeline completed in {elapsed:.2f} seconds.")
    print(f"Top 100 CSV: {sub_csv_path}")
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Run Candidate Verification (Phase 8).")
    parser.add_argument(
        "--final-candidates",
        type=str,
        default="output/final_ranked_candidates.parquet",
        help="Path to final ranked candidates parquet."
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="output/processed_candidates.parquet",
        help="Path to preprocessed candidates Parquet."
    )
    parser.add_argument(
        "--jd",
        type=str,
        default="output/hiring_specification.json",
        help="Path to hiring specification JSON."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory to save csv and logs."
    )

    args = parser.parse_args()
    
    config = VerificationConfig()
    
    try:
        run_verification_pipeline(
            final_candidates_path=args.final_candidates,
            processed_path=args.processed,
            spec_path=args.jd,
            output_dir=args.output,
            config=config
        )
    except Exception as e:
        print(f"Candidate Verification Engine failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
