#!/usr/bin/env python3
"""
CLI runner for the candidate data preprocessing pipeline.
Converts raw candidate JSONL/JSON data into a clean, normalized, and enriched Parquet dataset.
"""

import argparse
import json
import os
import sys

# Add the project root to sys.path to resolve imports properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.pipeline import PreprocessingPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run candidate profile preprocessing pipeline for the AI Candidate Ranking System."
    )
    parser.add_argument(
        "--candidates", 
        type=str, 
        default="data/sample_candidates.json", 
        help="Path to candidates dataset (JSONL or JSON format)."
    )
    parser.add_argument(
        "--schema", 
        type=str, 
        default="data/candidate_schema.json", 
        help="Path to candidate JSON Schema definition."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="Directory to save processed Parquet, DLQ, and metadata outputs."
    )
    parser.add_argument(
        "--reference-date", 
        type=str, 
        default="2026-06-30", 
        help="Reference date for parsing ongoing positions (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Chunk size/batch size for pipeline streaming."
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None, 
        help="Maximum worker processes (defaults to CPU count limits)."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=2.0, 
        help="Percentage limit of acceptable records failure before aborting execution."
    )

    args = parser.parse_args()

    # Load candidate schema if provided
    schema_dict = None
    if os.path.exists(args.schema):
        try:
            with open(args.schema, "r", encoding="utf-8") as f:
                schema_dict = json.load(f)
            print(f"Loaded schema definition from {args.schema}")
        except Exception as e:
            print(f"Warning: Failed to load schema from {args.schema}: {str(e)}. Falling back to manual structural checks.")
    else:
        print(f"Warning: Schema file not found at {args.schema}. Programmatic structural validator will run.")

    # Resolve candidates path (support JSON file or JSONL streaming)
    # The CandidateReader handles JSONL format. If the user passes sample_candidates.json (which is a standard JSON array),
    # we convert it to a temporary JSONL file dynamically so the streaming reader can parse it consistently.
    candidates_path = args.candidates
    temp_jsonl_created = False
    
    if candidates_path.endswith(".json") and not candidates_path.endswith(".jsonl"):
        print(f"Input is a standard JSON array file ({candidates_path}). Converting to temporary JSONL for streaming reader...")
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print("Error: Input JSON file must contain an array of candidates.")
                sys.exit(1)
                
            temp_jsonl_path = candidates_path + "l" # Convert .json -> .jsonl
            with open(temp_jsonl_path, "w", encoding="utf-8") as f:
                for record in data:
                    f.write(json.dumps(record) + "\n")
            candidates_path = temp_jsonl_path
            temp_jsonl_created = True
        except Exception as e:
            print(f"Error converting JSON to JSONL: {str(e)}")
            sys.exit(1)

    print(f"Initializing Preprocessing Pipeline on {candidates_path}...")
    print(f"Reference date: {args.reference_date}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")

    try:
        pipeline = PreprocessingPipeline(
            schema_dict=schema_dict,
            reference_date_str=args.reference_date,
            batch_size=args.batch_size,
            max_workers=args.workers,
            failure_threshold_percentage=args.threshold
        )
        
        report = pipeline.run(candidates_path, args.output_dir)
        
        print("\nPipeline run completed successfully!")
        print("----------------------------------------")
        print(json.dumps(report, indent=2))
        print("----------------------------------------")
        print(f"Output written to: {args.output_dir}/")
        
    except Exception as e:
        print(f"\nPipeline run failed with error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Cleanup temporary JSONL file if we created one
        if temp_jsonl_created and os.path.exists(candidates_path):
            try:
                os.remove(candidates_path)
            except OSError:
                pass

if __name__ == "__main__":
    main()
