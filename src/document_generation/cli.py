import argparse
import sys
from src.document_generation.pipeline import DocumentGenerationPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run Candidate Search Document Generation pipeline."
    )
    parser.add_argument(
        "--preprocessed",
        type=str,
        default="output/processed_candidates.parquet",
        help="Path to preprocessed candidates parquet file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write search document tables."
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=1024,
        help=" Capping token limit budget for comprehensive documents."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Chunk size of streaming processes."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of CPU subprocess workers."
    )

    args = parser.parse_args()

    pipeline = DocumentGenerationPipeline(
        target_token_limit=args.token_limit,
        batch_size=args.batch_size,
        max_workers=args.workers
    )

    try:
        report = pipeline.run(args.preprocessed, args.output_dir)
        print("\nDocument Generation Pipeline Completed Successfully!")
        print("--------------------------------------------------")
        print(f"Total candidates:  {report['total_candidates_ingested']}")
        print(f"Total processed:   {report['total_documents_generated']}")
        print(f"Elapsed time (s):  {report['elapsed_time_seconds']}")
        print(f"Generation rate:   {report['generation_rate_cps']} cps")
        print("--------------------------------------------------")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
