import os
import json
import argparse
import sys
from typing import Dict, Any, List

from src.jd_intelligence.parser import JDParser
from src.jd_intelligence.extractor import LlmExtractor
from src.jd_intelligence.validator import SpecificationValidator
from src.jd_intelligence.query_gen import QueryGenerator

def run_pipeline(jd_path: str, output_dir: str) -> None:
    """Executes the Job Description Intelligence pipeline and saves structural JSON deliverables."""
    print(f"Starting Job Description Intelligence pipeline on {jd_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Parse Document Text
    parser = JDParser()
    text = parser.parse(jd_path)
    print(f"Extracted {len(text)} characters from {jd_path}.")

    # 2. Extract Requirements
    extractor = LlmExtractor()
    spec = extractor.extract(text)
    print(f"Successfully extracted structured Hiring Specification.")

    # 3. Validate Requirements
    validator = SpecificationValidator()
    warnings = validator.validate(spec)
    print(f"Validation completed with {len(warnings)} warnings.")

    # 4. Generate Retrieval Queries
    query_generator = QueryGenerator()
    queries = query_generator.generate(spec)
    print(f"Generated optimized retrieval search queries.")

    # 5. Compile and serialize outputs
    spec_dict = spec.model_dump()
    
    # Extract hierarchical taxonomy subset
    taxonomy_data = {
        "AI/ML": {
            "required_domains": spec_dict.get("domains", {}).get("required", []),
            "preferred_domains": spec_dict.get("domains", {}).get("preferred", [])
        },
        "Programming": {
            "must_have_skills": [s["name"] for s in spec_dict.get("skills", {}).get("must_have", [])],
            "preferred_skills": [s["name"] for s in spec_dict.get("skills", {}).get("preferred", [])]
        }
    }

    # Format deliverables
    hiring_spec_path = os.path.join(output_dir, "hiring_specification.json")
    queries_path = os.path.join(output_dir, "search_queries.json")
    priorities_path = os.path.join(output_dir, "ranking_priorities.json")
    taxonomy_path = os.path.join(output_dir, "technology_taxonomy.json")
    concepts_path = os.path.join(output_dir, "semantic_concepts.json")
    validation_path = os.path.join(output_dir, "validation_report.json")

    # Save hiring_specification.json
    with open(hiring_spec_path, "w", encoding="utf-8") as f:
        json.dump(spec_dict, f, indent=2)

    # Save search_queries.json
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)

    # Save ranking_priorities.json
    priorities = [p.model_dump() for p in spec.ranking_priorities]
    with open(priorities_path, "w", encoding="utf-8") as f:
        json.dump(priorities, f, indent=2)

    # Save technology_taxonomy.json
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy_data, f, indent=2)

    # Save semantic_concepts.json
    concepts = spec.search_parameters.semantic_concepts
    with open(concepts_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2)

    # Save validation_report.json
    validation_report = {
        "status": "APPROVED" if not warnings else "WARNING",
        "warnings": warnings,
        "errors": []
    }
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    print("\nJD Intelligence Pipeline Completed successfully!")
    print("--------------------------------------------------")
    print(f"Hiring Spec: {hiring_spec_path}")
    print(f"Queries:     {queries_path}")
    print(f"Validation:  {validation_path}")
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Run Job Description Intelligence pipeline."
    )
    parser.add_argument(
        "--jd", 
        type=str, 
        default="data/job_description.docx",
        help="Path to job description file."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory to save JSON deliverables."
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(args.jd, args.output_dir)
    except Exception as e:
        print(f"Pipeline error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
