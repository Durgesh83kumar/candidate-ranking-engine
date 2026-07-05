import time
import os
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.preprocessing.exceptions import PipelineFailureThresholdExceeded
from src.preprocessing.reader import CandidateReader
from src.preprocessing.validator import CandidateSchemaValidator
from src.preprocessing.normalizers.text import TextNormalizer
from src.preprocessing.normalizers.skills import SkillNormalizer
from src.preprocessing.normalizers.dates import DateNormalizer
from src.preprocessing.enrichment.experience import ExperienceCalculator
from src.preprocessing.enrichment.text_aggregator import TextAggregator
from src.preprocessing.deduplicator import CandidateDeduplicator
from src.preprocessing.writer import CandidateWriter

# Module-level worker function to enable process pickling for ProcessPoolExecutor
def process_candidate_record(
    raw_record: Dict[str, Any], 
    schema_dict: Optional[Dict[str, Any]], 
    reference_date_str: str
) -> tuple:
    """Processes a single candidate record: validation, normalization, and enrichment.
    
    Returns:
        tuple: (is_success: bool, processed_or_raw_record: dict, error_message: str/None)
    """
    try:
        # 1. Ingestion / Schema Structural Validation
        validator = CandidateSchemaValidator(schema_dict)
        try:
            if validator.schema_dict:
                import jsonschema
                jsonschema.validate(instance=raw_record, schema=validator.schema_dict)
            else:
                validator._fallback_schema_validate(raw_record)
        except Exception as e:
            return False, raw_record, f"Schema validation error: {str(e)}"

        # Copy data to avoid side-effects across process boundaries
        candidate = dict(raw_record)

        # 2. Normalizations (Cleans and corrects inverted values)
        text_norm = TextNormalizer()
        skills_norm = SkillNormalizer()
        dates_norm = DateNormalizer(reference_date_str)
        
        candidate = text_norm.normalize(candidate)
        candidate = skills_norm.normalize(candidate)
        candidate = dates_norm.normalize(candidate)

        # 3. Custom Semantic Rules Validation (Evaluated on clean data)
        try:
            validator.validate_custom_rules(candidate)
        except CustomRuleValidationError as e:
            return False, raw_record, f"Custom rule validation error: {str(e)}"

        # 4. Enrichment
        exp_calc = ExperienceCalculator()
        aggregator = TextAggregator()
        
        candidate = exp_calc.calculate_experience_metrics(candidate)
        candidate = aggregator.build_search_documents(candidate)

        return True, candidate, None
    except Exception as e:
        return False, raw_record, f"Processing exception: {str(e)}"


class PreprocessingPipeline:
    """Orchestrates ingestion, parallel transformation, deduplication, and writing for candidate profiles."""

    def __init__(
        self,
        schema_dict: Optional[Dict[str, Any]] = None,
        reference_date_str: str = "2026-06-30",
        batch_size: int = 5000,
        max_workers: Optional[int] = None,
        failure_threshold_percentage: float = 2.0
    ):
        self.schema_dict = schema_dict
        self.reference_date_str = reference_date_str
        self.batch_size = batch_size
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self.failure_threshold_percentage = failure_threshold_percentage

    def run(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """Runs the preprocessing pipeline on candidate dataset.
        
        Returns:
            Dict[str, Any]: Execution metadata report.
        """
        start_time = time.time()
        
        # Instantiate helper modules
        reader = CandidateReader(input_path, self.batch_size)
        writer = CandidateWriter(output_dir)
        deduplicator = CandidateDeduplicator()

        total_ingested = 0
        total_valid = 0
        total_failed = 0
        total_duplicates = 0
        unique_skills_set = set()

        # We process the dataset batch by batch to be memory-safe
        for batch_index, raw_batch in enumerate(reader.stream_raw_candidates(), 1):
            batch_ingested = len(raw_batch)
            total_ingested += batch_ingested

            processed_batch_results = []
            
            # Execute batch records in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        process_candidate_record, 
                        record, 
                        self.schema_dict, 
                        self.reference_date_str
                    ): record 
                    for record in raw_batch
                }

                for future in as_completed(futures):
                    is_success, record, err_msg = future.result()
                    if is_success:
                        # Deduplication check (run sequentially in main thread to keep set synchronized)
                        is_dup = deduplicator.track_and_check_duplicate(record)
                        if is_dup:
                            total_duplicates += 1
                        else:
                            processed_batch_results.append(record)
                            # Record skill list metrics for metadata report
                            for skill in record.get("skills", []):
                                unique_skills_set.add(skill.get("name_normalized"))
                    else:
                        total_failed += 1
                        writer.write_to_dlq(record, err_msg)

            # Write the unique processed batch to the Parquet/JSONL output
            if processed_batch_results:
                writer.write_batch(processed_batch_results)
                total_valid += len(processed_batch_results)

            # Enforce Failure Threshold Check
            failure_rate = (total_failed / total_ingested) * 100.0 if total_ingested > 0 else 0.0
            if failure_rate > self.failure_threshold_percentage and total_ingested >= 1000:
                raise PipelineFailureThresholdExceeded(
                    f"Pipeline aborted: Failure rate of {failure_rate:.2f}% exceeded threshold limit "
                    f"of {self.failure_threshold_percentage}% after ingesting {total_ingested} records."
                )

        # Build final audit report
        elapsed_time = time.time() - start_time
        avg_rate = total_ingested / elapsed_time if elapsed_time > 0 else 0.0

        report = {
            "run_id": f"run_{int(start_time)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "total_records_ingested": total_ingested,
            "valid_records_processed": total_valid,
            "failed_records_to_dlq": total_failed,
            "duplicate_records_dropped": total_duplicates,
            "elapsed_time_seconds": round(elapsed_time, 2),
            "average_processing_rate_cps": round(avg_rate, 2),
            "unique_skills_discovered": len(unique_skills_set)
        }

        writer.write_metadata_report(report)
        return report
