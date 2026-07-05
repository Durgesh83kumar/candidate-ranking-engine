import os
import json
from typing import List, Dict, Any
from src.preprocessing.exceptions import PreprocessingError

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    _has_parquet_libs = True
except ImportError:
    _has_parquet_libs = False

class CandidateWriter:
    """Manages serialization of processed candidates to Parquet/JSONL and error routing to Dead-Letter Queue (DLQ)."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.dlq_path = os.path.join(output_dir, "dlq_candidates.jsonl")
        self.parquet_path = os.path.join(output_dir, "processed_candidates.parquet")
        self.jsonl_fallback_path = os.path.join(output_dir, "processed_candidates.jsonl")
        
        # Clean existing output files on initialization
        for path in [self.dlq_path, self.parquet_path, self.jsonl_fallback_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def write_batch(self, processed_batch: List[Dict[str, Any]]) -> None:
        """Saves a batch of processed candidate dicts to disk, prioritizing Parquet."""
        if not processed_batch:
            return

        if _has_parquet_libs:
            try:
                # Convert the list of dicts to an Arrow Table
                # Since nested structures are present, pyarrow handles them directly
                table = pa.Table.from_pylist(processed_batch)
                
                # Write/append to Parquet file
                if os.path.exists(self.parquet_path):
                    # For simple appending in local dev, we read existing and write combined, or use a parquet writer
                    # To keep it clean and performant, we initialize a ParquetWriter
                    existing_table = pq.read_table(self.parquet_path)
                    combined_table = pa.concat_tables([existing_table, table])
                    pq.write_table(combined_table, self.parquet_path, compression="snappy")
                else:
                    pq.write_table(table, self.parquet_path, compression="snappy")
                return
            except Exception as e:
                # If Parquet write fails, fall back to JSONL
                pass
                
        # Fallback JSONL logging
        try:
            with open(self.jsonl_fallback_path, "a", encoding="utf-8") as f:
                for record in processed_batch:
                    # Strip any non-serializable parsed date objects
                    cleaned_record = self._sanitize_record_for_json(record)
                    f.write(json.dumps(cleaned_record) + "\n")
        except IOError as e:
            raise PreprocessingError(f"Failed to write fallback JSONL processed batch: {str(e)}") from e

    def write_to_dlq(self, raw_candidate: Dict[str, Any], error_msg: str) -> None:
        """Appends a corrupted or invalid raw candidate record to the DLQ JSONL file."""
        dlq_entry = {
            "raw_candidate": raw_candidate,
            "preprocessing_error": error_msg
        }
        try:
            with open(self.dlq_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._sanitize_record_for_json(dlq_entry)) + "\n")
        except IOError as e:
            raise PreprocessingError(f"Failed to write to Dead-Letter Queue file: {str(e)}") from e

    def write_metadata_report(self, report: Dict[str, Any]) -> None:
        """Saves the runtime execution and metric report as a JSON file."""
        report_path = os.path.join(self.output_dir, "preprocessing_metadata.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except IOError as e:
            raise PreprocessingError(f"Failed to write metadata report: {str(e)}") from e

    def _sanitize_record_for_json(self, data: Any) -> Any:
        """Recursively removes datetime/date objects or serializes them to strings for JSON safety."""
        if isinstance(data, dict):
            return {k: self._sanitize_record_for_json(v) for k, v in data.items() if not k.endswith("_parsed")}
        elif isinstance(data, list):
            return [self._sanitize_record_for_json(item) for item in data]
        elif hasattr(data, "isoformat"):
            return data.isoformat()
        return data
