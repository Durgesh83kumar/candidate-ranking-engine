import os
import json
from typing import List, Dict, Any
from src.document_generation.exceptions import WriterError

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    _has_parquet_libs = True
except ImportError:
    _has_parquet_libs = False

class DocumentWriter:
    """Handles file serialization of generated search documents and separate metadata records to Parquet/JSONL."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.docs_parquet_path = os.path.join(output_dir, "search_documents.parquet")
        self.meta_parquet_path = os.path.join(output_dir, "candidate_metadata.parquet")
        
        self.docs_jsonl_path = os.path.join(output_dir, "search_documents.jsonl")
        self.meta_jsonl_path = os.path.join(output_dir, "candidate_metadata.jsonl")

        # Clean old files on initialization
        for path in [self.docs_parquet_path, self.meta_parquet_path, self.docs_jsonl_path, self.meta_jsonl_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def write_batch(self, docs_batch: List[Dict[str, Any]], meta_batch: List[Dict[str, Any]]) -> None:
        """Writes candidate search documents and metadata tables in batches."""
        if not docs_batch or not meta_batch:
            return

        if _has_parquet_libs:
            try:
                # 1. Write Search Documents Parquet
                docs_table = pa.Table.from_pylist(docs_batch)
                if os.path.exists(self.docs_parquet_path):
                    existing = pq.read_table(self.docs_parquet_path)
                    combined = pa.concat_tables([existing, docs_table])
                    pq.write_table(combined, self.docs_parquet_path, compression="snappy")
                else:
                    pq.write_table(docs_table, self.docs_parquet_path, compression="snappy")

                # 2. Write Candidate Metadata Parquet
                meta_table = pa.Table.from_pylist(meta_batch)
                if os.path.exists(self.meta_parquet_path):
                    existing = pq.read_table(self.meta_parquet_path)
                    combined = pa.concat_tables([existing, meta_table])
                    pq.write_table(combined, self.meta_parquet_path, compression="snappy")
                else:
                    pq.write_table(meta_table, self.meta_parquet_path, compression="snappy")
                return
            except Exception as e:
                # Fallback to JSONL on failure
                pass

        # Fallback JSONL Serialization
        try:
            with open(self.docs_jsonl_path, "a", encoding="utf-8") as f:
                for doc in docs_batch:
                    f.write(json.dumps(doc) + "\n")
            with open(self.meta_jsonl_path, "a", encoding="utf-8") as f:
                for meta in meta_batch:
                    f.write(json.dumps(meta) + "\n")
        except IOError as e:
            raise WriterError(f"Failed to write fallback JSONL search documents: {str(e)}") from e

    def write_report(self, report: Dict[str, Any], report_name: str) -> None:
        """Saves JSON execution reports (e.g. generation_report.json, document_versions.json)."""
        path = os.path.join(self.output_dir, report_name)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except IOError as e:
            raise WriterError(f"Failed to write report {report_name}: {str(e)}") from e
