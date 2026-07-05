import json
from typing import Generator, Dict, Any, List
from src.preprocessing.exceptions import IngestionError

class CandidateReader:
    """Streams candidate records from a JSONL file in memory-safe batches."""
    
    def __init__(self, file_path: str, batch_size: int = 5000):
        self.file_path = file_path
        self.batch_size = batch_size

    def stream_raw_candidates(self) -> Generator[List[Dict[str, Any]], None, None]:
        """Lazily reads lines from the JSONL file and yields them in batches."""
        batch = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        batch.append(record)
                    except json.JSONDecodeError as e:
                        raise IngestionError(
                            f"Malformed JSON on line {line_num} of file {self.file_path}: {str(e)}"
                        ) from e
                    
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                
                # Yield remaining records
                if batch:
                    yield batch
        except FileNotFoundError as e:
            raise IngestionError(f"Candidate file not found: {self.file_path}") from e
        except IOError as e:
            raise IngestionError(f"I/O error reading candidate file: {str(e)}") from e
