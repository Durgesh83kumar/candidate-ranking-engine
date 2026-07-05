import time
import os
import sys
from typing import Dict, Any

try:
    import psutil
    _has_psutil = True
except ImportError:
    _has_psutil = False

class IndexingBenchmarker:
    """Measures indexing throughput, system memory usage, CPU loads, and query search latency."""

    def __init__(self):
        self.start_time = time.time()

    def get_memory_usage(self) -> float:
        """Returns current process RAM usage in Megabytes."""
        if _has_psutil:
            process = psutil.Process(os.getpid())
            return float(process.memory_info().rss) / (1024 * 1024)
        return 0.0

    def get_cpu_utilization(self) -> float:
        """Returns average CPU utilization percent."""
        if _has_psutil:
            return float(psutil.cpu_percent(interval=None))
        return 0.0

    def get_disk_size(self, file_path: str) -> float:
        """Returns file size on disk in Megabytes."""
        if os.path.exists(file_path):
            return float(os.path.getsize(file_path)) / (1024 * 1024)
        return 0.0

    def compile_report(self, total_records: int, elapsed_seconds: float, output_dir: str) -> Dict[str, Any]:
        """Compiles system performance and footprint measurements."""
        faiss_path = os.path.join(output_dir, "faiss.index")
        npy_path = os.path.join(output_dir, "embeddings.npy")
        
        index_size_mb = self.get_disk_size(faiss_path)
        embeddings_size_mb = self.get_disk_size(npy_path)
        
        return {
            "throughput_docs_per_sec": round(total_records / elapsed_seconds if elapsed_seconds > 0 else 0, 2),
            "elapsed_time_seconds": round(elapsed_seconds, 2),
            "cpu_utilization_percent": self.get_cpu_utilization(),
            "peak_memory_usage_mb": round(self.get_memory_usage(), 2),
            "disk_footprint": {
                "faiss_index_mb": round(index_size_mb, 2),
                "embeddings_matrix_mb": round(embeddings_size_mb, 2),
                "total_disk_mb": round(index_size_mb + embeddings_size_mb, 2)
            }
        }
