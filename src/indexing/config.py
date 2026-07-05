import os
from typing import Dict, Any

class IndexingConfig:
    """Central configuration repository for the embedding and FAISS indexing pipeline."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "BAAI/bge-small-en-v1.5")
        self.backend = kwargs.get("backend", "sentence_transformers")  # Fallback friendly default
        self.batch_size = int(kwargs.get("batch_size", 256))
        self.document_version = kwargs.get("document_version", "search_document_v2")
        self.normalize = bool(kwargs.get("normalize", True))
        self.index_type = kwargs.get("index_type", "IndexFlatIP")
        self.cache_enabled = bool(kwargs.get("cache_enabled", True))
        self.eval_enabled = bool(kwargs.get("eval_enabled", True))
        self.benchmark_enabled = bool(kwargs.get("benchmark_enabled", True))
        self.top_k = int(kwargs.get("top_k", 100))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "batch_size": self.batch_size,
            "document_version": self.document_version,
            "normalize": self.normalize,
            "index_type": self.index_type,
            "cache_enabled": self.cache_enabled,
            "eval_enabled": self.eval_enabled,
            "benchmark_enabled": self.benchmark_enabled,
            "top_k": self.top_k
        }
