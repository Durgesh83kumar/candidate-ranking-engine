from typing import Dict, Any

class RerankerConfig:
    """Manages Cross-Encoder re-ranking configurations and hyperparameters."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "BAAI/bge-reranker-base")
        self.fallback_model_name = kwargs.get("fallback_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.max_length = int(kwargs.get("max_length", 512))
        self.device = kwargs.get("device", "cpu")
        self.top_candidates = int(kwargs.get("top_candidates", 300))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "fallback_model_name": self.fallback_model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "top_candidates": self.top_candidates
        }
