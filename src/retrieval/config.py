from typing import Dict, Any

class RetrievalConfig:
    """Manages system configuration and threshold parameters for Semantic Candidate Retrieval."""

    def __init__(self, **kwargs):
        self.top_k_per_query = int(kwargs.get("top_k_per_query", 200))
        self.fusion_constant = int(kwargs.get("fusion_constant", 60))
        self.minimum_similarity = float(kwargs.get("minimum_similarity", 0.0))
        self.batch_size = int(kwargs.get("batch_size", 256))
        self.enable_filters = bool(kwargs.get("enable_filters", True))
        self.enable_rrf = bool(kwargs.get("enable_rrf", True))
        self.output_pool_size = int(kwargs.get("output_pool_size", 1000))
        
        # Soft recruiter preferences default
        self.filters = kwargs.get("filters", {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_k_per_query": self.top_k_per_query,
            "fusion_constant": self.fusion_constant,
            "minimum_similarity": self.minimum_similarity,
            "batch_size": self.batch_size,
            "enable_filters": self.enable_filters,
            "enable_rrf": self.enable_rrf,
            "output_pool_size": self.output_pool_size,
            "filters": self.filters
        }
