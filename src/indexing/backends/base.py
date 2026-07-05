import numpy as np
from typing import List, Dict, Any

class BaseEmbeddingBackend:
    """Abstract base class representing interchangeable embedding computation engines."""

    def initialize(self, model_name: str, config: Dict[str, Any]) -> None:
        """Initializes model weights, tokenizers, and execution context."""
        raise NotImplementedError

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Computes dense embeddings for list of documents.
        
        Returns:
            np.ndarray: float32 matrix of dimensions (num_texts, dimension_size).
        """
        raise NotImplementedError
