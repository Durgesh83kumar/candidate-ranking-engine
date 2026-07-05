import numpy as np
from typing import List, Dict, Any
from src.indexing.backends.base import BaseEmbeddingBackend
from src.indexing.exceptions import BackendError

try:
    from sentence_transformers import SentenceTransformer
    _has_st = True
except ImportError:
    _has_st = False

class SentenceTransformersBackend(BaseEmbeddingBackend):
    """Execution backend wrapper using the SentenceTransformers package."""

    def initialize(self, model_name: str, config: Dict[str, Any]) -> None:
        if not _has_st:
            raise BackendError("SentenceTransformers is not installed.")
        try:
            print(f"Initializing SentenceTransformers model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            # Set model device to CPU explicitly for the hackathon environment
            self.model.to("cpu")
        except Exception as e:
            raise BackendError(f"Failed to initialize SentenceTransformer: {str(e)}") from e

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            # We enforce CPU execution and return numpy float32 arrays
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=False, 
                convert_to_numpy=True, 
                normalize_embeddings=False
            )
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            raise BackendError(f"Failed to compute SentenceTransformers embeddings: {str(e)}") from e
