import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from src.indexing.backends import get_backend
from src.indexing.exceptions import FAISSError
from src.indexing.faiss_index import FAISSIndexBuilder

class VectorSearcher:
    """Retrieval query executor. Encapsulates query embedding, FAISS similarity search, and candidate mapping."""

    def __init__(self, backend_name: str = "sentence_transformers", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.backend_name = backend_name
        self.model_name = model_name
        self.backend = None
        self.index_builder = None
        self.candidate_ids = []

    def load_index(self, index_path: str, metadata_path: str) -> None:
        """Loads FAISS index and candidate ID map file from disk."""
        # 1. Load FAISS index
        self.index_builder = FAISSIndexBuilder()
        self.index_builder.load_index(index_path)

        # 2. Load candidate ID mappings list
        if not os.path.exists(metadata_path):
            raise FAISSError(f"Index mapping file not found: {metadata_path}")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Map key to list of IDs
                if isinstance(data, dict):
                    self.candidate_ids = data.get("candidate_ids", [])
                elif isinstance(data, list):
                    # Fallback if stored as a list
                    self.candidate_ids = [item.get("candidate_id") for item in data]
        except Exception as e:
            raise FAISSError(f"Failed to read index mapping metadata: {str(e)}") from e

    def initialize_backend(self) -> None:
        """Dynamically loads and initializes the pluggable embedding backend for query vectorization."""
        if self.backend is None:
            self.backend = get_backend(self.backend_name)
            self.backend.initialize(self.model_name, {})

    def embed_query(self, query_text: str) -> np.ndarray:
        """Embeds and L2 normalizes a query string into a unit vector."""
        self.initialize_backend()
        
        # In E5 models, query inputs must start with the prefix "query: "
        formatted_query = query_text
        if "e5" in self.model_name.lower():
            formatted_query = f"query: {query_text}"

        raw_vector = self.backend.compute_embeddings([formatted_query])
        
        # L2 normalize
        norms = np.linalg.norm(raw_vector, axis=1, keepdims=True)
        normalized = raw_vector / np.clip(norms, a_min=1e-9, a_max=None)
        return normalized.astype(np.float32)

    def retrieve_top_k(self, query_vector: np.ndarray, k: int = 100) -> List[Dict[str, Any]]:
        """Performs flat inner product search on the FAISS index.
        
        Returns:
            List[Dict[str, Any]]: List of matching records containing candidate_id and score.
        """
        if self.index_builder is None or self.index_builder.index is None:
            raise FAISSError("FAISS index is not loaded. Call load_index() first.")

        # FAISS search expects float32 shape (1, dimension)
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # Search index
        scores, indices = self.index_builder.index.search(query_vector, k)
        
        results = []
        # scores and indices are arrays of shape (1, k)
        for score, idx in zip(scores[0], indices[0]):
            # -1 indicates no match found in small indexes
            if idx == -1 or idx >= len(self.candidate_ids):
                continue
            results.append({
                "candidate_id": self.candidate_ids[idx],
                "score": float(score)
            })
            
        return results
