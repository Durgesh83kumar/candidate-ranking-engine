import numpy as np
from src.indexing.exceptions import FAISSError

try:
    import faiss
    _has_faiss = True
except ImportError:
    _has_faiss = False

class FAISSIndexBuilder:
    """Manages the creation, normalization, population, and serialization of the FAISS Flat IP index."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        if not _has_faiss:
            raise FAISSError("FAISS is not installed. Please run pip install faiss-cpu.")
        self.index = faiss.IndexFlatIP(dimension)

    def normalize_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies L2 normalization to project vectors to a unit sphere (norm = 1.0)."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        return normalized.astype(np.float32)

    def build_and_populate(self, embeddings: np.ndarray) -> None:
        """Normalizes the input matrix and adds it to the IndexFlatIP index."""
        try:
            num_vecs, dim = embeddings.shape
            if dim != self.dimension:
                raise FAISSError(f"Embedding dimension {dim} does not match expected index dimension {self.dimension}.")
                
            print(f"Populating FAISS index with {num_vecs} vectors...")
            normalized = self.normalize_vectors(embeddings)
            
            # Reset index to clean state
            self.index.reset()
            self.index.add(normalized)
        except Exception as e:
            raise FAISSError(f"Failed to populate FAISS index: {str(e)}") from e

    def save_index(self, file_path: str) -> None:
        """Serializes the index to disk."""
        try:
            faiss.write_index(self.index, file_path)
            print(f"FAISS index successfully saved to {file_path}.")
        except Exception as e:
            raise FAISSError(f"Failed to write FAISS index to {file_path}: {str(e)}") from e

    def load_index(self, file_path: str) -> None:
        """Loads a FAISS index from disk."""
        try:
            self.index = faiss.read_index(file_path)
            self.dimension = self.index.d
            print(f"FAISS index loaded from {file_path} (dimension={self.dimension}, total={self.index.ntotal}).")
        except Exception as e:
            raise FAISSError(f"Failed to read FAISS index from {file_path}: {str(e)}") from e
