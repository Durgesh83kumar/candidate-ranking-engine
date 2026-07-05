import os
import hashlib
import numpy as np
from typing import Dict, Any, Optional
from src.indexing.exceptions import CacheError

class EmbeddingCache:
    """Manages disk-based key-value caching of embedding vectors using SHA-256 hashes."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def generate_hash(self, text: str, model_name: str, version: str) -> str:
        """Computes a unique SHA-256 cache key for the given text, model, and version."""
        if not text:
            text = ""
        payload = f"{text.strip()}||{model_name}||{version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, hash_key: str) -> Optional[np.ndarray]:
        """Looks up the hash key in the cache folder.
        
        Returns:
            np.ndarray: float32 vector or None if cache miss.
        """
        path = os.path.join(self.cache_dir, f"{hash_key}.npy")
        if os.path.exists(path):
            try:
                return np.load(path)
            except Exception as e:
                # Log warning and return None for safety
                return None
        return None

    def set(self, hash_key: str, vector: np.ndarray) -> None:
        """Stores the vector in the cache directory."""
        path = os.path.join(self.cache_dir, f"{hash_key}.npy")
        try:
            # Save raw numpy array
            np.save(path, vector.astype(np.float32))
        except Exception as e:
            raise CacheError(f"Failed to write vector to embedding cache: {str(e)}") from e
