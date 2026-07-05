class IndexingError(Exception):
    """Base exception for all indexing module errors."""
    pass

class BackendError(IndexingError):
    """Raised when embedding backend initialization or inference fails."""
    pass

class CacheError(IndexingError):
    """Raised when loading or writing embedding cache fails."""
    pass

class EvaluationError(IndexingError):
    """Raised when metrics evaluation or quality check fails."""
    pass

class FAISSError(IndexingError):
    """Raised when building or querying the FAISS index fails."""
    pass
