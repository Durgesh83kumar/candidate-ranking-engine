class RetrievalError(Exception):
    """Base exception for all retrieval module errors."""
    pass

class FilterError(RetrievalError):
    """Raised when soft or hard filter parsing/evaluation fails."""
    pass

class FusionError(RetrievalError):
    """Raised when reciprocal rank fusion computations fail."""
    pass

class DeduplicationError(RetrievalError):
    """Raised when candidate record aggregation or deduplication fails."""
    pass
