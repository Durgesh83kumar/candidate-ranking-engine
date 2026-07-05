class RerankerError(Exception):
    """Base exception for all reranker module errors."""
    pass

class ModelLoadError(RerankerError):
    """Raised when Cross-Encoder model loading fails."""
    pass

class CalibrationError(RerankerError):
    """Raised when scoring or probability calibration fails."""
    pass
