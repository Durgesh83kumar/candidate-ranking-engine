class DocumentGenerationError(Exception):
    """Base exception for all document generation module errors."""
    pass

class CleaningError(DocumentGenerationError):
    """Raised when text cleaning or noise filtering fails."""
    pass

class TemplatingError(DocumentGenerationError):
    """Raised when formatting templates or maps fail."""
    pass

class CompactionError(DocumentGenerationError):
    """Raised when summarizing or context window capping fails."""
    pass

class WriterError(DocumentGenerationError):
    """Raised when writing Parquet or JSON documents fails."""
    pass
