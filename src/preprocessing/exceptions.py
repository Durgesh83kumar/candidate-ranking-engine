class PreprocessingError(Exception):
    """Base exception for all preprocessing pipeline errors."""
    pass

class SchemaValidationError(PreprocessingError):
    """Raised when candidate record does not match the JSON Schema."""
    pass

class CustomRuleValidationError(PreprocessingError):
    """Raised when custom semantic or business rule validation fails."""
    pass

class DateNormalizationError(PreprocessingError):
    """Raised when date parsing or normalizations fail."""
    pass

class SkillNormalizationError(PreprocessingError):
    """Raised when skill mappings or normalization fails."""
    pass

class ExperienceCalculationError(PreprocessingError):
    """Raised when experience calculation fails."""
    pass

class IngestionError(PreprocessingError):
    """Raised when reading JSONL dataset fails."""
    pass

class PipelineFailureThresholdExceeded(PreprocessingError):
    """Raised when the number of failed records exceeds the allowable threshold."""
    pass
