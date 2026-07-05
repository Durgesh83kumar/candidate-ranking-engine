class VerificationError(Exception):
    """Base exception class for all verification engine errors."""
    pass

class PhraseScannerError(VerificationError):
    """Raised when regex phrase parsing fails."""
    pass

class SkillsValidationError(VerificationError):
    """Raised when mandatory skills checking fails."""
    pass

class AISpecialistError(VerificationError):
    """Raised when AI framework checks fail."""
    pass

class VerificationWarning(VerificationError):
    """Raised as a warning when quality validations fail."""
    pass
