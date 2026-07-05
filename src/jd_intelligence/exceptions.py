class JDError(Exception):
    """Base exception for Job Description Intelligence Module."""
    pass

class JDParserError(JDError):
    """Raised when reading or parsing the JD document fails."""
    pass

class JDExtractorError(JDError):
    """Raised when structured requirements extraction fails."""
    pass

class JDValidationError(JDError):
    """Raised when structured requirements fail consistency checks."""
    pass
