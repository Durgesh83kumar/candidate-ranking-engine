class RankingError(Exception):
    """Base exception class for all ranking engine errors."""
    pass

class ScorerError(RankingError):
    """Raised when the hybrid score calculation fails."""
    pass

class ValidationError(RankingError):
    """Raised when the output submission schema or monotonicity validation fails."""
    pass
