from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseNormalizer(ABC):
    """Abstract base class representing a normalization step in the preprocessing pipeline."""
    
    @abstractmethod
    def normalize(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Performs in-place normalization or returns a normalized copy of candidate_data.
        
        Args:
            candidate_data: Dictionary representing candidate record.
            
        Returns:
            Dict[str, Any]: The processed candidate record dictionary.
        """
        pass
