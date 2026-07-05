import torch
from sentence_transformers import CrossEncoder
from src.reranker.exceptions import ModelLoadError

class CrossEncoderModelRegistry:
    """Manages Cross-Encoder transformer model sessions, ensuring single-instance loading on CPU."""

    _model_instance = None
    _loaded_model_name = None

    @classmethod
    def get_model(cls, model_name: str, fallback_model_name: str, device: str = "cpu") -> CrossEncoder:
        """Loads and returns the Cross-Encoder model instance. Falls back if primary model fails.
        
        Args:
            model_name: Primary Hugging Face model repository.
            fallback_model_name: Fallback model repository.
            device: Execution hardware context (forced to CPU for hackathon).
            
        Returns:
            CrossEncoder: Shared Cross-Encoder instance.
        """
        # Load only once
        if cls._model_instance is not None:
            return cls._model_instance

        # Forced CPU constraints
        device_context = "cpu"
        
        # 1. Attempt Primary Model
        try:
            print(f"Loading primary Cross-Encoder model: {model_name}...")
            cls._model_instance = CrossEncoder(model_name, device=device_context)
            cls._loaded_model_name = model_name
            print("Primary Cross-Encoder successfully loaded.")
            return cls._model_instance
        except Exception as e:
            print(f"Warning: Failed to load primary model '{model_name}' due to error: {str(e)}.")
            
        # 2. Attempt Fallback Model
        try:
            print(f"Loading fallback Cross-Encoder model: {fallback_model_name}...")
            cls._model_instance = CrossEncoder(fallback_model_name, device=device_context)
            cls._loaded_model_name = fallback_model_name
            print("Fallback Cross-Encoder successfully loaded.")
            return cls._model_instance
        except Exception as e:
            raise ModelLoadError(
                f"Critical Error: Failed to load both primary '{model_name}' and fallback "
                f"'{fallback_model_name}' Cross-Encoder models. Details: {str(e)}"
            ) from e
