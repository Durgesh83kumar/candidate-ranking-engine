import numpy as np
from typing import List, Dict, Any
from src.indexing.backends.base import BaseEmbeddingBackend
from src.indexing.exceptions import BackendError

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _has_torch_transformers = True
except ImportError:
    _has_torch_transformers = False

class HFTransformersBackend(BaseEmbeddingBackend):
    """Raw HuggingFace Transformers + PyTorch CPU execution backend with mean pooling."""

    def initialize(self, model_name: str, config: Dict[str, Any]) -> None:
        if not _has_torch_transformers:
            raise BackendError("PyTorch or HuggingFace Transformers is not installed.")
        try:
            print(f"Initializing HF PyTorch model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            
            # Explicitly force CPU execution
            self.device = torch.device("cpu")
            self.model.to(self.device)
            # Set thread configuration
            torch.set_num_threads(config.get("workers", 4) or 4)
        except Exception as e:
            raise BackendError(f"Failed to initialize HF model context: {str(e)}") from e

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        if not hasattr(self, "model"):
            raise BackendError("HFTransformersBackend is not initialized.")
        try:
            with torch.no_grad():
                encoded_input = self.tokenizer(
                    texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                model_output = self.model(**encoded_input)
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                return sentence_embeddings.cpu().numpy().astype(np.float32)
        except Exception as e:
            raise BackendError(f"Failed to run inference via HF PyTorch backend: {str(e)}") from e
