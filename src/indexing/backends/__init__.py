from src.indexing.backends.base import BaseEmbeddingBackend
from src.indexing.backends.onnx_runtime import ONNXRuntimeBackend
from src.indexing.backends.sentence_transformers import SentenceTransformersBackend
from src.indexing.backends.hf_transformers import HFTransformersBackend
from src.indexing.exceptions import BackendError

def get_backend(backend_name: str) -> BaseEmbeddingBackend:
    """Returns the requested pluggable backend instance.
    
    Args:
        backend_name (str): Name of the backend engine ('onnx', 'sentence_transformers', 'hf').
        
    Returns:
        BaseEmbeddingBackend: Pluggable backend engine instance.
    """
    backends = {
        "onnx": ONNXRuntimeBackend,
        "sentence_transformers": SentenceTransformersBackend,
        "hf": HFTransformersBackend
    }
    
    name = backend_name.lower().strip()
    if name not in backends:
        raise BackendError(f"Unknown embedding backend type: '{backend_name}'. Choose from 'onnx', 'sentence_transformers', 'hf'.")
        
    return backends[name]()
