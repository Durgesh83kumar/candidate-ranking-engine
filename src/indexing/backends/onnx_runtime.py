import os
import numpy as np
from typing import List, Dict, Any
from src.indexing.backends.base import BaseEmbeddingBackend
from src.indexing.exceptions import BackendError

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    _has_ort = True
except ImportError:
    _has_ort = False

class ONNXRuntimeBackend(BaseEmbeddingBackend):
    """ONNX Runtime optimized CPU inference engine with automatic tokenization."""

    def initialize(self, model_name: str, config: Dict[str, Any]) -> None:
        if not _has_ort:
            raise BackendError("ONNX Runtime or HuggingFace Transformers is not installed.")
        
        # Check if local ONNX path is configured or exists
        self.onnx_path = config.get("onnx_model_path", "")
        if not self.onnx_path or not os.path.exists(self.onnx_path):
            raise BackendError(
                f"Local ONNX model path not configured or file not found: '{self.onnx_path}'. "
                "Triggering fallback to PyTorch/SentenceTransformers backends."
            )

        try:
            print(f"Initializing ONNX Runtime model from {self.onnx_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # CPU performance optimizations
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = config.get("workers", 4) or 4
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise BackendError(f"Failed to initialize ONNX Runtime session: {str(e)}") from e

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        if not hasattr(self, "session"):
            raise BackendError("ONNX Runtime session is not initialized.")
        try:
            # Tokenize inputs
            encoded = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors='np'
            )
            
            # Prepare feeds matching ONNX model inputs
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64)
            }
            if "token_type_ids" in encoded:
                inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
                
            outputs = self.session.run(None, inputs)
            
            # Simple mean pooling over sequence length
            token_embeddings = outputs[0]
            attention_mask = encoded["attention_mask"]
            
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            
            return (sum_embeddings / sum_mask).astype(np.float32)
        except Exception as e:
            raise BackendError(f"Failed to run ONNX Runtime inference: {str(e)}") from e
