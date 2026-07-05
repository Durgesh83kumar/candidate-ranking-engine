import numpy as np
from typing import List, Dict, Any, Tuple
from src.reranker.config import RerankerConfig
from src.reranker.models import CrossEncoderModelRegistry
from src.reranker.exceptions import CalibrationError

class BatchRerankingScorer:
    """Computes Cross-Encoder scores for candidate pairs in optimized CPU batches."""

    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = None

    def initialize_model(self) -> None:
        """Loads model inside the shared class registry."""
        if self.model is None:
            self.model = CrossEncoderModelRegistry.get_model(
                model_name=self.config.model_name,
                fallback_model_name=self.config.fallback_model_name,
                device=self.config.device
            )

    def _sigmoid(self, x: float) -> float:
        """Applies sigmoid function to map raw logit into probability score in [0.0, 1.0]."""
        return 1.0 / (1.0 + np.exp(-x))

    def score_pairs(self, pairs: List[Tuple[str, str, str]]) -> Dict[str, Tuple[float, float]]:
        """Scores candidate text pairs in batch.
        
        Args:
            pairs: List of (candidate_id, query_text, candidate_text) tuples.
            
        Returns:
            Dict[str, Tuple[float, float]]: Maps candidate_id -> (logit, probability) scores.
        """
        if not pairs:
            return {}
            
        self.initialize_model()
        
        candidate_ids = [p[0] for p in pairs]
        text_pairs = [(p[1], p[2]) for p in pairs]
        
        print(f"Running Cross-Encoder batch inference on {len(text_pairs)} pairs (batch_size={self.config.batch_size})...")
        
        try:
            # sentence-transformers CrossEncoder.predict returns a numpy float array
            raw_logits = self.model.predict(
                sentences=text_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Map candidate_id to logit and probability
            scored_candidates = {}
            
            # Handle predict returning scalar float if only 1 pair is passed
            if np.isscalar(raw_logits):
                raw_logits = np.array([raw_logits])
                
            for cid, logit in zip(candidate_ids, raw_logits):
                logit_val = float(logit)
                prob_val = self._sigmoid(logit_val)
                scored_candidates[cid] = (logit_val, prob_val)
                
            return scored_candidates
            
        except Exception as e:
            raise CalibrationError(f"Cross-Encoder inference run failed: {str(e)}") from e
