from typing import Dict, List, Any
from src.indexing.searcher import VectorSearcher
from src.retrieval.config import RetrievalConfig

class BatchSearcher:
    """Performs dense searches against the compiled FAISS IndexFlatIP index for multiple queries."""

    def __init__(self, config: RetrievalConfig, index_path: str, index_metadata_path: str, backend_name: str = "sentence_transformers", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.config = config
        self.index_path = index_path
        self.index_metadata_path = index_metadata_path
        self.searcher = VectorSearcher(backend_name=backend_name, model_name=model_name)

    def load_index(self) -> None:
        """Loads index weights and offset-ID mapping JSON into memory."""
        self.searcher.load_index(self.index_path, self.index_metadata_path)

    def search_all(self, queries: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """Runs vector search for each query and returns similarity scores.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dict mapping query_type -> list of retrieved candidates.
        """
        results = {}
        for q_type, q_text in queries.items():
            if not q_text:
                results[q_type] = []
                continue
            
            # Embed query
            q_vec = self.searcher.embed_query(q_text)
            
            # Retrieve top-K
            hits = self.searcher.retrieve_top_k(q_vec, k=self.config.top_k_per_query)
            
            # Record query tracking details
            for hit in hits:
                hit["query_type"] = q_type
                
            results[q_type] = hits
            
        return results
