import re
from typing import Dict, Any, List

class TaxonomyMapper:
    """Canonicalizes raw extracted skill and domain names to matching taxonomy standards."""

    # Map variation keys to standardized canonical skill tags
    SKILL_ALIAS_REGISTRY = {
        "embeddings-based retrieval systems": "large_language_models",
        "embeddings-based retrieval": "large_language_models",
        "embeddings based retrieval systems": "large_language_models",
        "embeddings based retrieval": "large_language_models",
        "llm": "large_language_models",
        "large language models": "large_language_models",
        "gpt": "large_language_models",
        "gpt-4": "large_language_models",
        "fine-tuning llms": "llm_fine_tuning",
        "llm fine-tuning": "llm_fine_tuning",
        "lora": "llm_fine_tuning",
        "qlora": "llm_fine_tuning",
        "rag": "rag",
        "retrieval augmented generation": "rag",
        "langchain": "langchain",
        "llama-index": "llama_index",
        "llamaindex": "llama_index",
        
        "nlp": "natural_language_processing",
        "natural language processing": "natural_language_processing",
        "text classification": "text_classification",
        "named entity recognition": "named_entity_recognition",
        "transformers": "transformers",
        
        "pytorch": "pytorch",
        "tensorflow": "tensorflow",
        "keras": "keras",
        "gans": "gans",
        "image classification": "image_classification",
        
        "vector database": "vector_databases",
        "vector databases": "vector_databases",
        "pinecone": "pinecone",
        "weaviate": "weaviate",
        "qdrant": "qdrant",
        "milvus": "milvus",
        "faiss": "faiss",
        
        "spark": "apache_spark",
        "pyspark": "apache_spark",
        "apache spark": "apache_spark",
        "kafka": "kafka",
        "apache kafka": "kafka",
        "airflow": "airflow",
        "apache airflow": "airflow",
        
        "python": "python",
        "golang": "golang",
        "fastapi": "fastapi",
        "flask": "flask",
        "docker": "docker",
        "kubernetes": "kubernetes",
        "k8s": "kubernetes",
        "aws": "aws",
        "gcp": "gcp"
    }

    def canonicalize_skill(self, skill_name: str) -> str:
        """Standardizes and maps raw skill to taxonomy key."""
        cleaned = skill_name.lower().strip()
        cleaned = re.sub(r"[^\w\s-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        
        if cleaned in self.SKILL_ALIAS_REGISTRY:
            return self.SKILL_ALIAS_REGISTRY[cleaned]
            
        if "large language model" in cleaned:
            return "large_language_models"
        if "fine-tuning" in cleaned or "finetuning" in cleaned:
            return "llm_fine_tuning"
        if "vector database" in cleaned:
            return "vector_databases"
            
        return cleaned.replace(" ", "_")

    def canonicalize_domain(self, domain_name: str) -> str:
        """Maps domains to canonical names."""
        cleaned = domain_name.lower().strip()
        if "natural language" in cleaned or "nlp" in cleaned:
            return "NLP"
        if "information retrieval" in cleaned or "search" in cleaned or "ir" in cleaned:
            return "Information Retrieval"
        if "recommend" in cleaned:
            return "Recommendation Systems"
        if "operations" in cleaned or "mlops" in cleaned:
            return "MLOps"
        return domain_name.title()
