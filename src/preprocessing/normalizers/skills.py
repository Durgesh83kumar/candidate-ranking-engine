import re
from typing import Dict, Any, List
from src.preprocessing.normalizers.base import BaseNormalizer

class SkillNormalizer(BaseNormalizer):
    """Normalizes skill listings using a predefined taxonomy and alias registry, handling deduplication and domain mapping."""
    
    # Pre-compiled mapping for skill aliases/synonyms
    SKILL_ALIAS_REGISTRY = {
        # Large Language Models family
        "llm": "large_language_models",
        "large language models": "large_language_models",
        "gpt": "large_language_models",
        "gpt-4": "large_language_models",
        "fine-tuning llms": "llm_fine_tuning",
        "llm fine-tuning": "llm_fine_tuning",
        "lora": "llm_fine_tuning",
        "qlora": "llm_fine_tuning",
        "rlhf": "llm_fine_tuning",
        "dpo": "llm_fine_tuning",
        
        # NLP family
        "nlp": "natural_language_processing",
        "natural language processing": "natural_language_processing",
        "text classification": "text_classification",
        "sentiment analysis": "text_classification",
        
        # Deep Learning / Frameworks
        "pytorch": "pytorch",
        "torch": "pytorch",
        "tensorflow": "tensorflow",
        "tf": "tensorflow",
        "keras": "keras",
        "gans": "gans",
        "image classification": "image_classification",
        
        # Distributed Systems / Pipelines
        "spark": "apache_spark",
        "pyspark": "apache_spark",
        "spark streaming": "apache_spark",
        "apache spark": "apache_spark",
        "kafka": "kafka",
        "apache kafka": "kafka",
        "airflow": "airflow",
        "apache airflow": "airflow",
        
        # Vector Databases
        "milvus": "milvus",
        "pinecone": "pinecone",
        "chromadb": "chromadb",
        "chroma": "chromadb",
        "qdrant": "qdrant"
    }

    # AI/ML Domain Skills
    AI_SKILL_SET = {
        "large_language_models", "llm_fine_tuning", "prompt_engineering", 
        "rag", "langchain", "llama_index", "natural_language_processing", 
        "text_classification", "named_entity_recognition", "transformers", 
        "neural_networks", "pytorch", "tensorflow", "keras", "gans", 
        "image_classification", "scikit-learn", "statistical_modeling", 
        "regression", "xgboost", "deep_learning", "machine_learning", 
        "fine-tuning llms"
    }

    # Map proficiency strings to numeric weights for comparisons/aggregation
    PROFICIENCY_WEIGHTS = {
        "beginner": 1,
        "intermediate": 2,
        "advanced": 3,
        "expert": 4
    }

    def clean_skill_name(self, name: str) -> str:
        """Standardizes raw skill name casing and spacing."""
        cleaned = name.lower().strip()
        cleaned = re.sub(r"[^\w\s-]", "", cleaned)  # Remove special characters except spaces/hyphens
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def normalize_skill(self, name: str) -> str:
        """Translates cleaned skill name to the canonical skill taxonomy name."""
        cleaned = self.clean_skill_name(name)
        # 1. Exact Dictionary Match
        if cleaned in self.SKILL_ALIAS_REGISTRY:
            return self.SKILL_ALIAS_REGISTRY[cleaned]
            
        # 2. Basic heuristic rules
        if "large language model" in cleaned:
            return "large_language_models"
        if "fine-tuning" in cleaned and "llm" in cleaned:
            return "llm_fine_tuning"
        if "natural language" in cleaned:
            return "natural_language_processing"
            
        # Fallback to standard word separation
        return cleaned.replace(" ", "_")

    def is_ai_related(self, skill_name_norm: str) -> bool:
        """Determines if the normalized skill belongs to the AI/ML domain."""
        if skill_name_norm in self.AI_SKILL_SET:
            return True
        # If standard normalized name contains AI keywords
        ai_keywords = ["llm", "ai", "ml", "nlp", "learning", "neural", "transformer", "gpt", "rag"]
        return any(kw in skill_name_norm for kw in ai_keywords)

    def normalize(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes skills list, merges duplicates, and annotates ML/AI tags."""
        raw_skills = candidate_data.get("skills", [])
        if not raw_skills:
            candidate_data["skills"] = []
            return candidate_data

        normalized_skills_map = {}
        for skill_entry in raw_skills:
            if not isinstance(skill_entry, dict) or "name" not in skill_entry:
                continue
                
            raw_name = skill_entry["name"]
            norm_name = self.normalize_skill(raw_name)
            
            proficiency = skill_entry.get("proficiency", "beginner").lower()
            endorsements = int(skill_entry.get("endorsements", 0))
            duration = int(skill_entry.get("duration_months", 0))
            
            # Entity Resolution/Deduplication within same profile
            if norm_name in normalized_skills_map:
                # Merge records: keep highest proficiency, sum durations, take max endorsements
                existing = normalized_skills_map[norm_name]
                
                # Check proficiency weights
                curr_weight = self.PROFICIENCY_WEIGHTS.get(proficiency, 1)
                prev_weight = self.PROFICIENCY_WEIGHTS.get(existing["proficiency"], 1)
                if curr_weight > prev_weight:
                    existing["proficiency"] = proficiency
                    
                existing["endorsements"] = max(existing["endorsements"], endorsements)
                existing["duration_months"] += duration
            else:
                normalized_skills_map[norm_name] = {
                    "name_raw": raw_name,
                    "name_normalized": norm_name,
                    "proficiency": proficiency,
                    "endorsements": endorsements,
                    "duration_months": duration,
                    "is_ai_skill": self.is_ai_related(norm_name)
                }

        # Convert back to list
        candidate_data["skills"] = list(normalized_skills_map.values())
        return candidate_data
