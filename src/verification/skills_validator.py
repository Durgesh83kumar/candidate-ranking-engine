import os
import json
from typing import Dict, Any, List, Tuple
from src.verification.config import VerificationConfig
from src.verification.exceptions import SkillsValidationError

class SkillsValidator:
    """Verifies that mandatory skills in JD are present in candidate skills registry or resume texts."""

    def __init__(self, config: VerificationConfig, spec_path: str):
        self.config = config
        self.spec_path = spec_path
        self.mandatory_skills = []
        self._load_mandatory_skills()

    def _load_mandatory_skills(self) -> None:
        """Loads critical must-have skills from hiring_specification.json."""
        if not os.path.exists(self.spec_path):
            raise SkillsValidationError(f"Hiring specification file not found: {self.spec_path}")
            
        try:
            with open(self.spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            
            skills_section = spec.get("skills", {})
            must_have = skills_section.get("must_have", [])
            
            # Map standard name tokens
            self.mandatory_skills = [
                str(s.get("name", "")).lower().strip() 
                for s in must_have if s.get("name")
            ]
        except Exception as e:
            raise SkillsValidationError(f"Failed to parse hiring specification must-haves: {str(e)}") from e

    def validate(self, candidate_skills: List[Any], search_doc: str) -> Tuple[float, List[str]]:
        """Verifies if candidate contains all mandatory skills.
        
        Args:
            candidate_skills: Standardized list of skills (dictionaries or strings).
            search_doc: Candidate search document text string (v2).
            
        Returns:
            Tuple[float, List[str]]: Multiplier coefficient and list of missing mandatory skills.
        """
        if not self.mandatory_skills:
            return 1.0, []
            
        # 1. Extract skill names from potential dictionaries
        cand_skills_set = set()
        if candidate_skills:
            for s in candidate_skills:
                if isinstance(s, dict):
                    name = s.get("name_normalized") or s.get("name_raw") or ""
                    cand_skills_set.add(str(name).lower().strip())
                else:
                    cand_skills_set.add(str(s).lower().strip())
                    
        doc_lower = str(search_doc).lower()
        
        # Synonyms/abbreviations expansion map for robust lookup
        skill_synonyms = {
            "large_language_models": ["large_language_models", "large language models", "llm", "llms", "gpt", "transformers"],
            "vector_databases": ["vector_databases", "vector database", "vector databases", "vectordb", "vector index", "vector search", "faiss", "pinecone", "milvus", "qdrant", "chroma"],
            "python": ["python", "py", "pytorch", "tensorflow", "keras", "jax", "transformers", "huggingface", "langchain", "llamaindex", "scikit-learn", "pandas", "numpy"],
            "evaluation_frameworks": ["evaluation_frameworks", "evaluation framework", "evaluation frameworks", "eval", "evaluation", "benchmarks", "ab testing", "a/b testing"]
        }
        
        missing_skills = []
        
        for required in self.mandatory_skills:
            syns = skill_synonyms.get(required, [required])
            
            # Check list or text search
            in_list = any(s in cand_skills_set for s in syns)
            in_text = any(s.replace("_", " ") in doc_lower for s in syns)
            
            if not in_list and not in_text:
                missing_skills.append(required)
                
        # If any mandatory skill is completely missing, apply hard penalty
        if missing_skills:
            return self.config.penalty_skills_missing, missing_skills
            
        return 1.0, []
ClassSkillsValidator = SkillsValidator
