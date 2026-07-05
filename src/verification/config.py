import re
from typing import Dict, Any

class VerificationConfig:
    """Manages the verification heuristics, multipliers, and regular expressions."""

    def __init__(self, **kwargs):
        # Scoring Coefficients
        self.penalty_recruiter = float(kwargs.get("penalty_recruiter", 0.70))
        self.boost_engineering = float(kwargs.get("boost_engineering", 1.05))
        self.penalty_skills_missing = float(kwargs.get("penalty_skills_missing", 0.20))
        self.penalty_ai_frameworks_missing = float(kwargs.get("penalty_ai_frameworks_missing", 0.80))
        
        # Honeypot penalty
        self.penalty_honeypot = float(kwargs.get("penalty_honeypot", 0.00))

        # Recruiter vs Engineering Regex Pattern Strings
        self.recruiter_pattern = re.compile(
            r"(?i)\b(hired|recruited|managed|managed a team of|sourced|interviewed|hiring for)\b.*\b(python|ml|machine learning|ai|data scientist|engineer|developer)\b"
        )
        self.engineering_pattern = re.compile(
            r"(?i)\b(deployed|implemented|built|optimized|wrote|trained|fine-tuned|designed|scaled)\b.*\b(python|microservices|rag|models|llm|pipeline|neural network|transformer)\b"
        )

        # True AI Framework Checklist
        self.ai_frameworks = [
            "pytorch", "tensorflow", "keras", "jax", "transformers", "huggingface", 
            "langchain", "llamaindex", "faiss", "pinecone", "milvus", "qdrant", 
            "onnx", "tensorrt", "spacy", "nltk", "scikit-learn"
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "penalty_recruiter": self.penalty_recruiter,
            "boost_engineering": self.boost_engineering,
            "penalty_skills_missing": self.penalty_skills_missing,
            "penalty_ai_frameworks_missing": self.penalty_ai_frameworks_missing,
            "penalty_honeypot": self.penalty_honeypot,
            "ai_frameworks": self.ai_frameworks
        }
