import os
import json
from typing import Dict, Any, List
from src.retrieval.exceptions import RetrievalError

class MultiQueryBuilder:
    """Parses hiring requirements and compiles multi-semantic search query targets."""

    def __init__(self, spec_path: str, queries_path: str):
        self.spec_path = spec_path
        self.queries_path = queries_path
        self.specification = {}
        self.raw_queries = {}
        
    def load_requirements(self) -> None:
        """Loads hiring specifications and search queries from disk."""
        if not os.path.exists(self.spec_path):
            raise RetrievalError(f"Hiring specification file not found: {self.spec_path}")
        if not os.path.exists(self.queries_path):
            raise RetrievalError(f"Search queries file not found: {self.queries_path}")
            
        try:
            with open(self.spec_path, "r", encoding="utf-8") as f:
                self.specification = json.load(f)
            with open(self.queries_path, "r", encoding="utf-8") as f:
                self.raw_queries = json.load(f)
        except Exception as e:
            raise RetrievalError(f"Failed to read requirements specifications: {str(e)}") from e

    def build_queries(self) -> Dict[str, str]:
        """Assembles 6 semantic queries covering General, Tech, AI, Leadership, Domain, and Exclusions."""
        if not self.specification:
            self.load_requirements()
            
        # 1. General Role Query
        general = self.raw_queries.get("primary_query", "")
        if not general:
            role = self.specification.get("role", {})
            general = f"{role.get('seniority', '')} {role.get('title', 'AI Engineer')}"
            
        # 2. Technical Skills Query
        tech = self.raw_queries.get("technology_query", "")
        if not tech:
            skills = self.specification.get("skills", {})
            must = [s.get("name", "") for s in skills.get("must_have", [])]
            pref = [s.get("name", "") for s in skills.get("preferred", [])]
            tech = " ".join(must + pref)
            
        # 3. AI/ML Query
        ai_ml = self.raw_queries.get("concept_query", "")
        if not ai_ml:
            ai_ml = "large language models vector databases serving serving serving Servingserving serve serving RAG serving servingservingServing"

        # 4. Leadership Query (extracted from responsibilities & experience)
        leader_responsibilities = [
            r.get("description", "") 
            for r in self.specification.get("responsibilities", []) 
            if r.get("category") in ("mentoring", "architecture", "leadership")
        ]
        if leader_responsibilities:
            leadership = " ".join(leader_responsibilities)
        else:
            leadership = "Lead, mentor, grow engineering team, architecture ownership, tech lead"

        # 5. Domain Query
        domains = self.specification.get("domains", {})
        domain_list = domains.get("required", []) + domains.get("preferred", [])
        industries = self.specification.get("preferences", {}).get("industries", [])
        domain = " ".join(domain_list + industries)
        if not domain:
            domain = "NLP Information Retrieval Recommendation Systems HR-Tech Recruiting"

        # 6. Negative Constraints Query (for exclusion keywords)
        neg_prefs = self.specification.get("preferences", {}).get("negative_preferences", {})
        neg_keywords = []
        for category, list_items in neg_prefs.items():
            if isinstance(list_items, list):
                neg_keywords.extend(list_items)
        
        # Add CV exclusion if present in JD contexts
        negative = " ".join(neg_keywords)
        if not negative:
            negative = "hadoop mapreduce consulting only pure research title chasing"

        return {
            "general": general.strip(),
            "technical": tech.strip(),
            "ai_ml": ai_ml.strip(),
            "leadership": leadership.strip(),
            "domain": domain.strip(),
            "negative": negative.strip()
        }
        
    def get_negative_keywords(self) -> List[str]:
        """Returns the list of raw negative exclusions keywords."""
        if not self.specification:
            self.load_requirements()
        neg_prefs = self.specification.get("preferences", {}).get("negative_preferences", {})
        keywords = []
        for category, list_items in neg_prefs.items():
            if isinstance(list_items, list):
                keywords.extend([str(item).lower() for item in list_items])
        return keywords
