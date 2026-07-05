import os
import json
from typing import Dict, Any, List, Tuple
from src.reranker.exceptions import RerankerError

class RerankingPairBuilder:
    """Assembles comprehensive recruiter query specifications and constructs text sequences for Cross-Encoder scoring."""

    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        self.recruiter_query = ""

    def build_recruiter_query(self) -> str:
        """Assembles comprehensive recruiter query from structured specification JSON.
        
        Combines title, summary, must-haves, preferred skills, responsibilities, experience thresholds, 
        leadership, AI/ML expectations, work mode, and exclusions into one coherent context string.
        """
        if not os.path.exists(self.spec_path):
            raise RerankerError(f"Hiring specification file not found: {self.spec_path}")
            
        try:
            with open(self.spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
        except Exception as e:
            raise RerankerError(f"Failed to read hiring specification JSON: {str(e)}") from e

        # Extract values
        role = spec.get("role", {})
        title = role.get("title", "AI Engineer")
        seniority = role.get("seniority", "senior")
        
        exp = spec.get("experience", {})
        min_years = exp.get("min_years", 5)
        max_years = exp.get("max_years", 9)
        startup_req = "Startup experience required." if exp.get("require_startup_experience") else ""
        prod_req = "Production systems experience required." if exp.get("require_production_experience") else ""
        
        skills = spec.get("skills", {})
        must_haves = [s.get("name", "") for s in skills.get("must_have", [])]
        preferred = [s.get("name", "") for s in skills.get("preferred", [])]
        
        responsibilities = [r.get("description", "") for r in spec.get("responsibilities", [])]
        
        pref = spec.get("preferences", {})
        work_mode = pref.get("work_mode", "hybrid")
        location = pref.get("location", "")
        max_notice = pref.get("max_notice_period_days", 30)
        
        neg_prefs = pref.get("negative_preferences", {})
        neg_keywords = []
        for cat, items in neg_prefs.items():
            if isinstance(items, list):
                neg_keywords.extend(items)
        neg_string = ", ".join(neg_keywords) if neg_keywords else "none"

        # Construct comprehensive recruiter query
        query_parts = [
            f"Role: {seniority.capitalize()} {title}.",
            f"Experience: {min_years} to {max_years} years. {startup_req} {prod_req}",
            f"Mandatory Technical Skills: {', '.join(must_haves)}.",
            f"Preferred Technical Skills: {', '.join(preferred)}.",
            f"Core Responsibilities: {' '.join(responsibilities)}",
            f"Work Environment Preferences: {work_mode} mode in {location}. notice period under {max_notice} days.",
            f"Explicit Disqualifiers and Exclusions: {neg_string}."
        ]
        
        self.recruiter_query = " ".join([p.strip() for p in query_parts if p.strip()])
        return self.recruiter_query

    def construct_pairs(self, candidates: List[Dict[str, Any]], search_docs_df: Any) -> List[Tuple[str, str, str]]:
        """Constructs list of pairs (JD_query, Candidate_search_doc) mapped to Candidate ID.
        
        Args:
            candidates: Retrieved candidates list from Phase 5.
            search_docs_df: DataFrame containing candidate search_document_v2 text.
            
        Returns:
            List[Tuple[str, str, str]]: List of tuples (candidate_id, query_text, candidate_text).
        """
        if not self.recruiter_query:
            self.build_recruiter_query()
            
        pairs = []
        
        # Load search_document_v2 text
        search_docs_df = search_docs_df.set_index("candidate_id")
        
        for cand in candidates:
            cid = cand["candidate_id"]
            if cid not in search_docs_df.index:
                continue
                
            row = search_docs_df.loc[cid]
            # Handle pandas Series vs DataFrame row
            if hasattr(row, "get"):
                cand_doc = row.get("search_document_v2", "")
            else:
                cand_doc = str(row)
                
            pairs.append((cid, self.recruiter_query, str(cand_doc)))
            
        return pairs
