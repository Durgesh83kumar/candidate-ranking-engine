import os
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.retrieval.exceptions import RetrievalError

class RetrievalCandidate(BaseModel):
    """Production-grade structured model representing a retrieved candidate and matching signals."""
    candidate_id: str
    anonymized_name: str
    rrf_score: float
    confidence_score: float = 0.0
    matched_queries: List[str] = Field(default_factory=list)
    matched_profile_sections: List[str] = Field(default_factory=list)
    years_of_experience: float
    current_title: str
    current_company: str
    location: str
    country: str
    expected_salary_lpa: Optional[float] = None
    notice_period_days: int
    open_to_work: bool
    github_activity: float
    profile_completeness: float
    work_mode: str
    relocation: bool
    query_similarities: Dict[str, float] = Field(default_factory=dict)
    query_ranks: Dict[str, int] = Field(default_factory=dict)


class EnrichmentJoiner:
    """Enriches fused candidate lists with candidate profile and signal metadata from Parquet files."""

    def __init__(self, candidates_parquet_path: str):
        self.parquet_path = candidates_parquet_path

    def join_and_enrich(self, fused_results: List[Dict[str, Any]], queries: Dict[str, str], negative_keywords: List[str]) -> List[Dict[str, Any]]:
        """Joins fused list with Parquet candidate records and performs match evidence analysis.
        
        Args:
            fused_results: List of candidates with RRF scores, similarities, and ranks.
            queries: Dict of generated queries to perform keyword matching evidence searches.
            negative_keywords: Exclusions keywords to verify.
            
        Returns:
            List[Dict[str, Any]]: Rich list of candidate data structures.
        """
        if not fused_results:
            return []
            
        if not os.path.exists(self.parquet_path):
            raise RetrievalError(f"Preprocessed candidates Parquet file not found: {self.parquet_path}")

        try:
            # Load candidate parquet
            df = pd.read_parquet(self.parquet_path)
            df.set_index("candidate_id", inplace=True)
            
            enriched_candidates = []
            
            # Map query terms to set for fast match evidence lookup
            all_query_terms = set()
            for q_text in queries.values():
                for word in q_text.lower().split():
                    if len(word) > 3:  # skip short stop words
                        all_query_terms.add(word)
                        
            for cand in fused_results:
                cid = cand["candidate_id"]
                if cid not in df.index:
                    continue
                    
                row = df.loc[cid]
                profile = row.get("profile", {})
                signals = row.get("redrob_signals", {})
                
                # Check for PyArrow NDArray parsing
                if hasattr(profile, "tolist"):
                    profile = profile.tolist()
                if hasattr(signals, "tolist"):
                    signals = signals.tolist()
                
                # If parsed as a list of dicts, get first item
                if isinstance(profile, list) and len(profile) > 0:
                    profile = profile[0]
                if isinstance(signals, list) and len(signals) > 0:
                    signals = signals[0]
                    
                profile = profile or {}
                signals = signals or {}
                
                # Basic metadata extraction
                current_title = profile.get("current_title_normalized", profile.get("current_title", "Unknown"))
                current_company = profile.get("current_company", "Unknown")
                location = profile.get("location_normalized", profile.get("location", "Unknown"))
                country = profile.get("country_normalized", profile.get("country", "Unknown"))
                
                exp_years = float(profile.get("years_of_experience_calculated", profile.get("years_of_experience", 0.0)))
                notice_period = int(signals.get("notice_period_days", 90))
                open_to_work = bool(signals.get("open_to_work_flag", False))
                github_score = float(signals.get("github_activity_score", 0.0))
                completeness = float(signals.get("profile_completeness_score", 0.0))
                
                work_mode = signals.get("preferred_work_mode", "Unknown")
                relocate = bool(signals.get("willing_to_relocate", False))
                
                # Salary processing (min/max range to average float LPA)
                salary_range = signals.get("expected_salary_range_inr_lpa", {})
                if isinstance(salary_range, dict):
                    sal_min = salary_range.get("min", 0.0)
                    sal_max = salary_range.get("max", 0.0)
                    expected_salary = (sal_min + sal_max) / 2.0 if (sal_min + sal_max) > 0 else None
                elif isinstance(salary_range, (int, float)):
                    expected_salary = float(salary_range)
                else:
                    expected_salary = None
                    
                # Match evidence tracking checks
                evidence = []
                
                # 1. Summary Evidence
                summary_text = str(profile.get("summary", "")).lower()
                if any(term in summary_text for term in all_query_terms):
                    evidence.append("Matched Summary")
                    
                # 2. Skills Evidence
                skills_list = row.get("skills", [])
                if hasattr(skills_list, "tolist"):
                    skills_list = skills_list.tolist()
                skills_text = " ".join([str(s).lower() for s in skills_list])
                if any(term in skills_text for term in all_query_terms):
                    evidence.append("Matched Skills")
                    
                # 3. Career History Evidence
                jobs = row.get("career_history", [])
                if hasattr(jobs, "tolist"):
                    jobs = jobs.tolist()
                jobs_text = ""
                if isinstance(jobs, list):
                    for job in jobs:
                        if isinstance(job, dict):
                            jobs_text += f" {job.get('job_title', '')} {job.get('description', '')}"
                if any(term in jobs_text.lower() for term in all_query_terms):
                    evidence.append("Matched Career History")
                    
                # 4. Education Evidence
                edu = row.get("education", [])
                if hasattr(edu, "tolist"):
                    edu = edu.tolist()
                edu_text = ""
                if isinstance(edu, list):
                    for deg in edu:
                        if isinstance(deg, dict):
                            edu_text += f" {deg.get('degree', '')} {deg.get('field_of_study', '')}"
                if any(term in edu_text.lower() for term in all_query_terms):
                    evidence.append("Matched Education")
                    
                # 5. Certifications Evidence
                certs = row.get("certifications", [])
                if hasattr(certs, "tolist"):
                    certs = certs.tolist()
                certs_text = " ".join([str(c).lower() for c in certs])
                if any(term in certs_text for term in all_query_terms):
                    evidence.append("Matched Certifications")
                    
                # 6. Github Activity Evidence
                if github_score > 40.0:
                    evidence.append("Matched Github Activity")
                    
                # If negative query exclusions are matched, flag the candidate
                is_excluded = False
                resume_full_text = f"{summary_text} {skills_text} {jobs_text}".lower()
                matched_negatives = [k for k in negative_keywords if k in resume_full_text]
                
                enriched_cand = {
                    "candidate_id": cid,
                    "anonymized_name": profile.get("anonymized_name", f"Candidate_{cid[:8]}"),
                    "rrf_score": cand["rrf_score"],
                    "matched_queries": cand.get("matched_queries", []),
                    "matched_profile_sections": list(set(evidence)),
                    "years_of_experience": exp_years,
                    "current_title": current_title,
                    "current_company": current_company,
                    "location": location,
                    "country": country,
                    "expected_salary_lpa": expected_salary,
                    "notice_period_days": notice_period,
                    "open_to_work": open_to_work,
                    "github_activity": github_score,
                    "profile_completeness": completeness,
                    "work_mode": work_mode,
                    "relocation": relocate,
                    "query_similarities": cand.get("query_similarities", {}),
                    "query_ranks": cand.get("query_ranks", {}),
                    "is_excluded": len(matched_negatives) > 0,
                    "matched_exclusions": matched_negatives
                }
                enriched_candidates.append(enriched_cand)
                
            return enriched_candidates
            
        except Exception as e:
            raise RetrievalError(f"Failed to join and enrich candidate metadata: {str(e)}") from e
