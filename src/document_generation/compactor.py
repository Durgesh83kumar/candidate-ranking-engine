import re
from typing import List, Dict, Any
from src.document_generation.exceptions import CompactionError

class LengthCompactor:
    """Manages the token length budget of generated documents, compacting them deterministically to fit within limits."""

    def __init__(self, target_token_limit: int = 1024):
        self.target_token_limit = target_token_limit

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count of a text block using a standard 1 word = 1.3 tokens approximation."""
        if not text:
            return 0
        words = text.split()
        return int(len(words) * 1.3)

    def compact(self, candidate_data: Dict[str, Any], doc_v2_str: str) -> str:
        """Truncates or compacts career description lists if the comprehensive document exceeds the target limit."""
        token_estimate = self.estimate_tokens(doc_v2_str)
        if token_estimate <= self.target_token_limit:
            return doc_v2_str

        # If it exceeds the limit, we re-build the document but compact the older career details
        try:
            profile = candidate_data.get("profile", {})
            skills = candidate_data.get("skills", [])
            career = candidate_data.get("career_history", [])
            education = candidate_data.get("education", [])
            
            # 1. We keep Profile header and Skills intact
            from src.document_generation.templates import SemanticTemplateEngine
            engine = SemanticTemplateEngine()
            
            header_str = engine.format_profile_header(profile)
            skills_str = engine.format_skills_section(skills)
            edu_str = engine.format_education(education)
            
            base_token_count = self.estimate_tokens(header_str) + self.estimate_tokens(skills_str) + self.estimate_tokens(edu_str)
            
            # 2. We dynamically summarize/compact the Career Timeline
            career_parts = ["# CAREER TIMELINE"]
            sorted_jobs = sorted(
                career, 
                key=lambda x: x.get("start_date_parsed") if x.get("start_date_parsed") else x.get("start_date", ""),
                reverse=True
            )
            
            # We budget the remaining token capacity for the jobs
            remaining_tokens = self.target_token_limit - base_token_count - 50 # 50 token buffer
            
            for idx, job in enumerate(sorted_jobs, 1):
                company = job.get("company", "Unknown Company")
                title = job.get("title_normalized", job.get("title", "Engineer"))
                dur = job.get("duration_months", 0)
                desc = job.get("description", "")
                
                # Format timeline header
                job_header = f"## Role {idx}: {title} at {company}"
                if dur > 0:
                    years = dur // 12
                    months = dur % 12
                    dur_str = f"{years} years" if years > 0 else ""
                    if months > 0:
                        dur_str += f" {months} months" if dur_str else f"{months} months"
                    job_header += f" (Duration: {dur_str})"
                
                # For jobs after the first 2 roles, if budget is low, we only include the header and first sentence
                if idx > 2 or remaining_tokens < 200:
                    # Truncate description to first sentence
                    first_sentence = desc.split(".")[0] + "." if desc else ""
                    job_desc = first_sentence
                else:
                    job_desc = desc
                    
                job_block = f"{job_header}\n{job_desc}"
                job_tokens = self.estimate_tokens(job_block)
                
                # If we still have budget or it's the very first role, append it
                if remaining_tokens > 50 or idx == 1:
                    career_parts.append(job_block)
                    remaining_tokens -= job_tokens
                else:
                    # Append just the job title and company as a one-liner
                    career_parts.append(f"## Role {idx}: {title} at {company} (Position details truncated for length)")
                    
            timeline_str = "\n".join(career_parts)
            
            # Reassemble
            sections = [header_str, skills_str, timeline_str, edu_str]
            languages = candidate_data.get("languages", [])
            lang_str = engine.format_languages(languages)
            if lang_str:
                sections.append(f"# ADDITIONAL DETAILS\n{lang_str}")
                
            compacted_str = "\n\n".join([s for s in sections if s])
            return compacted_str
            
        except Exception as e:
            raise CompactionError(f"Failed to compact document: {str(e)}") from e
