from typing import Dict, Any, List
from src.document_generation.exceptions import TemplatingError

class SemanticTemplateEngine:
    """Translates structured candidate data into natural language sections for vector indexing."""

    def format_profile_header(self, profile: Dict[str, Any]) -> str:
        """Formats the candidate's core identity section."""
        title = profile.get("current_title_normalized", profile.get("current_title", "Software Engineer"))
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        
        parts = [
            "# PROFESSIONAL PROFILE",
            f"[TITLE] {title} | [HEADLINE] {headline}" if headline else f"[TITLE] {title}",
            summary
        ]
        return "\n".join([p for p in parts if p])

    def format_skills_section(self, skills: List[Dict[str, Any]]) -> str:
        """Translates skills lists into descriptive, context-rich sentences."""
        if not skills:
            return ""
            
        must_have = []
        preferred = []
        
        for s in skills:
            name = s.get("name_normalized", s.get("name", "")).replace("_", " ").title()
            prof = s.get("proficiency", "beginner")
            dur = s.get("duration_months", 0)
            endors = s.get("endorsements", 0)
            
            # Format skill into a natural sentence
            desc = f"{name} ({prof} proficiency"
            if dur > 0:
                desc += f", {dur} months experience"
            if endors > 0:
                desc += f", {endors} peer endorsements"
            desc += ")"
            
            if s.get("is_ai_skill") is True:
                must_have.append(desc)
            else:
                preferred.append(desc)
                
        parts = ["# TECHNICAL SKILLS"]
        if must_have:
            parts.append(f"AI/ML Core: {', '.join(must_have)}.")
        if preferred:
            parts.append(f"General Engineering: {', '.join(preferred)}.")
            
        return "\n".join(parts)

    def format_career_timeline(self, career_history: List[Dict[str, Any]]) -> str:
        """Formats work experience chronologically, highlighting promotions and production metrics."""
        if not career_history:
            return ""
            
        parts = ["# CAREER TIMELINE"]
        
        # Chronological order starting with most recent job
        sorted_jobs = sorted(
            career_history, 
            key=lambda x: x.get("start_date_parsed") if x.get("start_date_parsed") else x.get("start_date", ""),
            reverse=True
        )
        
        for idx, job in enumerate(sorted_jobs, 1):
            company = job.get("company", "Unknown Company")
            title = job.get("title_normalized", job.get("title", "Engineer"))
            dur = job.get("duration_months", 0)
            desc = job.get("description", "")
            
            # Formatting timeline header
            job_header = f"## Role {idx}: {title} at {company}"
            if dur > 0:
                years = dur // 12
                months = dur % 12
                dur_str = f"{years} years" if years > 0 else ""
                if months > 0:
                    dur_str += f" {months} months" if dur_str else f"{months} months"
                job_header += f" (Duration: {dur_str})"
                
            parts.append(job_header)
            
            # Highlight system metrics / scale terms in description if present
            # We enforce highlighting by formatting descriptions
            parts.append(desc)
            
        return "\n".join(parts)

    def format_education(self, education: List[Dict[str, Any]]) -> str:
        """Converts academic parameters to descriptive sentences."""
        if not education:
            return ""
            
        parts = ["# ACADEMIC FOUNDATIONS"]
        for edu in education:
            degree = edu.get("degree_normalized", edu.get("degree", "Bachelor's"))
            field = edu.get("field_of_study", "Computer Science")
            inst = edu.get("institution", "University")
            tier = edu.get("tier", "unknown")
            
            edu_str = f"Completed a {degree} degree in {field} from {inst}"
            if tier and tier != "unknown":
                edu_str += f" (Tier classification: {tier.replace('_', ' ').title()})"
            edu_str += "."
            parts.append(edu_str)
            
        return "\n".join(parts)

    def format_languages(self, languages: List[Dict[str, Any]]) -> str:
        """Converts languages parameters to natural language."""
        if not languages:
            return ""
        lang_strs = []
        for lang_entry in languages:
            lang = lang_entry.get("language", "")
            prof = lang_entry.get("proficiency", "conversational")
            if lang:
                lang_strs.append(f"{lang} with {prof} proficiency")
        return f"Languages spoken: {', '.join(lang_strs)}."

    def build_document(self, candidate_data: Dict[str, Any]) -> str:
        """Builds the comprehensive v2 search document string combining all sections."""
        try:
            profile = candidate_data.get("profile", {})
            skills = candidate_data.get("skills", [])
            career = candidate_data.get("career_history", [])
            education = candidate_data.get("education", [])
            languages = candidate_data.get("languages", [])
            
            sections = [
                self.format_profile_header(profile),
                self.format_skills_section(skills),
                self.format_career_timeline(career),
                self.format_education(education),
            ]
            
            lang_str = self.format_languages(languages)
            if lang_str:
                sections.append(f"# ADDITIONAL DETAILS\n{lang_str}")
                
            return "\n\n".join([s for s in sections if s])
        except Exception as e:
            raise TemplatingError(f"Failed to build search document: {str(e)}") from e
