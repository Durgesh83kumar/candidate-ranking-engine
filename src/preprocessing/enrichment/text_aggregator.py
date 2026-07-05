from typing import Dict, Any, List

class TextAggregator:
    """Aggregates candidate fields and career history into clean structured text versions for search indexing."""

    def clean_text_segment(self, text: Any) -> str:
        """Sanitizes text by stripping whitespace and ensuring it is a string."""
        if not text or not isinstance(text, str):
            return ""
        return text.strip()

    def build_search_documents(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs search_document_v1 and search_document_v2 and adds them to candidate_data."""
        profile = candidate_data.get("profile", {})
        skills = candidate_data.get("skills", [])
        career_history = candidate_data.get("career_history", [])
        education = candidate_data.get("education", [])
        certifications = candidate_data.get("certifications", [])
        languages = candidate_data.get("languages", [])

        # Extract normalized attributes
        curr_title = self.clean_text_segment(profile.get("current_title_normalized", profile.get("current_title", "")))
        headline = self.clean_text_segment(profile.get("headline", ""))
        summary = self.clean_text_segment(profile.get("summary", ""))

        # Skills list
        skills_str = ", ".join(self.clean_text_segment(s.get("name_normalized", s.get("name", ""))) for s in skills if s)

        # ----------------------------------------------------
        # Build Document v1 (Compact)
        # ----------------------------------------------------
        doc_v1_parts = []
        if curr_title or headline:
            doc_v1_parts.append(f"[TITLE] {curr_title} [HEADLINE] {headline}")
        if summary:
            doc_v1_parts.append(f"[SUMMARY] {summary}")
        if skills_str:
            doc_v1_parts.append(f"[SKILLS] {skills_str}")
            
        candidate_data["search_document_v1"] = "\n".join(doc_v1_parts)

        # ----------------------------------------------------
        # Build Document v2 (Comprehensive)
        # ----------------------------------------------------
        doc_v2_parts = []
        
        # 1. Identity & Role
        if curr_title or headline:
            doc_v2_parts.append(f"[TITLE] {curr_title} [HEADLINE] {headline}")
            
        # 2. Summary
        if summary:
            doc_v2_parts.append(f"[SUMMARY] {summary}")
            
        # 3. Core Competencies (Skills)
        if skills_str:
            doc_v2_parts.append(f"[SKILLS] {skills_str}")

        # 4. Chronological Career History (Reverse chronological order: most recent first)
        sorted_history = sorted(
            career_history, 
            key=lambda x: x.get("start_date_parsed") if x.get("start_date_parsed") else x.get("start_date", ""),
            reverse=True
        )
        for job in sorted_history:
            company = self.clean_text_segment(job.get("company", ""))
            title = self.clean_text_segment(job.get("title_normalized", job.get("title", "")))
            desc = self.clean_text_segment(job.get("description", ""))
            if company or title or desc:
                doc_v2_parts.append(f"[EXPERIENCE] {company} - {title}: {desc}")

        # 5. Education
        for edu in education:
            inst = self.clean_text_segment(edu.get("institution", ""))
            degree = self.clean_text_segment(edu.get("degree_normalized", edu.get("degree", "")))
            field = self.clean_text_segment(edu.get("field_of_study", ""))
            tier = self.clean_text_segment(edu.get("tier", "unknown"))
            if inst or degree or field:
                doc_v2_parts.append(f"[EDUCATION] {inst} - {degree} in {field} (Tier: {tier})")

        # 6. Certifications
        for cert in certifications:
            name = self.clean_text_segment(cert.get("name", ""))
            issuer = self.clean_text_segment(cert.get("issuer", ""))
            if name:
                doc_v2_parts.append(f"[CERTIFICATIONS] {name} issued by {issuer}")

        # 7. Languages (Declarative transformation)
        lang_parts = []
        for lang_entry in languages:
            lang = self.clean_text_segment(lang_entry.get("language", ""))
            prof = self.clean_text_segment(lang_entry.get("proficiency", ""))
            if lang and prof:
                lang_parts.append(f"{lang} ({prof} proficiency)")
        if lang_parts:
            doc_v2_parts.append(f"[LANGUAGES] Languages spoken: {', '.join(lang_parts)}.")

        # Save to candidate records
        candidate_data["search_document_v2"] = "\n".join(doc_v2_parts)
        candidate_data["search_vector_text"] = candidate_data["search_document_v2"] # Legacy mapping for compliance

        return candidate_data
