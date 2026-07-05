import unicodedata
import re
from typing import Dict, Any, Tuple
from src.preprocessing.normalizers.base import BaseNormalizer

class TextNormalizer(BaseNormalizer):
    """Normalizes string fields including text cleaning, location parsing, and job titles standardization."""
    
    WHITESPACE_PATTERN = re.compile(r"\s+")
    
    TITLE_MAP = {
        r"(?i)\bbackend\b.*\beng\w*": "Backend Engineer",
        r"(?i)\bfrontend\b.*\beng\w*": "Frontend Engineer",
        r"(?i)\bfull\s*stack\b.*\beng\w*": "Fullstack Engineer",
        r"(?i)\bmachine\s*learning\b.*\beng\w*|\bml\b.*\beng\w*": "Machine Learning Engineer",
        r"(?i)\bdeep\s*learning\b.*\beng\w*": "Deep Learning Engineer",
        r"(?i)\bai\b.*\beng\w*|\bartificial\s*intelligence\b.*\beng\w*": "AI Engineer",
        r"(?i)\bdata\b.*\beng\w*": "Data Engineer",
        r"(?i)\boperations?\b.*\bman\w*": "Operations Manager",
        r"(?i)\bmarketing\b.*\bman\w*": "Marketing Manager",
        r"(?i)\bdevops\b.*\beng\w*": "DevOps Engineer",
        r"(?i)\bqa\b.*\beng\w*|\bquality\b.*\beng\w*|\btesting\b.*\beng\w*": "QA Engineer"
    }

    COUNTRY_MAP = {
        "india": "IN",
        "united states": "US",
        "usa": "US",
        "canada": "CA",
        "united kingdom": "UK",
        "uk": "UK",
        "germany": "DE",
        "australia": "AU"
    }

    def clean_text(self, text: Any) -> str:
        """Applies Unicode NFKC normalization, strips outer whitespace, and collapses inner whitespaces."""
        if text is None or not isinstance(text, str):
            return ""
        # Unicode normalization (NFKC)
        normalized = unicodedata.normalize("NFKC", text)
        # Collapse multiple spaces, tabs, and newlines
        cleaned = self.WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return cleaned

    def clean_description(self, text: Any) -> str:
        """Applies Unicode NFKC normalization and trims whitespace, but preserves newlines for readability."""
        if text is None or not isinstance(text, str):
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        # Trim leading/trailing whitespace on each line, collapse horizontal spaces
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
        # Remove empty lines
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def normalize_title(self, title: str) -> str:
        """Maps diverse title strings to a standardized role naming ontology."""
        cleaned_title = self.clean_text(title)
        if not cleaned_title:
            return "Software Engineer" # Safe fallback title
            
        for regex, target in self.TITLE_MAP.items():
            if re.search(regex, cleaned_title):
                return target
        return cleaned_title.title()

    def normalize_location(self, location: str, country: str) -> Tuple[str, str, str]:
        """Splits location string into City and State/Region, and normalizes Country.
        
        Returns:
            Tuple[str, str, str]: (city_normalized, region_normalized, country_normalized)
        """
        clean_loc = self.clean_text(location)
        clean_country = self.clean_text(country)
        
        city = "Unknown"
        region = "Unknown"
        
        if clean_loc and clean_loc != "Unknown":
            parts = [p.strip() for p in clean_loc.split(",") if p.strip()]
            if len(parts) >= 2:
                city = parts[0].title()
                region = parts[1].title()
            elif len(parts) == 1:
                city = parts[0].title()
                
        # Standardize Country
        country_key = clean_country.lower()
        country_norm = self.COUNTRY_MAP.get(country_key, clean_country.upper() if clean_country else "UNKNOWN")
        
        return city, region, country_norm

    def normalize(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Performs structural and field-level text normalization on the candidate dictionary."""
        profile = candidate_data.get("profile", {})
        if profile:
            profile["headline"] = self.clean_text(profile.get("headline"))
            profile["summary"] = self.clean_text(profile.get("summary"))
            
            raw_loc = profile.get("location", "")
            raw_country = profile.get("country", "")
            city, region, country_norm = self.normalize_location(raw_loc, raw_country)
            
            profile["location_raw"] = raw_loc
            profile["location_normalized"] = f"{city}, {region}" if region != "Unknown" else city
            profile["country_raw"] = raw_country
            profile["country_normalized"] = country_norm
            
            raw_title = profile.get("current_title", "")
            profile["current_title_raw"] = raw_title
            profile["current_title_normalized"] = self.normalize_title(raw_title)
            profile["current_company"] = self.clean_text(profile.get("current_company"))
            profile["current_industry"] = self.clean_text(profile.get("current_industry"))

        # Normalize Career History
        for job in candidate_data.get("career_history", []):
            job["company"] = self.clean_text(job.get("company"))
            raw_job_title = job.get("title", "")
            job["title_raw"] = raw_job_title
            job["title_normalized"] = self.normalize_title(raw_job_title)
            job["description"] = self.clean_description(job.get("description"))

        # Normalize Education degrees and institutions
        for edu in candidate_data.get("education", []):
            edu["institution"] = self.clean_text(edu.get("institution"))
            
            raw_degree = edu.get("degree", "")
            edu["degree_raw"] = raw_degree
            
            # Basic degree bucket mapping
            degree_lower = raw_degree.lower()
            if any(term in degree_lower for term in ["b.e", "b.tech", "bachelor", "b.s", "bsc"]):
                edu["degree_normalized"] = "Bachelor"
            elif any(term in degree_lower for term in ["m.e", "m.tech", "master", "m.s", "msc", "mba"]):
                edu["degree_normalized"] = "Master"
            elif any(term in degree_lower for term in ["phd", "ph.d", "doctorate", "doctor"]):
                edu["degree_normalized"] = "Doctorate"
            else:
                edu["degree_normalized"] = raw_degree.strip().title()
                
            edu["field_of_study"] = self.clean_text(edu.get("field_of_study"))
            
        # Clean redrob signals expected salary range (swap min/max if min > max)
        signals = candidate_data.get("redrob_signals", {})
        if signals and isinstance(signals, dict):
            salary = signals.get("expected_salary_range_inr_lpa")
            if salary and isinstance(salary, dict):
                s_min = salary.get("min")
                s_max = salary.get("max")
                if s_min is not None and s_max is not None and s_min > s_max:
                    salary["min"], salary["max"] = s_max, s_min
            
        return candidate_data
