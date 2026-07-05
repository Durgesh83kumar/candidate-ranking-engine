import re
import unicodedata
from typing import List, Dict, Any, Set
from src.document_generation.exceptions import CleaningError

class DocumentCleaner:
    """Cleans raw text fields, removes noise/boilerplate phrases, and deduplicates elements."""

    # Regex patterns for common resume noise/boilerplate
    BOILERPLATE_PATTERNS = [
        r"(?i)\breferences?\s+available\s+(?:upon\s+request|on\s+request)\b",
        r"(?i)\bhardworking\s+professional\s+looking\s+for\s+opportunities\b",
        r"(?i)\bobjective\s*:\s*seeking\s+a\s+challenging\s+position\b",
        r"(?i)\bcurriculum\s+vitae\b|\bresume\b"
    ]

    WHITESPACE_PATTERN = re.compile(r"\s+")

    def clean_text(self, text: Any) -> str:
        """Applies Unicode NFKC normalization, strips whitespace, and removes boilerplate noise."""
        if text is None or not isinstance(text, str):
            return ""
            
        try:
            # 1. Unicode Normalization
            normalized = unicodedata.normalize("NFKC", text)
            
            # 2. Boilerplate Removal
            for pattern in self.BOILERPLATE_PATTERNS:
                normalized = re.sub(pattern, "", normalized)
                
            # 3. Collapse Whitespace
            cleaned = self.WHITESPACE_PATTERN.sub(" ", normalized).strip()
            return cleaned
        except Exception as e:
            raise CleaningError(f"Failed to clean text field: {str(e)}") from e

    def clean_description(self, text: Any) -> str:
        """Trims whitespace and parses lines, preserving formatting but removing carriage return noise."""
        if text is None or not isinstance(text, str):
            return ""
        try:
            normalized = unicodedata.normalize("NFKC", text)
            for pattern in self.BOILERPLATE_PATTERNS:
                normalized = re.sub(pattern, "", normalized)
                
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
            # Filter empty lines
            lines = [line for line in lines if line]
            return "\n".join(lines)
        except Exception as e:
            raise CleaningError(f"Failed to clean job description: {str(e)}") from e

    def deduplicate_skills(self, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates a skills list, keeping the record with the highest proficiency."""
        if not skills:
            return []
            
        unique_skills = {}
        for skill in skills:
            if not isinstance(skill, dict) or "name_normalized" not in skill:
                continue
                
            name = skill["name_normalized"].strip().lower()
            if not name:
                continue
                
            # If skill already seen, compare proficiency to keep the highest quality entry
            if name in unique_skills:
                prev_prof = unique_skills[name].get("proficiency", "beginner").lower()
                curr_prof = skill.get("proficiency", "beginner").lower()
                
                # Simple weight map
                weights = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
                if weights.get(curr_prof, 1) > weights.get(prev_prof, 1):
                    unique_skills[name] = skill
            else:
                unique_skills[name] = skill
                
        return list(unique_skills.values())
