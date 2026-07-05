from src.jd_intelligence.schema import HiringSpecification
from typing import Dict, Any, List

class QueryGenerator:
    """Generates optimized query permutations (Primary, Expanded, Tech, Concept, Skill queries) for search indexes."""

    def generate(self, spec: HiringSpecification) -> Dict[str, Any]:
        """Translates the structured spec into downstream search queries.
        
        Returns:
            Dict[str, Any]: Map of query types to search strings.
        """
        # 1. Primary Query: dense query focusing on title & seniority
        role = spec.role
        primary = f"{role.seniority.title()} {role.title}"
        if spec.domains.required:
            primary += f" specializing in {', '.join(spec.domains.required)}"
        else:
            primary += " specializing in Artificial Intelligence and Machine Learning"

        # 2. Expanded Query: detailed description containing core technologies & scale
        must_skills = [s.name for s in spec.skills.must_have]
        pref_skills = [s.name for s in spec.skills.preferred]
        
        expanded_parts = [
            f"Senior Software Engineer specializing in {role.title}.",
            f"Required expertise in {', '.join(must_skills[:4])}." if must_skills else "",
            f"Highly value experience with {', '.join(pref_skills[:3])}." if pref_skills else "",
            "Strong background in production systems, scaling, and validation metrics." if spec.experience.require_production_experience else ""
        ]
        expanded = " ".join([p for p in expanded_parts if p])

        # 3. Technology Query: space-separated keywords for token search (sparse/BM25)
        tech_words = []
        for s in spec.skills.must_have + spec.skills.preferred:
            tech_words.append(s.name)
        # Add required domains
        tech_words.extend(spec.domains.required)
        # Clean and unique list
        unique_tech = list(dict.fromkeys([t.replace("_", " ").title() for t in tech_words if t]))
        technology_query = " ".join(unique_tech)

        # 4. Concept Query: derived high-level conceptual descriptors
        concept_query = " ".join(spec.search_parameters.semantic_concepts)

        # 5. Skill Query: space-separated canonical skill tags
        skill_query = " ".join(dict.fromkeys([s.name for s in spec.skills.must_have]))

        return {
            "primary_query": primary,
            "expanded_query": expanded,
            "technology_query": technology_query,
            "concept_query": concept_query,
            "skill_query": skill_query
        }
