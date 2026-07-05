from typing import List
from src.jd_intelligence.schema import HiringSpecification
from src.jd_intelligence.exceptions import JDValidationError

class SpecificationValidator:
    """Validates the logical consistency and completeness of structured hiring specifications."""

    def validate(self, spec: HiringSpecification) -> List[str]:
        """Validates a parsed HiringSpecification.
        
        Returns:
            List[str]: List of warnings or errors found.
        """
        warnings = []

        # 1. Chronological Experience Check
        exp = spec.experience
        if exp.min_years > exp.ideal_years:
            raise JDValidationError(
                f"Consistency error: minimum experience ({exp.min_years} years) "
                f"is greater than ideal experience ({exp.ideal_years} years)."
            )
            
        if exp.max_years is not None and exp.min_years > exp.max_years:
            raise JDValidationError(
                f"Consistency error: minimum experience ({exp.min_years} years) "
                f"is greater than maximum experience ({exp.max_years} years)."
            )

        # 2. Contradiction Check (Must-Have vs. Penalized tech)
        must_have_names = {s.name.lower() for s in spec.skills.must_have}
        preferred_names = {s.name.lower() for s in spec.skills.preferred}
        penalized_tech = {t.lower() for t in spec.preferences.negative_preferences.technologies}

        conflicting_must_have = must_have_names.intersection(penalized_tech)
        if conflicting_must_have:
            raise JDValidationError(
                f"Contradiction error: skills {conflicting_must_have} are listed as "
                f"both Must-Have requirements and Negative/Penalized preferences."
            )

        conflicting_preferred = preferred_names.intersection(penalized_tech)
        if conflicting_preferred:
            warnings.append(
                f"Conflicting Warning: skills {conflicting_preferred} are listed as "
                f"both Preferred requirements and Negative/Penalized preferences."
            )

        # 3. Experience & Seniority Sanity Checks
        role = spec.role
        if role.seniority == "junior" and exp.min_years > 3.0:
            warnings.append(
                f"Seniority Warning: role seniority is set to 'junior' but requires "
                f"minimum {exp.min_years} years of experience."
            )
        elif role.seniority in ["senior", "lead", "principal"] and exp.min_years < 3.0:
            warnings.append(
                f"Seniority Warning: role seniority is set to '{role.seniority}' but "
                f"requires only {exp.min_years} years of experience."
            )

        # 4. Incomplete/Ambiguous requirements warning
        if not spec.skills.must_have:
            warnings.append("Ambiguity Warning: No Must-Have skills were extracted from the job description.")
            
        if not spec.responsibilities:
            warnings.append("Completeness Warning: Responsibilities array is empty.")

        return warnings
