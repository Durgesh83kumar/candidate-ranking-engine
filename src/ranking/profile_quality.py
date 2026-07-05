from typing import Dict, Any, List

class ProfileQualityEvaluator:
    """Computes resume completeness, GitHub activity, skill richness, and educational signals in [0, 1]."""

    def evaluate(self, candidate_record: Dict[str, Any]) -> float:
        """Evaluates profile signals.
        
        Args:
            candidate_record: Raw candidate profile dictionary containing education, skills, and redrob_signals.
            
        Returns:
            float: Normalized profile quality score.
        """
        # Extract profiles signals
        signals = candidate_record.get("redrob_signals", {})
        if not signals:
            signals = candidate_record.get("profile", {})
            
        # 1. Profile Completeness
        comp_raw = signals.get("profile_completeness_score", 50.0)
        s_completeness = float(comp_raw) / 100.0
        
        # 2. GitHub Activity
        github_raw = signals.get("github_activity_score", 0.0)
        s_github = float(github_raw) / 100.0
        
        # 3. Skills Richness
        skills = candidate_record.get("skills", [])
        s_skills = min(1.0, len(skills) / 15.0) if skills else 0.0
        
        # 4. Education presence
        edu = candidate_record.get("education", [])
        s_education = 1.0 if (isinstance(edu, list) and len(edu) > 0) else 0.0
        
        # 5. Certifications presence
        certs = candidate_record.get("certifications", [])
        s_certs = 1.0 if (isinstance(certs, list) and len(certs) > 0) else 0.0
        
        # Weighted Aggregation
        w_comp, w_git, w_skills, w_edu, w_certs = 0.40, 0.20, 0.20, 0.10, 0.10
        score = (w_comp * s_completeness) + (w_git * s_github) + (w_skills * s_skills) + (w_edu * s_education) + (w_certs * s_certs)
        
        return max(0.0, min(1.0, float(score)))
