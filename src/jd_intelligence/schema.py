from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class RoleSpecification(BaseModel):
    title: str = Field(description="Role job title.")
    seniority: str = Field(description="Role seniority level: junior, mid, senior, lead, principal, manager.")

class ExperienceSpecification(BaseModel):
    min_years: float = Field(default=0.0, description="Minimum required years of experience.")
    ideal_years: float = Field(default=0.0, description="Ideal targeted years of experience.")
    max_years: Optional[float] = Field(default=None, description="Maximum experience limit if applicable.")
    require_production_experience: bool = Field(default=False, description="Whether production deployment experience is required.")
    require_startup_experience: bool = Field(default=False, description="Whether startup company experience is required.")
    leadership_required: bool = Field(default=False, description="Whether leadership/mentorship is required.")

class MustHaveSkill(BaseModel):
    name: str = Field(description="Normalized canonical skill name.")
    minimum_duration_months: int = Field(default=0, description="Minimum months of experience with this skill.")
    priority: str = Field(default="important", description="Priority level: critical, very_important, important.")

class PreferredSkill(BaseModel):
    name: str = Field(description="Normalized canonical skill name.")
    priority: str = Field(default="useful", description="Priority level: important, useful, bonus.")

class SkillsSpecification(BaseModel):
    must_have: List[MustHaveSkill] = Field(default_factory=list, description="Mandatory/must-have skills.")
    preferred: List[PreferredSkill] = Field(default_factory=list, description="Preferred/nice-to-have skills.")
    optional: List[str] = Field(default_factory=list, description="Optional skills.")

class DomainsSpecification(BaseModel):
    required: List[str] = Field(default_factory=list, description="Required domains (e.g. NLP, Search).")
    preferred: List[str] = Field(default_factory=list, description="Preferred domains.")

class ResponsibilityItem(BaseModel):
    description: str = Field(description="Key responsibility description.")
    category: str = Field(description="Function category: architecture, ml_modeling, data_engineering, deployment, mentoring.")

class EducationSpecification(BaseModel):
    min_degree: str = Field(default="none", description="Minimum degree level: bachelor, master, doctorate, none.")
    preferred_fields: List[str] = Field(default_factory=list, description="Preferred fields of study.")
    preferred_tiers: List[str] = Field(default_factory=list, description="Preferred university tiers: tier_1, tier_2, tier_3.")

class NegativePreferences(BaseModel):
    technologies: List[str] = Field(default_factory=list, description="Penalized or deprecated technologies.")
    behaviors: List[str] = Field(default_factory=list, description="Undesirable developer behaviors.")
    roles: List[str] = Field(default_factory=list, description="Penalized career backgrounds (e.g. pure research, consulting-only).")

class PreferencesSpecification(BaseModel):
    education: EducationSpecification = Field(default_factory=EducationSpecification, description="Educational backgrounds parameters.")
    industries: List[str] = Field(default_factory=list, description="Target corporate industries (e.g. SaaS, Fintech).")
    location: str = Field(default="anywhere", description="Geographic/relocation office hub parameters.")
    work_mode: str = Field(default="flexible", description="remote, hybrid, onsite, flexible.")
    max_notice_period_days: int = Field(default=180, description="Maximum acceptable notice period.")
    negative_preferences: NegativePreferences = Field(default_factory=NegativePreferences, description="Negative features to avoid.")

class SearchParameters(BaseModel):
    keywords: List[str] = Field(default_factory=list, description="Key search words for BM25 and term matching.")
    semantic_concepts: List[str] = Field(default_factory=list, description="Derived conceptual tags for semantic expansion.")

class RankingPriorityItem(BaseModel):
    dimension: str = Field(description="Ranking feature: skills, experience, signals, education, cultural.")
    weight_class: str = Field(description="Weight class: critical, very_important, important, useful, bonus.")

class ExplainabilityItem(BaseModel):
    rationale: str = Field(description="Architectural reason why this requirement was selected.")
    source_quote: str = Field(description="Raw text snippet from the JD document.")

class HiringSpecification(BaseModel):
    role: RoleSpecification
    experience: ExperienceSpecification
    skills: SkillsSpecification
    domains: DomainsSpecification
    responsibilities: List[ResponsibilityItem] = Field(default_factory=list)
    preferences: PreferencesSpecification
    cultural_expectations: List[str] = Field(default_factory=list)
    search_parameters: SearchParameters
    ranking_priorities: List[RankingPriorityItem] = Field(default_factory=list)
    explainability: Dict[str, ExplainabilityItem] = Field(default_factory=dict)
