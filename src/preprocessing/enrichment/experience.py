import re
from datetime import date
from typing import Dict, Any, List, Set
from src.preprocessing.exceptions import ExperienceCalculationError

class ExperienceCalculator:
    """Computes advanced career metrics, including net/relevant experience, production scores, and career consistency."""

    SENIORITY_WEIGHTS = {
        "junior": 1,
        "associate": 1,
        "mid": 2,
        "senior": 3,
        "lead": 4,
        "principal": 4,
        "architect": 5,
        "manager": 5,
        "director": 5,
        "head": 5
    }

    PRODUCTION_KEYWORDS = {
        "deployment": ["docker", "kubernetes", "k8s", "bentoml", "triton", "sagemaker", "mlflow", "serving", "deployment", "deploying"],
        "pipelines": ["kafka", "spark", "airflow", "feature store", "feast", "streaming", "pipelines", "pipeline"],
        "monitoring": ["prometheus", "grafana", "observability", "evidentlyai", "monitoring", "logging", "alerting"],
        "scale": ["latency", "qps", "throughput", "scale", "performance", "high-availability", "microservices"]
    }

    AI_RELEVANT_KEYWORDS = [
        "ai", "machine learning", "ml", "nlp", "llm", "deep learning", "computer vision", 
        "neural network", "transformer", "generative ai", "genai", "prompt engineering"
    ]

    def _calculate_overlap_free_duration(self, intervals: List[tuple]) -> float:
        """Helper to compute total years from list of date intervals (start, end), merging overlaps."""
        if not intervals:
            return 0.0
            
        # Sort intervals by start date
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        
        merged = []
        for current in sorted_intervals:
            if not merged:
                merged.append(current)
            else:
                last_start, last_end = merged[-1]
                curr_start, curr_end = current
                
                if curr_start <= last_end:
                    # Overlap found - merge by updating end date
                    merged[-1] = (last_start, max(last_end, curr_end))
                else:
                    merged.append(current)
                    
        total_days = sum((interval[1] - interval[0]).days for interval in merged)
        return total_days / 365.25

    def calculate_experience_metrics(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Performs all career and experience enrichment computations on the candidate record."""
        career_history = candidate_data.get("career_history", [])
        if not career_history:
            raise ExperienceCalculationError("Candidate record lacks career history.")

        # Sort career history chronologically (oldest to newest)
        career_history = sorted(career_history, key=lambda x: x.get("start_date_parsed", date.min))
        
        all_intervals = []
        ai_intervals = []
        
        # Track seniorities and industry sectors for consistency
        seniority_sequence = []
        industries = []
        
        # Production AI signals counters
        prod_matches_set = set()
        prod_jobs_count = 0
        total_description_words = 0
        total_action_verbs = 0
        
        # Action verbs for description quality check
        action_verbs = {"designed", "built", "implemented", "developed", "led", "architected", "optimized", "managed", "deployed", "scaled", "created", "integrated"}

        for job in career_history:
            start_date = job.get("start_date_parsed")
            end_date = job.get("end_date_parsed")
            
            if not start_date or not end_date:
                continue
                
            interval = (start_date, end_date)
            all_intervals.append(interval)
            
            # Map duration in years
            job_duration_years = (end_date - start_date).days / 365.25
            
            # Check relevance to AI
            title = job.get("title_normalized", "").lower()
            desc = job.get("description", "").lower()
            
            is_ai_job = any(kw in title or kw in desc for kw in self.AI_RELEVANT_KEYWORDS)
            if is_ai_job:
                ai_intervals.append(interval)

            # Title seniority weight tracking
            job_seniority = 2 # Mid default
            for word, weight in self.SENIORITY_WEIGHTS.items():
                if word in title:
                    job_seniority = max(job_seniority, weight)
            seniority_sequence.append(job_seniority)
            
            # Industry tracking
            industry = job.get("industry")
            if industry:
                industries.append(industry)
                
            # Production AI keyword scanning
            job_prod_keywords = 0
            for category, kw_list in self.PRODUCTION_KEYWORDS.items():
                for kw in kw_list:
                    if kw in desc or kw in title:
                        prod_matches_set.add(kw)
                        job_prod_keywords += 1
            if job_prod_keywords >= 2:
                prod_jobs_count += 1
                
            # Action verbs count
            words = desc.split()
            total_description_words += len(words)
            total_action_verbs += sum(1 for w in words if w.strip(",.()").lower() in action_verbs)

        # Net Calculations
        net_exp_years = self._calculate_overlap_free_duration(all_intervals)
        net_ai_exp_years = self._calculate_overlap_free_duration(ai_intervals)
        
        # Store in candidate profile
        profile = candidate_data.setdefault("profile", {})
        profile["years_of_experience_calculated"] = round(net_exp_years, 2)
        profile["years_of_relevant_ai_experience"] = round(net_ai_exp_years, 2)

        # AI Career Indicators
        ai_career_score = net_ai_exp_years / max(net_exp_years, 0.1)
        profile["ai_career_score"] = round(min(1.0, max(0.0, ai_career_score)), 2)
        
        skills = candidate_data.get("skills", [])
        total_skills = len(skills)
        ai_skills = sum(1 for s in skills if s.get("is_ai_skill") is True)
        profile["ai_skill_ratio"] = round(ai_skills / max(total_skills, 1), 2)
        
        # Domain Specific AI Scores
        llm_keywords = ["llm", "large language model", "rag", "langchain", "prompt", "gpt", "transformer"]
        nlp_keywords = ["nlp", "text", "translation", "ner", "parsing", "nltk", "spacy"]
        ml_keywords = ["classification", "regression", "xgboost", "scikit-learn", "random forest", "tabular"]
        
        profile["llm_experience_score"] = round(self._calculate_overlap_free_duration([
            intv for idx, intv in enumerate(all_intervals) 
            if any(kw in career_history[idx].get("description", "").lower() for kw in llm_keywords)
        ]), 2)
        
        profile["nlp_experience_score"] = round(self._calculate_overlap_free_duration([
            intv for idx, intv in enumerate(all_intervals) 
            if any(kw in career_history[idx].get("description", "").lower() for kw in nlp_keywords)
        ]), 2)
        
        profile["ml_experience_score"] = round(self._calculate_overlap_free_duration([
            intv for idx, intv in enumerate(all_intervals) 
            if any(kw in career_history[idx].get("description", "").lower() for kw in ml_keywords)
        ]), 2)

        # Production Experience Indicators
        total_production_terms = sum(len(kw_list) for kw_list in self.PRODUCTION_KEYWORDS.values())
        production_ai_score = len(prod_matches_set) / max(total_production_terms, 1)
        profile["production_ai_score"] = round(production_ai_score, 2)
        profile["production_ai_experience_year_count"] = round(
            sum(
                (career_history[idx].get("end_date_parsed") - career_history[idx].get("start_date_parsed")).days / 365.25
                for idx, job in enumerate(career_history)
                if sum(1 for cat, kw_list in self.PRODUCTION_KEYWORDS.items() for kw in kw_list if kw in job.get("description", "").lower()) >= 2
            ), 2
        )

        # Career Consistency Indicators
        # 1. Progression Score (Slope of seniority levels over career steps)
        if len(seniority_sequence) >= 2:
            # Simple slope calculations (last seniority - first seniority) / steps
            progression_score = (seniority_sequence[-1] - seniority_sequence[0]) / (len(seniority_sequence) - 1)
        else:
            progression_score = 0.0
        profile["career_progression_score"] = round(progression_score, 2)

        # 2. Job Hopping Score (average months per job)
        num_jobs = max(len(career_history), 1)
        job_hopping_score = (net_exp_years * 12) / num_jobs
        profile["job_hopping_score"] = round(job_hopping_score, 2)

        # 3. Domain Consistency (Ratio of jobs in the primary industry)
        if industries:
            primary_industry_count = max(industries.count(ind) for ind in set(industries))
            domain_consistency = primary_industry_count / len(industries)
        else:
            domain_consistency = 1.0
        profile["domain_consistency"] = round(domain_consistency, 2)

        # 4. Consulting Ratio
        consulting_jobs = sum(1 for job in career_history if "consult" in job.get("title_normalized", "").lower() or "freelance" in job.get("title_normalized", "").lower())
        profile["consulting_ratio"] = round(consulting_jobs / num_jobs, 2)

        # 5. Technical Depth Score
        tech_words_count = sum(1 for s in skills for job in career_history if s.get("name_normalized") in job.get("description", "").lower())
        profile["technical_depth_score"] = round(tech_words_count / max(total_description_words, 1), 3)

        # Profile Quality Indicators
        redrob_signals = candidate_data.setdefault("redrob_signals", {})
        completeness = float(redrob_signals.get("profile_completeness_score", 0))
        
        # Verification strength (0 to 3 score scaled to 0-100)
        email_v = 1 if redrob_signals.get("verified_email") is True else 0
        phone_v = 1 if redrob_signals.get("verified_phone") is True else 0
        linkedin_v = 1 if redrob_signals.get("linkedin_connected") is True else 0
        verification_strength = ((email_v + phone_v + linkedin_v) / 3.0) * 100.0
        
        # Documentation/Description quality based on description words and action verbs density
        avg_description_len = total_description_words / num_jobs
        desc_quality = min(100.0, (avg_description_len / 150.0) * 50.0 + (total_action_verbs / max(num_jobs, 1)) * 10.0)
        
        # Weighted Profile Quality Score
        profile_quality = (0.4 * completeness) + (0.3 * desc_quality) + (0.3 * verification_strength)
        profile["profile_quality_score"] = round(profile_quality, 2)
        
        return candidate_data
