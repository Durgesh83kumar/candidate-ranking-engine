import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime

class HoneypotDetector:
    """Detects suspicious or mathematically impossible resume details and applies rank penalties."""

    def __init__(self, multipliers: Dict[int, float]):
        self.multipliers = multipliers

    def _parse_year(self, date_str: Any) -> int:
        """Helper to extract year integer from various date formats."""
        if not date_str:
            return 0
        try:
            # Check if it's already an integer/float
            if isinstance(date_str, (int, float)):
                return int(date_str)
            
            # String parsing
            date_str = str(date_str).strip()
            if len(date_str) >= 4:
                # Try parsing as year directly
                if date_str.isdigit():
                    return int(date_str)
                # Try standard pandas datetime parsing
                dt = pd.to_datetime(date_str, errors="coerce")
                if not pd.isna(dt):
                    return dt.year
        except Exception:
            pass
        return 0

    def check_temporal_inconsistencies(self, edu_list: List[Dict[str, Any]], jobs: List[Dict[str, Any]]) -> List[str]:
        """Flags cases where work experience starts significantly before university graduation."""
        anomalies = []
        if not edu_list or not jobs:
            return anomalies
            
        # Get earliest graduation year (excluding high school/secondary school if specified)
        grad_years = []
        for edu in edu_list:
            degree = str(edu.get("degree", "")).lower()
            if "school" in degree or "high school" in degree or "matric" in degree:
                continue
            end_val = edu.get("end_date") or edu.get("end_year")
            y = self._parse_year(end_val)
            if y > 1980:
                grad_years.append(y)
                
        if not grad_years:
            return anomalies
            
        earliest_grad = min(grad_years)
        
        # Check start dates of jobs
        for job in jobs:
            # Skip internships
            job_title = str(job.get("job_title", "")).lower()
            job_desc = str(job.get("description", "")).lower()
            if "intern" in job_title or "intern" in job_desc or "trainee" in job_title:
                continue
                
            start_val = job.get("start_date") or job.get("start_year")
            y_start = self._parse_year(start_val)
            if y_start > 1980:
                # If work started more than 3 years before university graduation
                if earliest_grad - y_start > 3:
                    anomalies.append(
                        f"Temporal anomaly: Career job '{job.get('job_title')}' started in {y_start} "
                        f"which is {earliest_grad - y_start} years before university graduation in {earliest_grad}."
                    )
                    break
        return anomalies

    def check_impossible_skills(self, jobs: List[Dict[str, Any]]) -> List[str]:
        """Flags resumes claiming experience with recently-invented technologies before their creation date."""
        anomalies = []
        if not jobs:
            return anomalies
            
        # Tech invention thresholds
        # technology -> creation_year
        recent_techs = {
            "llama": 2023,
            "langchain": 2022,
            "llamaindex": 2022,
            "gpt-4": 2023,
            "gpt4": 2023,
            "pinecone": 2020,
            "milvus": 2019,
            "vector database": 2018,
            "large language model": 2018,
            "llm": 2018,
            "rag": 2020,
            "retrieval augmented generation": 2020,
            "transformer": 2017,  # Attention Is All You Need is 2017
            "bge-m3": 2023
        }
        
        for job in jobs:
            end_val = job.get("end_date") or job.get("end_year") or job.get("start_date")
            y_end = self._parse_year(end_val)
            if y_end <= 1980:
                continue
                
            job_title = str(job.get("job_title", "")).lower()
            job_desc = str(job.get("description", "")).lower()
            combined_text = f"{job_title} {job_desc}"
            
            for tech, invention_year in recent_techs.items():
                if tech in combined_text:
                    # If the job ended before the technology was even invented!
                    if y_end < invention_year:
                        anomalies.append(
                            f"Impossible Skill Claim: Work history details '{tech}' in a job "
                            f"ending in {y_end} (invented in {invention_year})."
                        )
                        break
        return anomalies

    def check_timeline_overlaps(self, jobs: List[Dict[str, Any]]) -> List[str]:
        """Flags concurrent overlapping full-time employment timelines."""
        anomalies = []
        if not jobs or len(jobs) < 2:
            return anomalies
            
        parsed_intervals = []
        for job in jobs:
            # Skip freelance/part-time
            title = str(job.get("job_title", "")).lower()
            desc = str(job.get("description", "")).lower()
            if any(k in title or k in desc for k in ["freelance", "part-time", "contract", "consultant"]):
                continue
                
            start_val = job.get("start_date") or job.get("start_year")
            end_val = job.get("end_date") or job.get("end_year")
            
            y_start = self._parse_year(start_val)
            y_end = self._parse_year(end_val) if end_val else datetime.now().year
            
            if y_start > 1980 and y_end >= y_start:
                parsed_intervals.append((y_start, y_end, job.get("job_title", "")))
                
        # Look for concurrent overlapping timelines
        for i in range(len(parsed_intervals)):
            for j in range(i + 1, len(parsed_intervals)):
                s1, e1, t1 = parsed_intervals[i]
                s2, e2, t2 = parsed_intervals[j]
                
                # Check for direct overlap of more than 1 year (to allow small transition buffer)
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                
                if overlap_end - overlap_start > 1:
                    anomalies.append(
                        f"Employment timeline overlap: Concurrent full-time roles "
                        f"'{t1}' ({s1}-{e1}) and '{t2}' ({s2}-{e2}) overlap by {overlap_end - overlap_start} years."
                    )
                    break
        return anomalies

    def scan_candidate(self, candidate_record: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Scans candidate timeline records and returns multiplier factor + warning reasons."""
        edu = candidate_record.get("education", [])
        jobs = candidate_record.get("career_history", [])
        
        # Standardize PyArrow nested objects
        if hasattr(edu, "tolist"):
            edu = edu.tolist()
        if hasattr(jobs, "tolist"):
            jobs = jobs.tolist()
            
        anomalies = []
        
        # 1. Temporal Check
        anomalies.extend(self.check_temporal_inconsistencies(edu, jobs))
        
        # 2. Impossible Skill claims Check
        anomalies.extend(self.check_impossible_skills(jobs))
        
        # 3. Timeline Overlaps Check
        anomalies.extend(self.check_timeline_overlaps(jobs))
        
        # Apply score penalty multiplier
        anomaly_count = len(anomalies)
        multiplier = self.multipliers.get(anomaly_count, 0.10)
        if anomaly_count > 3:
            multiplier = 0.10
            
        return float(multiplier), anomalies
