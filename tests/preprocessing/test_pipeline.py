import unittest
import tempfile
import os
import json
from datetime import date

# Import modules under test
from src.preprocessing.exceptions import SchemaValidationError, CustomRuleValidationError
from src.preprocessing.reader import CandidateReader
from src.preprocessing.validator import CandidateSchemaValidator
from src.preprocessing.normalizers.text import TextNormalizer
from src.preprocessing.normalizers.skills import SkillNormalizer
from src.preprocessing.normalizers.dates import DateNormalizer
from src.preprocessing.enrichment.experience import ExperienceCalculator
from src.preprocessing.enrichment.text_aggregator import TextAggregator
from src.preprocessing.deduplicator import CandidateDeduplicator
from src.preprocessing.pipeline import PreprocessingPipeline

class TestPreprocessingPipeline(unittest.TestCase):
    
    def setUp(self):
        # Sample clean candidate data representing CAND_0000001
        self.sample_candidate = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Ira Vora",
                "headline": "Backend Engineer | SQL, Spark, Cloud",
                "summary": "Software engineer with 6.9 years of experience. Fine-tuning LLMs is my focus.",
                "location": "Toronto, Ontario",
                "country": "Canada",
                "years_of_experience": 6.9,
                "current_title": "Backend Engineer",
                "current_company": "Mindtree",
                "current_company_size": "10001+",
                "current_industry": "IT Services"
            },
            "career_history": [
                {
                    "company": "Mindtree",
                    "title": "Backend Engineer",
                    "start_date": "2024-03-08",
                    "end_date": None,
                    "duration_months": 27,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Implemented streaming data pipelines on Kafka and Spark. Worked on fine-tuning LLMs."
                },
                {
                    "company": "Dunder Mifflin",
                    "title": "Analytics Engineer",
                    "start_date": "2019-07-03",
                    "end_date": "2024-01-08",
                    "duration_months": 55,
                    "is_current": False,
                    "industry": "Paper Products",
                    "company_size": "201-500",
                    "description": "Built and maintained batch data pipelines on Airflow and Snowflake."
                }
            ],
            "education": [
                {
                    "institution": "Lovely Professional University",
                    "degree": "B.E.",
                    "field_of_study": "Computer Science",
                    "start_year": 2017,
                    "end_year": 2020,
                    "grade": "8.24 CGPA",
                    "tier": "tier_3"
                }
            ],
            "skills": [
                {
                    "name": "Fine-tuning LLMs",
                    "proficiency": "advanced",
                    "endorsements": 21,
                    "duration_months": 36
                },
                {
                    "name": "NLP",
                    "proficiency": "advanced",
                    "endorsements": 37,
                    "duration_months": 26
                },
                {
                    "name": "Photoshop",
                    "proficiency": "intermediate",
                    "endorsements": 8,
                    "duration_months": 24
                }
            ],
            "certifications": [],
            "languages": [
                {"language": "English", "proficiency": "professional"}
            ],
            "redrob_signals": {
                "profile_completeness_score": 86.9,
                "signup_date": "2025-10-16",
                "last_active_date": "2026-05-20",
                "open_to_work_flag": True,
                "profile_views_received_30d": 23,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.34,
                "avg_response_time_hours": 177.8,
                "skill_assessment_scores": {
                    "NLP": 38.8,
                    "Fine-tuning LLMs": 41.6
                },
                "connection_count": 356,
                "endorsements_received": 35,
                "notice_period_days": 60,
                "expected_salary_range_inr_lpa": {
                    "min": 18.7,
                    "max": 36.1
                },
                "preferred_work_mode": "onsite",
                "willing_to_relocate": False,
                "github_activity_score": 9.2,
                "search_appearance_30d": 249,
                "saved_by_recruiters_30d": 4,
                "interview_completion_rate": 0.71,
                "offer_acceptance_rate": 0.58,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": False
            }
        }

    def test_validator_valid_record(self):
        validator = CandidateSchemaValidator()
        is_valid, err_msg = validator.validate_candidate(self.sample_candidate)
        self.assertTrue(is_valid, f"Expected valid record, got error: {err_msg}")

    def test_validator_invalid_id_schema(self):
        bad_candidate = dict(self.sample_candidate)
        bad_candidate["candidate_id"] = "CAND_123" # Must be 7 digits
        validator = CandidateSchemaValidator()
        is_valid, err_msg = validator.validate_candidate(bad_candidate)
        self.assertFalse(is_valid)
        self.assertIn("Invalid candidate_id format", err_msg)

    def test_validator_custom_rules_date_chronology(self):
        bad_candidate = dict(self.sample_candidate)
        # Set start date after end date
        bad_candidate["career_history"] = [
            {
                "company": "Company A",
                "title": "Engineer",
                "start_date": "2024-03-08",
                "end_date": "2023-03-08",
                "duration_months": 12,
                "is_current": False,
                "industry": "IT",
                "company_size": "1-10",
                "description": "desc"
            }
        ]
        validator = CandidateSchemaValidator()
        is_valid, err_msg = validator.validate_candidate(bad_candidate)
        self.assertFalse(is_valid)
        self.assertIn("start_date '2024-03-08' is after end_date '2023-03-08'", err_msg)

    def test_text_normalizer(self):
        normalizer = TextNormalizer()
        candidate = normalizer.normalize(self.sample_candidate)
        
        # Verify current title mapping
        self.assertEqual(candidate["profile"]["current_title_normalized"], "Backend Engineer")
        
        # Verify location parsing
        self.assertEqual(candidate["profile"]["location_normalized"], "Toronto, Ontario")
        self.assertEqual(candidate["profile"]["country_normalized"], "CA")
        
        # Verify degree normalizations
        self.assertEqual(candidate["education"][0]["degree_normalized"], "Bachelor")

    def test_skill_normalizer(self):
        normalizer = SkillNormalizer()
        candidate = normalizer.normalize(self.sample_candidate)
        
        # Check canonical naming lookup
        skills = {s["name_normalized"]: s for s in candidate["skills"]}
        self.assertIn("llm_fine_tuning", skills)
        self.assertIn("natural_language_processing", skills)
        
        # Verify is_ai_skill flag
        self.assertTrue(skills["llm_fine_tuning"]["is_ai_skill"])
        self.assertFalse(skills["photoshop"]["is_ai_skill"])

    def test_date_normalizer(self):
        normalizer = DateNormalizer("2026-06-30")
        candidate = normalizer.normalize(self.sample_candidate)
        
        # Check active job maps to Reference Date
        active_job = candidate["career_history"][0]
        self.assertEqual(active_job["end_date"], "2026-06-30")
        self.assertEqual(active_job["end_date_parsed"], date(2026, 6, 30))

    def test_experience_calculator(self):
        # Inplace dependencies mapping first
        DateNormalizer("2026-06-30").normalize(self.sample_candidate)
        TextNormalizer().normalize(self.sample_candidate)
        SkillNormalizer().normalize(self.sample_candidate)

        calculator = ExperienceCalculator()
        candidate = calculator.calculate_experience_metrics(self.sample_candidate)
        
        # Assert net experience computed
        self.assertGreater(candidate["profile"]["years_of_experience_calculated"], 0)
        self.assertGreater(candidate["profile"]["years_of_relevant_ai_experience"], 0)
        self.assertIn("ai_career_score", candidate["profile"])
        self.assertIn("ai_skill_ratio", candidate["profile"])

    def test_text_aggregator(self):
        DateNormalizer("2026-06-30").normalize(self.sample_candidate)
        TextNormalizer().normalize(self.sample_candidate)
        SkillNormalizer().normalize(self.sample_candidate)

        aggregator = TextAggregator()
        candidate = aggregator.build_search_documents(self.sample_candidate)
        
        self.assertIn("search_document_v1", candidate)
        self.assertIn("search_document_v2", candidate)
        
        # Verify structuring content tags are present
        self.assertTrue(candidate["search_document_v2"].startswith("[TITLE]"))
        self.assertIn("[SUMMARY]", candidate["search_document_v2"])
        self.assertIn("[SKILLS]", candidate["search_document_v2"])
        self.assertIn("[EXPERIENCE]", candidate["search_document_v2"])

    def test_deduplicator(self):
        # Setup two similar candidates
        deduper = CandidateDeduplicator()
        
        cand_a = dict(self.sample_candidate)
        cand_b = dict(self.sample_candidate)
        cand_b["candidate_id"] = "CAND_0000002"
        
        # Set normalized values for deduplication checks
        cand_a["profile"]["location_normalized"] = "Toronto"
        cand_a["profile"]["country_normalized"] = "CA"
        cand_a["profile"]["current_title_normalized"] = "Backend Engineer"
        
        cand_b["profile"]["location_normalized"] = "Toronto"
        cand_b["profile"]["country_normalized"] = "CA"
        cand_b["profile"]["current_title_normalized"] = "Backend Engineer"
        
        # First candidate
        is_dup_a = deduper.track_and_check_duplicate(cand_a)
        self.assertFalse(is_dup_a)
        
        # Second candidate shares same composite bio
        is_dup_b = deduper.track_and_check_duplicate(cand_b)
        self.assertTrue(is_dup_b)

    def test_reader(self):
        # Create a temp file to test candidate reader streaming
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
            f.write(json.dumps({"candidate_id": "CAND_0000001"}) + "\n")
            f.write(json.dumps({"candidate_id": "CAND_0000002"}) + "\n")
            temp_path = f.name
            
        try:
            reader = CandidateReader(temp_path, batch_size=1)
            batches = list(reader.stream_raw_candidates())
            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[0][0]["candidate_id"], "CAND_0000001")
        finally:
            os.remove(temp_path)

if __name__ == "__main__":
    unittest.main()
