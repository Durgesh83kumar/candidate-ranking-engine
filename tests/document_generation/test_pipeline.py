import unittest
import tempfile
import os
import pandas as pd

from src.document_generation.cleaner import DocumentCleaner
from src.document_generation.templates import SemanticTemplateEngine
from src.document_generation.compactor import LengthCompactor
from src.document_generation.pipeline import process_single_candidate, DocumentGenerationPipeline

class TestDocumentGeneration(unittest.TestCase):

    def setUp(self):
        self.candidate_data = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "current_title": "Software Engineer",
                "current_title_normalized": "Software Engineer",
                "headline": "AI Enthusiast",
                "summary": "References available upon request. Seeking a challenging role.",
                "years_of_experience": 5.0,
                "years_of_experience_calculated": 5.0,
                "location": "Noida",
                "location_normalized": "Noida",
                "country": "India",
                "country_normalized": "India"
            },
            "skills": [
                {"name": "python", "name_normalized": "python", "proficiency": "expert", "is_ai_skill": False},
                {"name": "Python", "name_normalized": "python", "proficiency": "beginner", "is_ai_skill": False},
                {"name": "pytorch", "name_normalized": "pytorch", "proficiency": "advanced", "is_ai_skill": True, "duration_months": 24, "endorsements": 10}
            ],
            "career_history": [
                {
                    "company": "Company A",
                    "title": "Developer",
                    "title_normalized": "Developer",
                    "start_date": "2024-01-01",
                    "duration_months": 12,
                    "description": "Built search systems at scale. references available upon request."
                },
                {
                    "company": "Company B",
                    "title": "Intern",
                    "title_normalized": "Intern",
                    "start_date": "2023-01-01",
                    "duration_months": 6,
                    "description": "Learnt Python."
                }
            ],
            "education": [
                {
                    "degree": "BTech",
                    "degree_normalized": "bachelor",
                    "field_of_study": "Computer Science",
                    "institution": "IIT Noida",
                    "tier": "tier_1"
                }
            ],
            "languages": [
                {"language": "English", "proficiency": "fluent"}
            ],
            "redrob_signals": {
                "open_to_work_flag": True,
                "notice_period_days": 30,
                "profile_completeness_score": 95.0,
                "github_activity_score": 85.0
            }
        }

    def test_cleaner_noise_and_deduplication(self):
        cleaner = DocumentCleaner()
        
        # Test noise filter
        cleaned_boilerplate = cleaner.clean_text("References available upon request")
        self.assertEqual(cleaned_boilerplate, "")
        
        # Test skills deduplication
        deduplicated = cleaner.deduplicate_skills(self.candidate_data["skills"])
        self.assertEqual(len(deduplicated), 2)
        # Verify it kept the "expert" Python entry instead of the "beginner" one
        python_skill = next(s for s in deduplicated if s["name_normalized"] == "python")
        self.assertEqual(python_skill["proficiency"], "expert")

    def test_template_engine(self):
        engine = SemanticTemplateEngine()
        
        # Check overall v2 builder
        doc_str = engine.build_document(self.candidate_data)
        self.assertIn("# PROFESSIONAL PROFILE", doc_str)
        self.assertIn("# TECHNICAL SKILLS", doc_str)
        self.assertIn("# CAREER TIMELINE", doc_str)
        self.assertIn("# ACADEMIC FOUNDATIONS", doc_str)
        
        # Verify natural language timeline text
        self.assertIn("Role 1: Developer at Company A (Duration: 1 years)", doc_str)
        self.assertIn("IIT Noida (Tier classification: Tier 1)", doc_str)

    def test_compactor_budget(self):
        compactor = LengthCompactor(target_token_limit=130) # Small budget
        doc_v2_str = "A very long text " * 100
        
        # Estimate token check
        est = compactor.estimate_tokens(doc_v2_str)
        self.assertGreater(est, 130)
        
        # Compact run
        compacted = compactor.compact(self.candidate_data, doc_v2_str)
        self.assertLessEqual(compactor.estimate_tokens(compacted), 130)
        self.assertIn("Position details truncated for length", compacted)

    def test_process_single_candidate(self):
        doc_rec, meta_rec = process_single_candidate(self.candidate_data, 1024)
        
        # Verify search documents
        self.assertIn("candidate_id", doc_rec)
        self.assertIn("search_document_v1", doc_rec)
        self.assertIn("search_document_v2", doc_rec)
        self.assertIn("search_document_v2_e5", doc_rec)
        self.assertTrue(doc_rec["search_document_v2_e5"].startswith("passage: "))
        
        # Verify metadata extraction
        self.assertEqual(meta_rec["years_of_experience_calculated"], 5.0)
        self.assertTrue(meta_rec["open_to_work_flag"])
        self.assertEqual(meta_rec["notice_period_days"], 30)

if __name__ == "__main__":
    unittest.main()
