import unittest
import tempfile
import os
import json

from src.jd_intelligence.exceptions import JDValidationError
from src.jd_intelligence.parser import JDParser
from src.jd_intelligence.extractor import LlmExtractor
from src.jd_intelligence.validator import SpecificationValidator
from src.jd_intelligence.query_gen import QueryGenerator

class TestJDIntelligence(unittest.TestCase):

    def setUp(self):
        self.sample_jd_text = """
        Redrob Senior AI Engineer Job Description.
        Looking for a Senior AI Engineer.
        Must have: Production experience with embeddings-based retrieval systems and vector databases.
        Strong Python and experience with evaluation frameworks (NDCG, MRR).
        disqualifiers we actually apply: pure research environments or Hadoop.
        """

    def test_parser_txt(self):
        # Create temp text file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            f.write("Role: Senior AI Engineer\n5 years experience required.")
            temp_path = f.name
            
        try:
            parser = JDParser()
            text = parser.parse(temp_path)
            self.assertIn("Senior AI Engineer", text)
            self.assertIn("5 years", text)
        finally:
            os.remove(temp_path)

    def test_extractor_redrob_matching(self):
        extractor = LlmExtractor()
        spec = extractor.extract(self.sample_jd_text)
        
        self.assertEqual(spec.role.title, "Senior AI Engineer")
        self.assertEqual(spec.role.seniority, "senior")
        self.assertEqual(spec.experience.min_years, 5.0)
        self.assertTrue(spec.experience.require_production_experience)

        # Check mapped skills canonical names
        must_have_skills = [s.name for s in spec.skills.must_have]
        self.assertIn("large_language_models", must_have_skills) # normalized from embeddings-based retrieval
        self.assertIn("vector_databases", must_have_skills)
        self.assertIn("python", must_have_skills)

    def test_validator_chronology_error(self):
        extractor = LlmExtractor()
        spec = extractor.extract(self.sample_jd_text)
        
        # Cause chronological inconsistency
        spec.experience.min_years = 10.0
        spec.experience.ideal_years = 5.0 # Min > Ideal
        
        validator = SpecificationValidator()
        with self.assertRaises(JDValidationError):
            validator.validate(spec)

    def test_validator_contradiction_error(self):
        extractor = LlmExtractor()
        spec = extractor.extract(self.sample_jd_text)
        
        # Add conflict
        spec.preferences.negative_preferences.technologies = ["python"] # Python is Must-Have and Penalized
        
        validator = SpecificationValidator()
        with self.assertRaises(JDValidationError):
            validator.validate(spec)

    def test_query_generator(self):
        extractor = LlmExtractor()
        spec = extractor.extract(self.sample_jd_text)
        
        generator = QueryGenerator()
        queries = generator.generate(spec)
        
        self.assertIn("primary_query", queries)
        self.assertIn("expanded_query", queries)
        self.assertIn("technology_query", queries)
        self.assertIn("concept_query", queries)
        self.assertIn("skill_query", queries)
        
        self.assertEqual(queries["primary_query"], "Senior Senior AI Engineer specializing in NLP, Information Retrieval")

if __name__ == "__main__":
    unittest.main()
