import unittest
import os
import tempfile
import pandas as pd

from src.verification.config import VerificationConfig
from src.verification.phrase_scanner import PhraseScanner
from src.verification.skills_validator import SkillsValidator
from src.verification.ai_specialist import AISpecialist
from src.verification.calibrator import ScoreCalibrator
from src.verification.cli import validate_and_save_submission

class TestVerificationPipeline(unittest.TestCase):

    def setUp(self):
        self.config = VerificationConfig()
        
        # Temp hiring specification
        self.spec_dir = tempfile.TemporaryDirectory()
        self.spec_path = os.path.join(self.spec_dir.name, "hiring_specification.json")
        mock_spec = {
            "skills": {
                "must_have": [
                    {"name": "python"},
                    {"name": "large_language_models"}
                ]
            }
        }
        import json
        with open(self.spec_path, "w", encoding="utf-8") as f:
            json.dump(mock_spec, f)
            
        self.mock_candidate = {
            "candidate_id": "CAND_VER_001",
            "final_score": 0.85,
            "reasoning": "Proven experience in software engineering and system design. Passed all candidate verification and skills checks.",
            "skills": ["python", "large_language_models", "pytorch"],
            "search_document_v2": "Deployed python microservices and trained large language models with pytorch.",
            "profile": {
                "years_of_relevant_ai_experience": 4.0
            },
            "honeypot_multiplier": 1.0,
            "triggered_honeypot_checks": []
        }

    def tearDown(self):
        self.spec_dir.cleanup()

    def test_phrase_scanner(self):
        scanner = PhraseScanner(self.config)
        
        # Engineering boost
        mult, has_rec, has_eng = scanner.scan("Deployed python microservices at scale.")
        self.assertTrue(has_eng)
        self.assertFalse(has_rec)
        self.assertAlmostEqual(mult, 1.05)
        
        # Recruiter penalty
        mult, has_rec, has_eng = scanner.scan("Hired python developers for a machine learning team.")
        self.assertTrue(has_rec)
        self.assertFalse(has_eng)
        self.assertAlmostEqual(mult, 0.70)

    def test_skills_validator(self):
        validator = SkillsValidator(self.config, self.spec_path)
        
        # Clean candidate passes must-haves
        mult, missing = validator.validate(["python", "large_language_models"], "Deployed python models")
        self.assertEqual(mult, 1.0)
        self.assertEqual(len(missing), 0)
        
        # Candidate missing skills
        mult, missing = validator.validate(["java"], "Wrote java microservices")
        self.assertAlmostEqual(mult, 0.20)
        self.assertIn("python", missing)
        self.assertIn("large_language_models", missing)

    def test_ai_specialist(self):
        specialist = AISpecialist(self.config)
        
        # Candidate claims AI and has PyTorch
        mult, matched = specialist.check_specialist(["pytorch"], "Uses PyTorch", claims_ai_experience=True)
        self.assertEqual(mult, 1.0)
        self.assertIn("pytorch", matched)
        
        # Candidate claims AI but lacks frameworks
        mult, matched = specialist.check_specialist(["java"], "Uses Java", claims_ai_experience=True)
        self.assertAlmostEqual(mult, 0.80)
        self.assertEqual(len(matched), 0)

    def test_score_calibrator_clean(self):
        calibrator = ScoreCalibrator(self.config, self.spec_path)
        res = calibrator.calibrate_candidate(self.mock_candidate)
        
        self.assertEqual(res["candidate_id"], "CAND_VER_001")
        # Boost applied: 0.85 * 1.05 = 0.8925
        self.assertAlmostEqual(res["calibrated_score"], 0.8925)
        self.assertFalse(res["is_flaged_honeypot"])

    def test_score_calibrator_honeypot(self):
        calibrator = ScoreCalibrator(self.config, self.spec_path)
        bad_cand = dict(self.mock_candidate)
        bad_cand["triggered_honeypot_checks"] = ["Timeline overlap"]
        
        res = calibrator.calibrate_candidate(bad_cand)
        self.assertAlmostEqual(res["calibrated_score"], 0.0)
        self.assertTrue(res["is_flaged_honeypot"])
        self.assertIn("timeline inconsistencies", res["reasoning"])

    def test_submission_and_monotonicity(self):
        # Create a list of 100 mock candidates with descending scores
        ranked_pool = []
        for idx in range(100):
            ranked_pool.append({
                "candidate_id": f"CAND_{idx:03d}",
                "calibrated_score": 0.95 - (idx * 0.005),
                "reasoning": "Proven experience in software engineering and system design. Passed all candidate verification and skills checks."
            })
            
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "submission.csv")
            validate_and_save_submission(ranked_pool, csv_path)
            
            self.assertTrue(os.path.exists(csv_path))
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 100)
            self.assertEqual(df.loc[0, "candidate_id"], "CAND_000")
            self.assertEqual(df.loc[99, "candidate_id"], "CAND_099")
            self.assertEqual(df.loc[0, "rank"], 1)
            self.assertEqual(df.loc[99, "rank"], 100)

if __name__ == "__main__":
    unittest.main()
