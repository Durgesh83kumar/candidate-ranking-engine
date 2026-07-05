import unittest
import pandas as pd
import tempfile
import os
import json

from src.ranking.config import RankingConfig
from src.ranking.career_quality import CareerQualityEvaluator
from src.ranking.profile_quality import ProfileQualityEvaluator
from src.ranking.business_rules import BusinessRulesEngine
from src.ranking.honeypot import HoneypotDetector
from src.ranking.scorer import HybridScorer
from src.ranking.reasoning import ReasoningEngine
from src.ranking.submission import SubmissionGenerator
from src.ranking.cli import run_ranking_pipeline

class TestRankingPipeline(unittest.TestCase):

    def setUp(self):
        # Setup mock configs
        self.config = RankingConfig()
        
        # Candidate mock row data
        self.mock_candidate = {
            "candidate_id": "CAND_RANK_001",
            "rrf_score": 0.08,
            "cross_encoder_probability": 0.85,
            "years_of_experience": 6.0,
            "current_title": "AI Specialist",
            "current_company": "BigTech",
            "location": "Noida",
            "country": "India",
            "work_mode": "hybrid",
            "relocation": True,
            "notice_period_days": 15,
            "expected_salary_lpa": 30.0,
            "matched_profile_sections": ["Matched Skills", "Matched Summary", "Matched Education"],
            "profile": {
                "career_progression_score": 80.0,
                "job_hopping_score": 20.0,
                "years_of_relevant_ai_experience": 5.0,
                "technical_depth_score": 85.0
            },
            "redrob_signals": {
                "profile_completeness_score": 90.0,
                "github_activity_score": 85.0
            },
            "skills": ["python", "pytorch", "transformers"],
            "education": [{"degree": "BTech Computer Science", "end_date": 2022}],
            "certifications": ["AWS ML Certification"],
            "career_history": [
                {"job_title": "AI Dev", "start_date": 2022, "end_date": 2024, "description": "built neural networks"}
            ]
        }

    def test_career_scoring(self):
        evaluator = CareerQualityEvaluator()
        score = evaluator.evaluate(self.mock_candidate)
        self.assertTrue(0.0 <= score <= 1.0)
        # Check that high relevance gives high score
        self.assertTrue(score > 0.6)

    def test_profile_scoring(self):
        evaluator = ProfileQualityEvaluator()
        score = evaluator.evaluate(self.mock_candidate)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertTrue(score > 0.6)

    def test_business_penalties(self):
        engine = BusinessRulesEngine(self.config)
        
        # Test candidate passing rules
        multiplier, warnings = engine.evaluate(self.mock_candidate)
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(len(warnings), 0)
        
        # Test candidate exceeding notice period and salary cap
        bad_cand = dict(self.mock_candidate)
        bad_cand["notice_period_days"] = 90
        bad_cand["expected_salary_lpa"] = 70.0 # ceiling is 50.0
        
        multiplier, warnings = engine.evaluate(bad_cand)
        self.assertTrue(multiplier < 1.0)
        self.assertIn("Notice Period Exceeded (90 days)", warnings)
        self.assertIn("Salary Ceiling Exceeded (70.0 LPA)", warnings)

    def test_honeypot_detection(self):
        detector = HoneypotDetector(self.config.honeypot_multipliers)
        
        # Clean candidate passes checks
        multiplier, anomalies = detector.scan_candidate(self.mock_candidate)
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(len(anomalies), 0)
        
        # Fake candidate with impossible tech creation timeline check
        fake_cand = dict(self.mock_candidate)
        fake_cand["career_history"] = [
            # claiming LLaMA experience in 2015 (invented in 2023)
            {"job_title": "ML Engineer", "start_date": 2012, "end_date": 2015, "description": "fine-tuned llama models"}
        ]
        
        multiplier, anomalies = detector.scan_candidate(fake_cand)
        self.assertTrue(multiplier < 1.0)
        self.assertIn("Impossible Skill Claim: Work history details 'llama' in a job ending in 2015 (invented in 2023).", anomalies)

    def test_hybrid_score_calculation(self):
        scorer = HybridScorer(self.config)
        output = scorer.compute_hybrid_score(self.mock_candidate, max_rrf_score=0.10)
        
        self.assertEqual(output["candidate_id"], "CAND_RANK_001")
        self.assertTrue("final_score" in output)
        self.assertTrue(0.0 <= output["final_score"] <= 1.0)

    def test_reasoning_generation(self):
        engine = ReasoningEngine()
        cand = dict(self.mock_candidate)
        cand["explainability_evidence"] = {
            "Matched AI Experience": True,
            "Matched Search/Retrieval Experience": True,
            "Matched Leadership": True,
            "Matched Production Experience": True
        }
        reasoning = engine.generate_reasoning(cand)
        
        self.assertTrue(len(reasoning) > 0)
        words = reasoning.split()
        self.assertTrue(10 <= len(words) <= 50)

    def test_submission_and_monotonicity(self):
        # Create a list of 100 mock candidates with descending scores
        ranked_pool = []
        for idx in range(100):
            ranked_pool.append({
                "candidate_id": f"CAND_{idx:03d}",
                "final_score": 0.95 - (idx * 0.005),
                "reasoning": "Proven experience in software engineering and technical system design. High alignment to core python development and skills requirements."
            })
            
        generator = SubmissionGenerator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "submission.csv")
            generator.validate_and_save(ranked_pool, csv_path)
            
            # Verify file exists and has correct rows
            self.assertTrue(os.path.exists(csv_path))
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 100)
            self.assertEqual(df.loc[0, "candidate_id"], "CAND_000")
            self.assertEqual(df.loc[99, "candidate_id"], "CAND_099")
            self.assertEqual(df.loc[0, "rank"], 1)
            self.assertEqual(df.loc[99, "rank"], 100)

if __name__ == "__main__":
    unittest.main()
