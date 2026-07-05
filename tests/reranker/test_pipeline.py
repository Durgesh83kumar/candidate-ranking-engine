import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import tempfile
import os
import json

from src.reranker.config import RerankerConfig
from src.reranker.pair_builder import RerankingPairBuilder
from src.reranker.scorer import BatchRerankingScorer
from src.reranker.feature_engineering import FeatureEngineeringManager
from src.reranker.explainability import ExplainabilityEngine
from src.reranker.evaluator import RerankerEvaluator
from src.reranker.cli import run_reranking_pipeline

class TestRerankerPipeline(unittest.TestCase):

    def setUp(self):
        # Setup mock specifications
        self.mock_spec = {
            "role": {"title": "Senior AI Engineer", "seniority": "senior"},
            "experience": {"min_years": 5.0, "max_years": 9.0, "require_production_experience": True},
            "skills": {
                "must_have": [{"name": "large_language_models"}, {"name": "python"}],
                "preferred": [{"name": "vector_databases"}]
            },
            "responsibilities": [{"description": "Own intelligence layer"}],
            "preferences": {
                "work_mode": "hybrid",
                "location": "Noida",
                "max_notice_period_days": 30,
                "negative_preferences": {
                    "technologies": ["hadoop"]
                }
            }
        }
        
        # Phase 5 mock candidates listing
        self.mock_candidates = [
            {
                "candidate_id": "CAND_001",
                "anonymized_name": "Alice",
                "rrf_score": 0.08,
                "confidence_score": 0.85,
                "years_of_experience": 6.0,
                "current_title": "AI Specialist",
                "current_company": "BigTech",
                "location": "Noida, UP",
                "country": "India",
                "work_mode": "hybrid",
                "relocation": True,
                "notice_period_days": 15,
                "profile_completeness": 90.0,
                "matched_profile_sections": ["Matched Skills", "Matched Summary"],
                "query_similarities": {"general": 0.82, "technical": 0.79},
                "query_ranks": {"general": 1, "technical": 3}
            },
            {
                "candidate_id": "CAND_002",
                "anonymized_name": "Bob",
                "rrf_score": 0.05,
                "confidence_score": 0.65,
                "years_of_experience": 3.0,
                "current_title": "Python Dev",
                "current_company": "SmallFirm",
                "location": "Delhi",
                "country": "India",
                "work_mode": "office",
                "relocation": False,
                "notice_period_days": 60,
                "profile_completeness": 70.0,
                "matched_profile_sections": ["Matched Education"],
                "query_similarities": {"general": 0.65},
                "query_ranks": {"general": 10}
            }
        ]
        
        # Phase 3 mock search documents
        self.mock_search_docs = pd.DataFrame([
            {"candidate_id": "CAND_001", "search_document_v2": "Alice profile details python large language models Noida"},
            {"candidate_id": "CAND_002", "search_document_v2": "Bob profile details legacy python developer"}
        ])

    def test_pair_builder_recruiter_query(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path = os.path.join(tmp_dir, "spec.json")
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(self.mock_spec, f)
                
            builder = RerankingPairBuilder(spec_path)
            query = builder.build_recruiter_query()
            
            # Check keywords are incorporated in query
            self.assertIn("Senior", query)
            self.assertIn("AI Engineer", query)
            self.assertIn("large_language_models", query)
            self.assertIn("Noida", query)
            self.assertIn("hadoop", query)
            
            # Construct pairs
            pairs = builder.construct_pairs(self.mock_candidates, self.mock_search_docs)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0][0], "CAND_001")
            self.assertEqual(pairs[0][1], query)
            self.assertEqual(pairs[0][2], "Alice profile details python large language models Noida")

    @patch("src.reranker.scorer.CrossEncoderModelRegistry")
    def test_scorer_inference_and_sigmoid(self, mock_registry):
        # Mock cross-encoder predict returning logits
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.5, -2.0]
        mock_registry.get_model.return_value = mock_model
        
        config = RerankerConfig(model_name="test-model", batch_size=32)
        scorer = BatchRerankingScorer(config)
        
        pairs = [
            ("CAND_001", "query", "Alice doc"),
            ("CAND_002", "query", "Bob doc")
        ]
        
        scores = scorer.score_pairs(pairs)
        
        self.assertEqual(len(scores), 2)
        self.assertIn("CAND_001", scores)
        
        logit1, prob1 = scores["CAND_001"]
        logit2, prob2 = scores["CAND_002"]
        
        self.assertEqual(logit1, 1.5)
        self.assertAlmostEqual(prob1, 1.0 / (1.0 + 2.718281828 ** -1.5), places=4)
        self.assertEqual(logit2, -2.0)
        self.assertAlmostEqual(prob2, 1.0 / (1.0 + 2.718281828 ** 2.0), places=4)

    def test_explainability_evidence(self):
        engine = ExplainabilityEngine()
        
        # Alice check
        cand_alice = dict(self.mock_candidates[0])
        evidence_alice = engine.extract_evidence(cand_alice)
        
        self.assertTrue(evidence_alice["Matched Skills"])
        self.assertTrue(evidence_alice["Matched AI Experience"])
        self.assertTrue(evidence_alice["Matched Career Experience"])
        self.assertTrue(evidence_alice["Matched Company Context"]) # Noida local
        
        # Bob check
        cand_bob = dict(self.mock_candidates[1])
        evidence_bob = engine.extract_evidence(cand_bob)
        self.assertFalse(evidence_bob["Matched Skills"])
        self.assertFalse(evidence_bob["Matched AI Experience"])
        self.assertFalse(evidence_bob["Matched Company Context"]) # office, delhi

    def test_feature_engineering(self):
        manager = FeatureEngineeringManager()
        
        # Let's mock top candidates with CE scores
        candidates_with_ce = []
        for c in self.mock_candidates:
            cc = dict(c)
            cc["cross_encoder_logit"] = 1.0
            cc["cross_encoder_probability"] = 0.73
            cc["explainability_evidence"] = {
                "Matched Leadership": True,
                "Matched AI Experience": True,
                "Matched Search/Retrieval Experience": False,
                "Matched Responsibilities": True
            }
            candidates_with_ce.append(cc)
            
        df_features = manager.compile_features(candidates_with_ce)
        
        self.assertEqual(len(df_features), 2)
        self.assertIn("cross_encoder_score", df_features.columns)
        self.assertIn("query_coverage", df_features.columns)
        self.assertIn("matched_skills_count", df_features.columns)
        self.assertEqual(df_features.loc[0, "matched_leadership_signals"], 1)
        self.assertEqual(df_features.loc[0, "matched_ai_signals"], 1)

    @patch("src.reranker.scorer.CrossEncoderModelRegistry")
    def test_end_to_end_cli_pipeline(self, mock_registry):
        # Mock cross encoder model
        mock_model = MagicMock()
        mock_model.predict.return_value = [2.0, -1.0]
        mock_registry.get_model.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save mock inputs
            spec_path = os.path.join(tmp_dir, "spec.json")
            retrieval_path = os.path.join(tmp_dir, "retrieval.parquet")
            search_docs_path = os.path.join(tmp_dir, "search_docs.parquet")
            
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(self.mock_spec, f)
                
            pd.DataFrame(self.mock_candidates).to_parquet(retrieval_path)
            self.mock_search_docs.to_parquet(search_docs_path)
            
            config = RerankerConfig(top_candidates=10)
            
            # Execute pipeline orchestration function
            run_reranking_pipeline(
                retrieval_path=retrieval_path,
                search_docs_path=search_docs_path,
                spec_path=spec_path,
                output_dir=tmp_dir,
                config=config
            )
            
            # Verify outputs
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "reranked_candidates.parquet")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "reranker_features.parquet")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "reranker_scores.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "reranker_statistics.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "reranker_benchmark.json")))
            
            # Verify score preservation
            df_reranked = pd.read_parquet(os.path.join(tmp_dir, "reranked_candidates.parquet"))
            self.assertEqual(len(df_reranked), 2)
            self.assertIn("cross_encoder_logit", df_reranked.columns)
            self.assertIn("cross_encoder_probability", df_reranked.columns)
            self.assertIn("retrieval_score", df_reranked.columns)
            self.assertIn("rrf_score", df_reranked.columns)

if __name__ == "__main__":
    unittest.main()
