import unittest
import numpy as np
import tempfile
import os
import json
import pandas as pd

from src.retrieval.config import RetrievalConfig
from src.retrieval.query_builder import MultiQueryBuilder
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.filters import SoftFiltersEvaluator
from src.retrieval.deduplicator import CandidateDeduplicator
from src.retrieval.metadata import EnrichmentJoiner
from src.retrieval.retriever import SemanticRetriever

class TestRetrievalEngine(unittest.TestCase):

    def setUp(self):
        # Setup mock specifications
        self.mock_spec = {
            "role": {"title": "Senior AI Engineer", "seniority": "senior"},
            "experience": {"min_years": 5.0},
            "skills": {
                "must_have": [{"name": "large_language_models"}, {"name": "python"}],
                "preferred": []
            },
            "domains": {"required": ["NLP"], "preferred": []},
            "responsibilities": [{"description": "Own intelligence layer", "category": "architecture"}],
            "preferences": {
                "industries": ["HR-Tech"],
                "negative_preferences": {
                    "technologies": ["hadoop"],
                    "roles": ["consulting"]
                }
            }
        }
        
        self.mock_queries = {
            "primary_query": "Senior AI Engineer NLP",
            "technology_query": "python large_language_models",
            "concept_query": "vector databases serving RAG",
            "skill_query": "python"
        }
        
        # Candidate mock row data
        self.mock_candidate_row = {
            "candidate_id": "CAND_TEST_001",
            "profile": {
                "anonymized_name": "Test Candidate",
                "current_title_normalized": "AI Developer",
                "current_company": "Acme Corp",
                "location_normalized": "San Francisco",
                "country_normalized": "USA",
                "years_of_experience_calculated": 6.0,
                "summary": "Experienced python engineer building large language models"
            },
            "redrob_signals": {
                "notice_period_days": 15,
                "open_to_work_flag": True,
                "github_activity_score": 85.0,
                "profile_completeness_score": 95.0,
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "expected_salary_range_inr_lpa": {"min": 25.0, "max": 35.0}
            },
            "skills": ["python", "large_language_models", "vector_databases"],
            "career_history": [{"job_title": "AI Dev", "description": "built serving models"}],
            "education": [],
            "certifications": [],
            "languages": [],
            "search_document_v2": "python developer building large language models serving systems",
            "search_document_v1": "summary skills career",
            "search_vector_text": "text"
        }

    def test_query_builder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path = os.path.join(tmp_dir, "spec.json")
            queries_path = os.path.join(tmp_dir, "queries.json")
            
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(self.mock_spec, f)
            with open(queries_path, "w", encoding="utf-8") as f:
                json.dump(self.mock_queries, f)
                
            builder = MultiQueryBuilder(spec_path, queries_path)
            queries = builder.build_queries()
            
            # Check 6 queries are generated
            self.assertEqual(len(queries), 6)
            self.assertIn("general", queries)
            self.assertIn("negative", queries)
            self.assertEqual(queries["negative"], "hadoop consulting")
            
            negatives = builder.get_negative_keywords()
            self.assertIn("hadoop", negatives)

    def test_rrf_fusion(self):
        # 2 queries, candidate A is rank 1 and 2, candidate B is rank 2 and 1
        query_results = {
            "general": [
                {"candidate_id": "CAND_A", "score": 0.9},
                {"candidate_id": "CAND_B", "score": 0.8}
            ],
            "technical": [
                {"candidate_id": "CAND_B", "score": 0.95},
                {"candidate_id": "CAND_A", "score": 0.85}
            ]
        }
        
        fusion = ReciprocalRankFusion(k=60)
        fused = fusion.compute_rrf(query_results)
        
        self.assertEqual(len(fused), 2)
        # RRF for CAND_A: 1/(60+1) + 1/(60+2)
        # RRF for CAND_B: 1/(60+2) + 1/(60+1)
        # Since they have same ranks, they should have equal scores
        self.assertAlmostEqual(fused[0]["rrf_score"], fused[1]["rrf_score"], places=4)

    def test_soft_filters(self):
        # Relocation check, experience check, country check
        filter_prefs = {
            "min_experience": 5.0,
            "preferred_country": "USA",
            "salary_max_lpa": 40.0
        }
        
        evaluator = SoftFiltersEvaluator(filter_prefs)
        
        # Test candidate passing
        cand_pass = {
            "years_of_experience": 6.0,
            "country": "USA",
            "expected_salary_lpa": 30.0,
            "is_excluded": False
        }
        failed, penalty = evaluator.evaluate_candidate(cand_pass)
        self.assertEqual(len(failed), 0)
        self.assertEqual(penalty, 1.0)
        
        # Test candidate failing soft filters (should apply penalty, not drop)
        cand_fail = {
            "years_of_experience": 4.0,
            "country": "Canada",
            "expected_salary_lpa": 50.0,
            "is_excluded": False
        }
        failed, penalty = evaluator.evaluate_candidate(cand_fail)
        self.assertIn("Minimum Experience", failed)
        self.assertIn("Preferred Country", failed)
        self.assertIn("Salary Cap", failed)
        self.assertTrue(penalty < 1.0)

    def test_deduplicator(self):
        candidates = [
            {
                "candidate_id": "CAND_A",
                "rrf_score": 0.05,
                "matched_queries": ["general"],
                "matched_profile_sections": ["Matched Summary"],
                "query_similarities": {"general": 0.8},
                "query_ranks": {"general": 1}
            },
            {
                "candidate_id": "CAND_A",
                "rrf_score": 0.03,
                "matched_queries": ["technical"],
                "matched_profile_sections": ["Matched Skills"],
                "query_similarities": {"technical": 0.75},
                "query_ranks": {"technical": 2}
            }
        ]
        
        dedup = CandidateDeduplicator()
        unique = dedup.deduplicate(candidates)
        
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["rrf_score"], 0.05) # Keeps highest RRF
        self.assertIn("general", unique[0]["matched_queries"])
        self.assertIn("technical", unique[0]["matched_queries"])
        self.assertIn("Matched Summary", unique[0]["matched_profile_sections"])
        self.assertIn("Matched Skills", unique[0]["matched_profile_sections"])

    def test_metadata_enrichment_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = os.path.join(tmp_dir, "candidates.parquet")
            
            # Save mock df
            df = pd.DataFrame([self.mock_candidate_row])
            df.to_parquet(parquet_path)
            
            joiner = EnrichmentJoiner(parquet_path)
            fused_list = [{
                "candidate_id": "CAND_TEST_001",
                "rrf_score": 0.08,
                "query_similarities": {"general": 0.85},
                "query_ranks": {"general": 1}
            }]
            
            enriched = joiner.join_and_enrich(
                fused_results=fused_list,
                queries={"general": "python language model serving"},
                negative_keywords=["hadoop"]
            )
            
            self.assertEqual(len(enriched), 1)
            cand = enriched[0]
            self.assertEqual(cand["years_of_experience"], 6.0)
            self.assertEqual(cand["expected_salary_lpa"], 30.0) # Average of range
            
            # Check match evidence tracking
            self.assertIn("Matched Summary", cand["matched_profile_sections"])
            self.assertIn("Matched Skills", cand["matched_profile_sections"])
            self.assertIn("Matched Career History", cand["matched_profile_sections"])

if __name__ == "__main__":
    unittest.main()
