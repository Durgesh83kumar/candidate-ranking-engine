import unittest
import numpy as np
import tempfile
import os
import json

from src.indexing.cache import EmbeddingCache
from src.indexing.faiss_index import FAISSIndexBuilder
from src.indexing.searcher import VectorSearcher
from src.indexing.evaluator import QualityEvaluator
from src.indexing.benchmarker import IndexingBenchmarker
from src.indexing.config import IndexingConfig

class TestVectorIndexing(unittest.TestCase):

    def setUp(self):
        # 3 mock vectors, dimension 4
        self.mock_embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.707, 0.707, 0.0, 0.0]
        ], dtype=np.float32)

    def test_cache_layer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = EmbeddingCache(tmp_dir)
            text = "Senior Python Developer"
            model = "bge-small"
            version = "v2"
            
            h_key = cache.generate_hash(text, model, version)
            
            # Cache miss
            self.assertIsNone(cache.get(h_key))
            
            # Cache set
            vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            cache.set(h_key, vector)
            
            # Cache hit
            cached_vec = cache.get(h_key)
            self.assertIsNotNone(cached_vec)
            np.testing.assert_array_almost_equal(cached_vec, vector)

    def test_faiss_index_and_search(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = FAISSIndexBuilder(dimension=4)
            builder.build_and_populate(self.mock_embeddings)
            
            # Save index
            index_file = os.path.join(tmp_dir, "faiss.index")
            builder.save_index(index_file)
            self.assertTrue(os.path.exists(index_file))
            
            # Mapping candidate IDs
            meta_file = os.path.join(tmp_dir, "index_metadata.json")
            mapping_data = {
                "candidate_ids": ["CAND_0000001", "CAND_0000002", "CAND_0000003"]
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f)
                
            # Test Searcher
            searcher = VectorSearcher()
            searcher.load_index(index_file, meta_file)
            
            # Search query vector
            q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            results = searcher.retrieve_top_k(q_vec, k=2)
            
            # First match should be CAND_0000001 with score 1.0 (exact match)
            self.assertEqual(results[0]["candidate_id"], "CAND_0000001")
            self.assertAlmostEqual(results[0]["score"], 1.0, places=3)
            
            # Second match should be CAND_0000003 with score 0.707
            self.assertEqual(results[1]["candidate_id"], "CAND_0000003")
            self.assertAlmostEqual(results[1]["score"], 0.707, places=3)

    def test_quality_evaluator(self):
        evaluator = QualityEvaluator()
        stats = evaluator.evaluate_vector_statistics(self.mock_embeddings)
        
        self.assertEqual(stats["total_vectors"], 3)
        self.assertEqual(stats["dimension"], 4)
        # Check standard norms are 1.0
        self.assertAlmostEqual(stats["norm"]["mean"], 1.0, places=2)
        # Verify no vectors collapse warning
        self.assertFalse(stats["vector_collapse_warning"])

    def test_benchmarker(self):
        benchmarker = IndexingBenchmarker()
        report = benchmarker.compile_report(100, 1.5, tempfile.gettempdir())
        
        self.assertEqual(report["throughput_docs_per_sec"], 66.67)
        self.assertEqual(report["elapsed_time_seconds"], 1.5)
        self.assertIn("disk_footprint", report)

if __name__ == "__main__":
    unittest.main()
