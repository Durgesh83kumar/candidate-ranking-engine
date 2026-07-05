import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.indexing.config import IndexingConfig
from src.indexing.backends import get_backend
from src.indexing.exceptions import IndexingError
from src.indexing.cache import EmbeddingCache
from src.indexing.tracker import ExperimentTracker
from src.indexing.evaluator import QualityEvaluator
from src.indexing.benchmarker import IndexingBenchmarker
from src.indexing.faiss_index import FAISSIndexBuilder

def run_indexing_pipeline(config: IndexingConfig, input_dir: str, output_dir: str) -> None:
    """Orchestrates candidate vector generation, caching, FAISS indexing, evaluation, and tracking."""
    benchmarker = IndexingBenchmarker()
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    docs_path = os.path.join(input_dir, "search_documents.parquet")
    meta_path = os.path.join(input_dir, "candidate_metadata.parquet")
    
    if not os.path.exists(docs_path):
        raise IndexingError(f"Search documents file not found: {docs_path}. Please run Phase 3 first.")
        
    # 1. Load Data
    print(f"Loading search documents from {docs_path}...")
    df_docs = pd.read_parquet(docs_path)
    
    doc_col = config.document_version
    if doc_col not in df_docs.columns:
        raise IndexingError(f"Configured document column '{doc_col}' not found in search_documents.parquet.")
        
    candidates = df_docs.to_dict(orient="records")
    total_records = len(candidates)
    print(f"Loaded {total_records} candidates for embedding index generation.")

    # 2. Initialize Cache
    cache_dir = os.path.join(output_dir, "embedding_cache")
    cache = EmbeddingCache(cache_dir)
    
    # 3. Load Pluggable Backend with Fallback Chain
    backend = None
    selected_backend = config.backend
    backends_to_try = [selected_backend, "sentence_transformers", "hf"]
    # Ensure unique list
    backends_to_try = list(dict.fromkeys(backends_to_try))
    
    for b_name in backends_to_try:
        try:
            backend = get_backend(b_name)
            backend.initialize(config.model_name, {"workers": 4})
            print(f"Successfully initialized embedding backend: {b_name}")
            config.backend = b_name  # Update actual backend run in config
            break
        except Exception as e:
            print(f"Warning: Failed to load backend '{b_name}' due to error: {str(e)}. Attempting fallback...")
            
    if backend is None:
        raise IndexingError("Critical Error: All embedding backends failed to initialize. Aborting pipeline.")

    # 4. Generate/Retrieve Embeddings
    print("Generating embeddings in batches...")
    embeddings_list = []
    candidate_ids_mapping = []
    
    cache_hits = 0
    cache_misses = 0
    
    batch_size = config.batch_size
    for i in range(0, total_records, batch_size):
        chunk = candidates[i:i + batch_size]
        
        chunk_texts_to_embed = []
        chunk_miss_indices = []
        chunk_vectors = [None] * len(chunk)
        
        for idx, record in enumerate(chunk):
            cid = record.get("candidate_id")
            text = record.get(doc_col, "")
            candidate_ids_mapping.append(cid)
            
            h_key = cache.generate_hash(text, config.model_name, config.document_version)
            
            cached_vec = None
            if config.cache_enabled:
                cached_vec = cache.get(h_key)
                
            if cached_vec is not None:
                chunk_vectors[idx] = cached_vec
                cache_hits += 1
            else:
                chunk_texts_to_embed.append(text)
                chunk_miss_indices.append((idx, h_key))
                cache_misses += 1
                
        # Batch inference on cache misses
        if chunk_texts_to_embed:
            # Huggingface/SentenceTransformers expects query/passage prefix check
            processed_texts = chunk_texts_to_embed
            if "e5" in config.model_name.lower():
                # E5 passage prefix
                processed_texts = [f"passage: {t}" for t in chunk_texts_to_embed]
                
            new_vectors = backend.compute_embeddings(processed_texts)
            
            for (idx, h_key), vec in zip(chunk_miss_indices, new_vectors):
                chunk_vectors[idx] = vec
                if config.cache_enabled:
                    cache.set(h_key, vec)
                    
        for vec in chunk_vectors:
            embeddings_list.append(vec)
            
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
    print(f"Completed vector calculations. Cache Hits: {cache_hits}, Cache Misses: {cache_misses}.")

    # 5. Build and serialize FAISS index
    dim = embeddings_matrix.shape[1]
    index_builder = FAISSIndexBuilder(dimension=dim)
    index_builder.build_and_populate(embeddings_matrix)
    
    faiss_path = os.path.join(output_dir, "faiss.index")
    index_builder.save_index(faiss_path)

    # 6. Save raw numpy matrix
    npy_path = os.path.join(output_dir, "embeddings.npy")
    np.save(npy_path, embeddings_matrix)

    # 7. Save index candidate mappings
    index_metadata = {
        "model_name": config.model_name,
        "dimension": dim,
        "document_version": config.document_version,
        "total_records": total_records,
        "candidate_ids": candidate_ids_mapping
    }
    with open(os.path.join(output_dir, "index_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(index_metadata, f, indent=2)

    elapsed_time = time.time() - start_time

    # 8. Evaluation
    eval_metrics = {}
    if config.eval_enabled:
        print("Running quality evaluations...")
        evaluator = QualityEvaluator()
        stats = evaluator.evaluate_vector_statistics(embeddings_matrix)
        
        # Save embedding statistics
        with open(os.path.join(output_dir, "embedding_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
            
        # Run retrieval sanity test
        from src.indexing.searcher import VectorSearcher
        mock_searcher = VectorSearcher(backend_name=config.backend, model_name=config.model_name)
        mock_searcher.index_builder = index_builder
        mock_searcher.candidate_ids = candidate_ids_mapping
        
        # Load candidate metadata for recall check
        meta_records = []
        if os.path.exists(meta_path):
            try:
                meta_records = pd.read_parquet(meta_path).to_dict(orient="records")
            except Exception:
                pass
                
        test_results = evaluator.run_retrieval_tests(mock_searcher, meta_records)
        eval_metrics = {
            "retrieval_evaluation": test_results,
            "statistics": stats
        }
        with open(os.path.join(output_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)

    # 9. Benchmarking
    bench_report = {}
    if config.benchmark_enabled:
        bench_report = benchmarker.compile_report(total_records, elapsed_time, output_dir)
        bench_report["cache_hits"] = cache_hits
        bench_report["cache_misses"] = cache_misses
        with open(os.path.join(output_dir, "benchmark_report.json"), "w", encoding="utf-8") as f:
            json.dump(bench_report, f, indent=2)

    # 10. Experiment Tracking
    tracker = ExperimentTracker(output_dir)
    exp_metrics = {
        "throughput_docs_per_sec": bench_report.get("throughput_docs_per_sec", 0.0),
        "elapsed_time_seconds": elapsed_time,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses
    }
    if eval_metrics:
        exp_metrics["recall_at_10"] = eval_metrics.get("retrieval_evaluation", {}).get("recall_at_10", 0.0)
        exp_metrics["duplicate_vectors"] = eval_metrics.get("statistics", {}).get("duplicate_vectors_found", 0)
        
    tracker.log_run(config.to_dict(), exp_metrics)

    print("\nVector Indexing Pipeline Completed Successfully!")
    print("--------------------------------------------------")
    print(f"FAISS index written: {faiss_path}")
    print(f"Raw embeddings:      {npy_path}")
    print(f"Build time:          {elapsed_time:.2f} seconds")
    print(f"Throughput rate:     {total_records / elapsed_time:.2f} doc/sec")
    print("--------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Run Embedding & Vector Indexing pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output",
        help="Directory containing search documents parquet."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save index, matrix, caches and registries."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace model repo to load."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sentence_transformers",
        help="Inference engine backend ('onnx', 'sentence_transformers', 'hf')."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for vector computation."
    )
    parser.add_argument(
        "--doc-version",
        type=str,
        default="search_document_v2",
        help="Column inside search_documents.parquet to index."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable vector cache layer."
    )
    
    args = parser.parse_args()
    
    config = IndexingConfig(
        model_name=args.model,
        backend=args.backend,
        batch_size=args.batch_size,
        document_version=args.doc_version,
        cache_enabled=not args.no_cache
    )
    
    try:
        run_indexing_pipeline(config, args.input_dir, args.output_dir)
    except Exception as e:
        print(f"Indexing pipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
