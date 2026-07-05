---
title: Candidate Ranking Engine
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Intelligent Candidate Discovery & Ranking System

An offline, CPU-optimized, production-grade search and ranking pipeline built for large-scale information retrieval, semantic matching, and deterministic candidate verification. 

The system processes raw candidate resume corpora, extracts hiring specifications from unstructured job descriptions, generates vector representations, retrieves semantic matches using FAISS, re-ranks candidates with a deep Cross-Encoder model, and applies a multi-factor hybrid scorer with timeline anomaly (honeypot) checks and regex validation to output a verified Top 100 candidate ranking.

---

## 🚀 Key Capabilities & Pipeline Stages

The system consists of eight distinct sequential phases, coordinated via root-level runner scripts:

```
Raw Resumes (JSONL)           Unstructured JD (DOCX/TXT)
       │                                   │
       ▼                                   ▼
┌──────────────┐                   ┌──────────────┐
│   Phase 1    │                   │   Phase 2    │
│Preprocessing │                   │JD Intel Parse│
└──────┬───────┘                   └──────┬───────┘
       │                                   │
       ▼                                   │
┌──────────────┐                   ┌───────┴──────┐
│   Phase 3    │                   │   Phase 5    │
│Search Doc Gen│                   │  RRF Fusion  │
└──────┬───────┘                   │  Retrieval   │
       │                           └──────┬───────┘
       ▼                                  │
┌──────────────┐                          │
│   Phase 4    │                          │
│FAISS Indexing│                          │
└──────┬───────┘                          │
       │                                  │
       └─────────────────┬────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   Phase 6    │
                  │Cross-Encoder │
                  │  Re-ranking  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   Phase 7    │
                  │Hybrid Scorer │
                  │& Honeypot Det│
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   Phase 8    │
                  │Deterministic │
                  │ Verification │
                  └──────┬───────┘
                         │
                         ▼
                   submission.csv
```

### 1. Phase 1: Candidate Preprocessing Module
*   **Ingestion:** Safe streaming JSONL data parser handling memory constraints.
*   **Normalization:** Cleans text, corrects inverted min/max salary boundaries, standardizes chronological date records against a reference date (`2026-06-30`), and maps raw skills to a normalized alias dictionary.
*   **Enrichment:** Computes non-overlapping net career tenure, relevant AI/ML experience years, AI skill ratio, and job hopping thresholds.
*   **Output:** Generates clean parquet files partitioned by schema structures.

### 2. Phase 2: Job Description Intelligence Module
*   **Parsing:** Extracts structured properties from DOCX/TXT briefs using Pydantic models.
*   **Taxonomy Mapping:** Matches skill requirements to canonical dictionary mappings.
*   **Query Synthesis:** Computes six distinct query profile strings (Role, Tech Stack, AI/ML focus, Leadership, Domain context, and Exclusions) to feed semantic search retrieval.

### 3. Phase 3: Candidate Search Document Generation
*   **Compilation:** Concatenates normalized title profiles, chronological work summaries, education, certifications, and skills into clean markdown-like narratives.
*   **Compaction:** Enforces a hard 1024-token tokenizer limit, trimming older work history experiences first to maintain chronological integrity when profiles exceed context boundaries.

### 4. Phase 4: Embedding & Vector Indexing Module
*   **Embeddings:** Computes dense vector representations using pre-trained sentence encoders (`BAAI/bge-small-en-v1.5`) with configurable backends (HuggingFace Transformers, ONNX Runtime, or SentenceTransformers).
*   **Index:** Builds L2-normalized FAISS Flat Inner-Product index files for rapid dot-product cosine similarity searches.
*   **Caching:** Implements a disk-based cache keyed by SHA-256 hashes of input texts and model weights to avoid redundant calculations.

### 5. Phase 5: Semantic Candidate Retrieval Engine
*   **Search:** Batch queries the FAISS vector index across the 6 synthesized query dimensions.
*   **Fusion:** Merges ranked result lists using Reciprocal Rank Fusion (RRF).
*   **Filters:** Applies soft constraints (notice period penalty decay, salary limits, minimum experience bounds) to output a filtered candidate parquet pool.

### 6. Phase 6: Cross-Encoder Re-ranking Engine
*   **Re-ranking:** Jointly evaluates candidate search documents against the full recruiter specification using a sequence classification model (`BAAI/bge-reranker-base`).
*   **Explainability:** Generates 12 ranking features (CE similarity score, keyword overlap, score quantiles, rank steps) and captures evidence maps for transparency.

### 7. Phase 7: Final Hybrid Ranking Engine
*   **Scoring:** Computes a Unified Hybrid Score fusing retrieval scores (20%), Cross-Encoder scores (40%), career progression quality (25%), and profile completeness checks (15%).
*   **Honeypot Detection:** Identifies fake timelines (e.g. claiming experience with recent tools like GPT-4 or Llama in job slots ending years before release) to disqualify suspicious applicants.

### 8. Phase 8: Candidate Verification Engine
*   **Phrase Scanner:** Runs regex checks to boost hands-on developers and penalize management/recruiter profiles.
*   **Verification:** Asserts must-have skill compliance using synonymous dictionary keys (e.g. checking PyTorch/TensorFlow to verify Python proficiency).
*   **Calibrator:** Recalculates final scores, constructs dynamic metadata-injected reasoning descriptions, asserts quality validation rules (exactly 100 rows, no failed candidates inside the Top 100 list), and outputs the official `submission.csv`.

---

## 📂 Repository File Structure

```
├── data/                        # Project input files (ignored by default)
│   ├── candidates.jsonl         # Raw candidate profiles (large dataset)
│   ├── job_description.docx     # Job description spec sheet
│   ├── candidate_schema.json    # Candidate structural validation schema
│   ├── sample_candidates.json   # 50 sample profiles for dry runs
│   ├── sample_submission.csv    # Sample submission formatting template
│   └── validate_submission.py   # Utility: Asserts formatting correctness of submission.csv
│
├── output/                      # Ignored by default (except deliverables)
│   ├── submission.csv           # Verified Top 100 candidates submission format
│   └── robustness_report.json   # Sensitivity check output metrics
│
├── src/                         # System core modules
│   ├── document_generation/     # Phase 3 Narrative compilers & compactor
│   ├── evaluation/              # Standalone Robustness & Sensitivity Suite
│   ├── indexing/                # Phase 4 Vector computation & FAISS setup
│   ├── jd_intelligence/         # Phase 2 Unstructured JD parser & schemas
│   ├── preprocessing/           # Phase 1 Streaming normalizers & estimators
│   ├── ranking/                 # Phase 7 Hybrid scorer & honeypot scans
│   ├── reranker/                # Phase 6 Cross-Encoder neural re-ranker
│   ├── retrieval/               # Phase 5 Multi-query FAISS search & RRF
│   └── verification/            # Phase 8 Verification calibrator & output rules
│
├── tests/                       # Complete unit test suite (Phases 1-8)
│
├── run_preprocessing.py         # Entrypoint: Preprocesses data/candidates.jsonl
├── run_jd_intelligence.py       # Entrypoint: Extracts specification schemas
├── run_document_generation.py   # Entrypoint: Compiles search narrative text
├── run_indexing.py              # Entrypoint: Computes embeddings & FAISS index
├── run_retrieval.py             # Entrypoint: Runs vector search & RRF fusion
├── run_reranker.py              # Entrypoint: Runs batch Cross-Encoder re-ranker
├── run_ranking.py               # Entrypoint: Runs hybrid scoring & honeypot sweeps
├── run_verification.py          # Entrypoint: Runs verification & exports Top 100
└── .gitignore                   # Excludes large binaries (*.parquet, *.npy, *.index)
```

---

## 🛠️ Installation & Setup

### Requirements
*   Python 3.10+
*   CPU Execution (No CUDA required, fully optimized for offline resource constraints)
*   Memory: $\le 16$ GB RAM

### Environment Setup
1. Clone the repository to your local workspace:
   ```bash
   git clone <repository_url>
   cd India_runs_data_and_ai_challenge
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scipy scopt scikit-learn torch transformers sentence-transformers faiss-cpu pydantic python-docx pyarrow psutil
   ```

---

## 🔄 End-to-End Pipeline Execution

Follow these steps to execute the pipeline stages in sequence.

### Step 1: Preprocess Candidates
Cleans raw resumes and builds preprocessed parquet records:
```bash
python run_preprocessing.py --candidates data/candidates.jsonl --schema data/candidate_schema.json --output-dir output --batch-size 10000
```

### Step 2: Parse Job Description
Extracts structures from unstructured JDs:
```bash
python run_jd_intelligence.py --jd data/job_description.docx --output-dir output
```

### Step 3: Generate Search Documents
Compiles narrative profiles and trims text to fit within token boundaries:
```bash
python run_document_generation.py --preprocessed output/processed_candidates.parquet --output-dir output --batch-size 10000
```

### Step 4: Run FAISS Vector Indexing
Computes embeddings for all candidates and builds the search index:
```bash
python run_indexing.py --input-dir output --output-dir output --model BAAI/bge-small-en-v1.5 --backend sentence_transformers --batch-size 1024
```

### Step 5: Execute Semantic Candidate Retrieval
Retrieves candidates and applies notice period/salary filters:
```bash
python run_retrieval.py --jd output/hiring_specification.json --queries output/search_queries.json --index output/faiss.index --index-metadata output/index_metadata.json --metadata output/processed_candidates.parquet --output output --top-k 200 --min-exp 5.0 --max-notice 60
```

### Step 6: Cross-Encoder Neural Re-ranking
Performs joint Cross-Encoder scoring on the retrieved pool:
```bash
python run_reranker.py --retrieval output/retrieval_candidates.parquet --search-docs output/search_documents.parquet --jd output/hiring_specification.json --output output --model BAAI/bge-reranker-base --batch-size 32
```

### Step 7: Run Hybrid Ranking
Aggregates scoring weights and runs honeypot validation checks:
```bash
python run_ranking.py --reranked output/reranked_candidates.parquet --reranker-features output/reranker_features.parquet --processed output/processed_candidates.parquet --jd output/hiring_specification.json --output output
```

### Step 8: Deterministic Verification & CSV Export
Runs contextual filters, validates skill sets, asserts row limits, and writes the submission file:
```bash
python run_verification.py --final-candidates output/final_ranked_candidates.parquet --processed output/processed_candidates.parquet --jd output/hiring_specification.json --output output
```

---

## 🧪 Testing and Verification

### Unit Tests
The codebase is fully covered by unit tests located under `tests/`. Run all validation suites recursively:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Submission Validation Utility
Asserts that the exported `submission.csv` is correctly structured:
```bash
python data/validate_submission.py --submission output/submission.csv
```

---

## 📈 Pipeline Robustness & Sensitivity Checks

To ensure the system doesn't experience **Pipeline / Heuristic Overfitting** (e.g. over-tuning weights and regex filters to one specific job description), we provide a standalone diagnostic checking script:

```bash
python src/evaluation/robustness_check.py
```

It executes:
1.  **Weight Sensitivity Analysis:** Perturbs Phase 7 weights by $\pm 2\%$ and asserts rank stability using the **Spearman Rank Correlation** ($r_s$) against the baseline.
2.  **Context Regex Stress Test:** Evaluates Phrase Scanners against synthetic edge-case recruiter vs. developer sentences.
3.  **Hiring Spec Cross-Validation:** Runs mock queries against a mock "Senior Data Engineer" specification to ensure search results follow normal distributions.

Output results are exported directly to `output/robustness_report.json`.

---

## 🖥️ Interactive Web Dashboard (Phase 9)

🚀 **Live Hugging Face Space Application:** [Candidate Ranking Engine App](https://durgesh1234xyz-candidate-ranking-engine.hf.space/)

An interactive, responsive Streamlit dashboard is provided under `app.py` to wrap the offline recruitment pipeline in a rich graphical interface optimized for 16GB RAM Hugging Face Spaces.

### ⚙️ Core Dashboard Capabilities
*   **Dual Execution Modes:**
    *   *Search 100K Candidate Database:* Performs fast vector searches against the pre-indexed FAISS database and preprocessed Parquet tables.
    *   *Screen Uploaded Resumes (Interactive Screener):* Dynamically parses live PDF, DOCX, or TXT resume uploads, extracts text, computes similarities, and classification scores in real-time.
*   **Model Caching Strategy:** Implements CPU-enforced singleton caching of `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base` models inside RAM via `@st.cache_resource` to prevent memory leaks and Out-Of-Memory (OOM) failures.
*   **Dynamic Visual Scorecards:** Instantly renders candidate match metrics cards, dynamic AI reasoning reports (incorporating fresher fallback and custom logic branching), and error banners if timeline honeypots or fraud indicators are detected.
*   **Hugging Face Spaces Guardrails:** Integrated index availability checkers automatically prompt the user with instructions if `faiss.index` or candidate parquets are missing from the environment.

### Running the App Locally:
1. Ensure all packages are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the local server:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to the local URL (usually `http://localhost:8501`).

---

## 🛡️ License

This project is licensed under the MIT License:

```text
MIT License

Copyright (c) 2026 Durgesh Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
